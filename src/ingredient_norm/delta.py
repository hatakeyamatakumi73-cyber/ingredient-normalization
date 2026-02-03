from sentence_transformers import SentenceTransformer, util, CrossEncoder
from rapidfuzz.distance import JaroWinkler
import unicodedata, re, networkx as nx
import numpy as np
import pickle
import pandas as pd
import os
from collections import defaultdict, Counter
import ast
import math
from sudachipy import dictionary
from sudachipy import tokenizer as sudachi_tokenizer


def unify_once2(file_name):
    pair_set = set()
    print("unify_once開始", flush=True)

    base = os.path.dirname(__file__)
    csv_path = os.path.abspath(os.path.join(base, "..", "datasets", f"synonym_ners{file_name}.csv"))
    pickle_path = os.path.abspath(os.path.join(base, "..", "datasets", f"labels{file_name}.pickle"))

    print("読み込みパス:", csv_path, flush=True)
    print("labels pickle:", pickle_path, flush=True)

    # -------
    # helpers
    # -------
    def jw(a, b):
        return JaroWinkler.normalized_similarity(a, b)

    tok = dictionary.Dictionary().create()
    SPLIT_MODE = sudachi_tokenizer.Tokenizer.SplitMode.C

    def normalize_base(s: str) -> str:
        s = str(s)
        s = unicodedata.normalize("NFKC", s)
        s = re.split(r"[・/／,、\s]+", s)[0]
        s = re.sub(r"[『』「」\"'`]", "", s)
        return s

    def get_reading(text: str) -> str:
        ms = tok.tokenize(text, SPLIT_MODE)
        out = []
        for m in ms:
            y = m.reading_form()
            if y and y != "*":
                out.append(y)
            else:
                out.append(m.surface())
        return "".join(out)

    def ensure_list(x):
        if isinstance(x, list):
            return x
        if pd.isna(x):
            return []
        if isinstance(x, str):
            try:
                v = ast.literal_eval(x)
                return v if isinstance(v, list) else [str(v)]
            except Exception:
                return [x]
        return [str(x)]

    def build_bi_input(nm, labs):
        return nm + " " + " ".join(labs) if isinstance(labs, list) and labs else nm

    # -----------------
    # 0) labels 読み込み
    # -----------------
    labels_df = pd.read_pickle(pickle_path)
    labels_df = labels_df.copy()
    labels_df["name"] = labels_df["name"].astype(str)
    labels_df["labels"] = labels_df["labels"].apply(lambda x: x if isinstance(x, list) else ensure_list(x))

    # -------------------------
    # 1) syno 読み込み / 初期生成
    # -------------------------
    if os.path.exists(csv_path):
        syno = pd.read_csv(csv_path, encoding="utf-8-sig")
        syno.columns = [c.strip() for c in syno.columns]
        if "name" not in syno.columns and "ner_name" in syno.columns:
            syno = syno.rename(columns={"ner_name": "name"})
        if "labels" not in syno.columns:
            syno["labels"] = [[]] * len(syno)
        if "canonical" not in syno.columns:
            syno["canonical"] = "none"
        syno["name"] = syno["name"].astype(str)
        syno["labels"] = syno["labels"].apply(ensure_list)
        print("既存 synonym_ners を読み込み:", len(syno), flush=True)
    else:
        syno = labels_df[["name", "labels"]].copy()
        syno["canonical"] = "none"
        print("初期 synonym_ners を生成:", len(syno), flush=True)

    # --------------
    # 2) 語彙（縮約前）
    # --------------
    orig_terms0 = syno["name"].tolist()
    terms0 = [normalize_base(t) for t in orig_terms0]
    readings0 = [get_reading(t) for t in terms0]
    term_labels0 = syno["labels"].tolist()
    terms_for_bi0 = [build_bi_input(t, labs) for t, labs in zip(terms0, term_labels0)]

    # -------------------------
    # 3) core suffix 自動抽出 → token & idf（縮約前語彙で作る）
    # -------------------------
    SUF_MIN_LEN = 4
    SUF_MAX_LEN = 7
    SUF_MIN_DF = 8
    JW_MIN_FOR_CORE = 0.90

    def is_bad_suffix(s: str) -> bool:
        return len(s) == 0 or s[0] in "ァィゥェォャュョッー"

    suf_df = Counter()
    for r in readings0:
        r = str(r)
        seen = set()
        Lmax = min(SUF_MAX_LEN, len(r))
        for L in range(SUF_MIN_LEN, Lmax + 1):
            suf = r[-L:]
            if not is_bad_suffix(suf):
                seen.add(suf)
        for suf in seen:
            suf_df[suf] += 1

    suffixes = [s for s, c in suf_df.items() if c >= SUF_MIN_DF and not is_bad_suffix(s)]
    suffixes.sort(key=lambda s: (-len(s), -suf_df[s], s))

    def attach_core_suffix(reading: str, max_add=2):
        hits = []
        for suf in suffixes:
            if reading.endswith(suf):
                hits.append(suf)

        if not hits:
            return []

        cores = []
        
        for suf in hits:
            cores.append(suf)
            if len(cores) >= max_add:
                break

        shortest = hits[-1]
        if shortest not in cores:
            cores.append(shortest)

        return cores

    token_lists0 = []
    for r in readings0:
        cores = attach_core_suffix(r, max_add=2)
        token_lists0.append([r] + cores)

    N0 = len(token_lists0)
    df_map = Counter()
    for toks in token_lists0:
        for t in set(toks):
            df_map[t] += 1
    idf_map = {t: math.log((1 + N0) / (1 + c)) + 1.0 for t, c in df_map.items()}

    def core_feature(toksA, toksB):
        best = 0.0
        for a in toksA:
            for b in toksB:
                s = jw(a, b)
                if s < JW_MIN_FOR_CORE:
                    continue
                w = s * min(idf_map.get(a, 0.0), idf_map.get(b, 0.0))
                if w > best:
                    best = w
        return best

    max_idf = max(idf_map.values()) if idf_map else 1.0
    CORE_MAX = max_idf

    def core_norm(x):
        if CORE_MAX <= 0:
            return 0.0
        return float(min(1.0, max(0.0, x / CORE_MAX)))

    # -------------------------
    # 4) reading 完全一致で縮約
    # -------------------------
    def pick_rep(idxs):
        idxs = sorted(idxs)
        best = idxs[0]
        for i in idxs[1:]:
            a = terms0[best]
            b = terms0[i]
            if len(b) < len(a):
                best = i
            elif len(b) == len(a) and b < a:
                best = i
        return best

    parent = list(range(len(orig_terms0)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    rb = defaultdict(list)
    for i, r in enumerate(readings0):
        if r:
            rb[r].append(i)
    for r, idxs in rb.items():
        if len(idxs) <= 1:
            continue
        base_i = idxs[0]
        for j in idxs[1:]:
            union(base_i, j)

    groups = defaultdict(list)
    for i in range(len(orig_terms0)):
        groups[find(i)].append(i)

    rep_of = {}
    for _, idxs in groups.items():
        rep = pick_rep(idxs)
        for i in idxs:
            rep_of[i] = rep

    rep_set = sorted(set(rep_of.values()))
    old2new = {old: new for new, old in enumerate(rep_set)}

    # 縮約後語彙
    orig_terms = [orig_terms0[old] for old in rep_set]
    terms = [terms0[old] for old in rep_set]
    readings = [readings0[old] for old in rep_set]
    term_labels = [term_labels0[old] for old in rep_set]
    terms_for_bi = [build_bi_input(t, labs) for t, labs in zip(terms, term_labels)]

    # 元name -> 縮約代表(base)
    name2rep_base = {orig_terms0[i]: terms[old2new[rep_of[i]]] for i in range(len(orig_terms0))}

    # 縮約後 token_lists
    token_lists = []
    for r in readings:
        cores = attach_core_suffix(r, max_add=2)
        token_lists.append([r] + cores)

    # bucket（先頭文字）
    bucket = defaultdict(list)
    for idx, r in enumerate(readings):
        if r:
            bucket[r[0]].append(idx)

    # -------------------------
    # 5) Bi encode + 候補探索
    # -------------------------
    k = 30
    bi = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    print("Bi-Encoder ベクトル化開始", flush=True)
    emb = bi.encode(terms_for_bi, normalize_embeddings=True, batch_size=256)
    print("terms のベクトル化完了", flush=True)

    cos_topk = util.semantic_search(emb, emb, top_k=k)
    print("cos_topk の長さ:", len(cos_topk), flush=True)

    # ----------------
    # 6) pair_set 構築
    # ----------------
    def add_pair(i, j):
        if i == j:
            return
        a, b = (i, j) if i < j else (j, i)
        if terms[a] == terms[b]:
            return
        pair_set.add((a, b))

    # bi候補
    for i, neighs in enumerate(cos_topk):
        for cand in neighs:
            add_pair(i, cand["corpus_id"])

    # 追加候補：JW(reading) もしくは core_norm が強いもの
    TOP_STR = 20
    STR_MIN = 0.92
    CORE_CAND_MIN = 0.55 

    for i in range(len(terms)):
        r_i = readings[i]
        if not r_i:
            continue
        cands = bucket[r_i[0]]
        best = []
        for j in cands:
            if i == j:
                continue
            if abs(len(terms[j]) - len(terms[i])) > 2:
                continue

            s_jw = jw(readings[i], readings[j])
            s_core = core_norm(core_feature(token_lists[i], token_lists[j]))

            if (s_jw >= STR_MIN) or (s_core >= CORE_CAND_MIN):
                best.append((max(s_jw, s_core), j))

        best.sort(reverse=True)
        for _, j in best[:TOP_STR]:
            add_pair(i, j)
    core2idxs = defaultdict(list)
    for idx, toks in enumerate(token_lists):
        for core in toks[1:]:
            core2idxs[core].append(idx)

    TOP_CORE = 50
    for i in range(len(terms)):
        cores_i = token_lists[i][1:]
        for core in cores_i:
            js = core2idxs.get(core, [])
            if not js:
                continue
            # coreが超頻出だと候補が増えすぎるので上限
            for j in js[:TOP_CORE]:
                if i != j:
                    add_pair(i, j)
    index_pairs = list(pair_set)
    print("pair_set size:", len(index_pairs), flush=True)

    # -------------------------
    # 7) Cross-Encoder
    # -------------------------
    print("Cross-Encoder ロード開始", flush=True)
    ce = CrossEncoder("BAAI/bge-reranker-v2-m3")
    print("Cross-Encoder ロード完了", flush=True)

    pairs = [(terms_for_bi[i], terms_for_bi[j]) for i, j in index_pairs]
    ce_scores = []
    for st in range(0, len(pairs), 16):
        batch = pairs[st:st+16]
        ce_scores.extend(ce.predict(batch))
        if st % (16 * 200) == 0:
            print(f"Cross-Encoder {st}/{len(pairs)}", flush=True)

    # -------------------------
    # 8) スコア融合 → edges（★重み/閾値はそのまま）
    # -------------------------
    alpha, beta, gamma = 0.550, 0.250, 0.200
    THRESH = 0.9420

    edges = []
    for (i, j), ce_s in zip(index_pairs, ce_scores):
        cos_s = util.cos_sim(emb[i], emb[j]).item()

        jw_s   = jw(readings[i], readings[j])
        core_s = core_norm(core_feature(token_lists[i], token_lists[j]))  # 0..1
        s_str = max(jw_s, core_s)

        score = alpha * float(ce_s) + beta * float(cos_s) + gamma * float(s_str)
        if score >= THRESH:
            edges.append((i, j, score))

    print("edges:", len(edges), flush=True)

    # -------------------
    # 9) グラフ → クラスタ
    # -------------------
    G = nx.Graph()
    G.add_nodes_from(range(len(terms)))
    for i, j, w in edges:
        G.add_edge(i, j, weight=w)

    clusters = [sorted(list(comp)) for comp in nx.connected_components(G)]
    print("clusters:", len(clusters), flush=True)

    name_freq = Counter(orig_terms)

    def canon_penalty(s: str) -> int:
        s = str(s)
        p = 0
        if len(s) >= 8: p += 1
        if re.search(r"[ぁ-ん]{4,}", s): p += 1
        if "の" in s: p += 1
        return p

    base2idxs = defaultdict(list)
    for i, t in enumerate(terms):
        base2idxs[t].append(i)

    def choose_canonical_strict(idxs):
        sub = np.vstack([emb[i] for i in idxs])
        sims = util.cos_sim(sub, sub)
        central = sims.sum(dim=1).cpu().numpy()

        best_i = idxs[0]
        best_key = None
        for pos, i in enumerate(idxs):
            surface = orig_terms[i]
            key = (
                name_freq.get(surface, 1),
                -canon_penalty(surface),
                -len(surface),
                central[pos],
            )
            if best_key is None or key > best_key:
                best_key = key
                best_i = i
        return best_i


    def choose_canonical_freq(idxs):

        sub = np.vstack([emb[i] for i in idxs])
        sims = util.cos_sim(sub, sub)
        central = sims.sum(dim=1).cpu().numpy()

        core_counter = Counter()
        for i in idxs:
            for t in set(token_lists[i][1:]):
                core_counter[t] += 1
        common_core = core_counter.most_common(1)[0][0] if core_counter else None

        def freq_penalty(surface: str) -> int:
            p = 0
            if len(surface) >= 10: p += 1
            if "の" in surface: p += 1
            return p

        best_i = idxs[0]
        best_key = None
        for pos, i in enumerate(idxs):
            surface = orig_terms[i]
            toks = token_lists[i]

            has_common_core = 1 if (common_core and common_core in toks) else 0

            key = (
                has_common_core,                
                name_freq.get(surface, 1),    
                -freq_penalty(surface),          
                -len(surface),                
                central[pos],              
            )
            if best_key is None or key > best_key:
                best_key = key
                best_i = i
        return best_i


    canonical_map_strict = {i: terms[i] for i in range(len(terms))}
    canonical_map_freq   = {i: terms[i] for i in range(len(terms))}

    for idxs in clusters:
        root_s = choose_canonical_strict(idxs)
        root_f = choose_canonical_freq(idxs)

        root_base_s = terms[root_s]
        root_base_f = terms[root_f]

        for i in idxs:
            canonical_map_strict[i] = root_base_s
            canonical_map_freq[i]   = root_base_f


    def apply_cluster_canon_with(mapper, nm):
        rep_base = name2rep_base.get(nm, None)
        if rep_base is None:
            return "none"
        idxs = base2idxs.get(rep_base, [])
        if not idxs:
            return rep_base
        i = idxs[0]
        return mapper[i]

    syno["canonical_strict"] = syno["name"].apply(lambda nm: apply_cluster_canon_with(canonical_map_strict, nm))
    syno["canonical_freq"]   = syno["name"].apply(lambda nm: apply_cluster_canon_with(canonical_map_freq, nm))

    # ------------------
    # 10) canonical 決定
    # ------------------
    name_freq = Counter(orig_terms)  

    def canon_penalty(s: str) -> int:
        s = str(s)
        p = 0
        if len(s) >= 8:
            p += 1
        if re.search(r"[ぁ-ん]{4,}", s):
            p += 1
        if "の" in s:
            p += 1
        return p


    syno["canonical"] = syno["canonical_strict"]
    # --------
    # 11) 保存
    # --------
    out_csv = os.path.abspath(os.path.join(base, "..", "datasets", f"synonym_ners{file_name}.csv"))
    out_pkl = os.path.abspath(os.path.join(base, "..", "datasets", f"synonym_ners{file_name}.pickle"))

    syno.to_csv(out_csv, index=False, encoding="utf-8-sig")
    with open(out_pkl, "wb") as f:
        pickle.dump(syno, f)
    print("syno saved:", out_csv, flush=True)

    # scores 保存
    scores_list = []
    for (i, j), ce_s in zip(index_pairs, ce_scores):
        cos_s = util.cos_sim(emb[i], emb[j]).item()
        s_jw = jw(readings[i], readings[j])
        s_core = core_norm(core_feature(token_lists[i], token_lists[j])) 

        total = alpha * float(ce_s) + beta * float(cos_s) + gamma * float(s_jw)

        scores_list.append({
            "term1": terms[i],
            "term2": terms[j],
            "ce_score": float(ce_s),
            "cos_score": float(cos_s),
            "string_score": float(s_jw),
            "core_score_dbg": float(s_core),
            "total_score": float(total),
        })

    df_scores = pd.DataFrame(scores_list)
    df_scores.to_csv(os.path.abspath(os.path.join(base, "..", "datasets", f"scores_{file_name}.csv")),
                     index=False, encoding="utf-8-sig")
    with open(os.path.abspath(os.path.join(base, "..", "datasets", f"scores_{file_name}.pickle")), "wb") as f:
        pickle.dump(scores_list, f)

    print("スコア保存完了", flush=True)
    print("unify_once終了", flush=True)
