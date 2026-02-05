# src/ingredient_norm/delta.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict, Counter
import ast
import math
import os
import pickle
import re
import unicodedata

import numpy as np
import pandas as pd
import networkx as nx
from rapidfuzz.distance import JaroWinkler
from sentence_transformers import SentenceTransformer, util, CrossEncoder
from sudachipy import dictionary
from sudachipy import tokenizer as sudachi_tokenizer


# =========================
# Config
# =========================
@dataclass(frozen=True)
class UnifyConfig:
    # suffix/core
    suf_min_len: int = 4
    suf_max_len: int = 7
    suf_min_df: int = 8
    jw_min_for_core: float = 0.90
    core_cand_min: float = 0.55
    str_min: float = 0.92
    top_str: int = 20
    top_core: int = 50

    # retrieval
    topk_bi: int = 30
    bi_model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    ce_model: str = "BAAI/bge-reranker-v2-m3"
    ce_batch: int = 16

    # fusion
    alpha: float = 0.550
    beta: float = 0.250
    gamma: float = 0.200
    thresh: float = 0.9420


# =========================
# Utilities
# =========================
def jw(a: str, b: str) -> float:
    return float(JaroWinkler.normalized_similarity(a, b))

def normalize_base(s: str) -> str:
    s = str(s)
    s = unicodedata.normalize("NFKC", s)
    s = re.split(r"[・/／,、\s]+", s)[0]
    s = re.sub(r"[『』「」\"'`]", "", s)
    return s

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

def build_bi_input(nm: str, labs: list[str]) -> str:
    return nm + " " + " ".join(labs) if isinstance(labs, list) and labs else nm


def get_sudachi_tokenizer():
    tok = dictionary.Dictionary().create()
    mode = sudachi_tokenizer.Tokenizer.SplitMode.C
    return tok, mode

def get_reading(tok, mode, text: str) -> str:
    ms = tok.tokenize(text, mode)
    out = []
    for m in ms:
        y = m.reading_form()
        out.append(y if (y and y != "*") else m.surface())
    return "".join(out)

def is_bad_suffix(s: str) -> bool:
    return (len(s) == 0) or (s[0] in "ァィゥェォャュョッー")


# =========================
# Core feature (suffix/core)
# =========================
def build_suffixes(readings0: list[str], cfg: UnifyConfig) -> list[str]:
    suf_df = Counter()
    for r in readings0:
        r = str(r)
        seen = set()
        Lmax = min(cfg.suf_max_len, len(r))
        for L in range(cfg.suf_min_len, Lmax + 1):
            suf = r[-L:]
            if not is_bad_suffix(suf):
                seen.add(suf)
        for suf in seen:
            suf_df[suf] += 1

    suffixes = [s for s, c in suf_df.items() if c >= cfg.suf_min_df and not is_bad_suffix(s)]
    suffixes.sort(key=lambda s: (-len(s), -suf_df[s], s))
    return suffixes

def attach_core_suffix(reading: str, suffixes: list[str], max_add=2) -> list[str]:
    hits = [suf for suf in suffixes if reading.endswith(suf)]
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

def build_token_lists(readings: list[str], suffixes: list[str]) -> list[list[str]]:
    token_lists = []
    for r in readings:
        cores = attach_core_suffix(r, suffixes, max_add=2)
        token_lists.append([r] + cores)
    return token_lists

def build_idf_map(token_lists0: list[list[str]]) -> dict[str, float]:
    N0 = len(token_lists0)
    df_map = Counter()
    for toks in token_lists0:
        for t in set(toks):
            df_map[t] += 1
    return {t: math.log((1 + N0) / (1 + c)) + 1.0 for t, c in df_map.items()}

def make_core_feature(idf_map: dict[str, float], cfg: UnifyConfig):
    max_idf = max(idf_map.values()) if idf_map else 1.0
    core_max = max_idf

    def core_norm(x: float) -> float:
        if core_max <= 0:
            return 0.0
        return float(min(1.0, max(0.0, x / core_max)))

    def core_feature(toksA: list[str], toksB: list[str]) -> float:
        best = 0.0
        for a in toksA:
            for b in toksB:
                s = jw(a, b)
                if s < cfg.jw_min_for_core:
                    continue
                w = s * min(idf_map.get(a, 0.0), idf_map.get(b, 0.0))
                if w > best:
                    best = w
        return best

    return core_feature, core_norm


# =========================
# Reading exact-match compression
# =========================
def pick_rep(terms0: list[str], idxs: list[int]) -> int:
    idxs = sorted(idxs)
    best = idxs[0]
    for i in idxs[1:]:
        a = terms0[best]
        b = terms0[i]
        if len(b) < len(a) or (len(b) == len(a) and b < a):
            best = i
    return best

def compress_by_reading(orig_terms0, terms0, readings0, term_labels0):
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
        rep = pick_rep(terms0, idxs)
        for i in idxs:
            rep_of[i] = rep

    rep_set = sorted(set(rep_of.values()))
    old2new = {old: new for new, old in enumerate(rep_set)}

    orig_terms = [orig_terms0[old] for old in rep_set]
    terms = [terms0[old] for old in rep_set]
    readings = [readings0[old] for old in rep_set]
    term_labels = [term_labels0[old] for old in rep_set]

    name2rep_base = {orig_terms0[i]: terms[old2new[rep_of[i]]] for i in range(len(orig_terms0))}
    return orig_terms, terms, readings, term_labels, name2rep_base


# =========================
# Canonical selection
# =========================
def canon_penalty(surface: str) -> int:
    s = str(surface)
    p = 0
    if len(s) >= 8: p += 1
    if re.search(r"[ぁ-ん]{4,}", s): p += 1
    if "の" in s: p += 1
    return p

def choose_canonical_strict(idxs, emb, orig_terms, name_freq):
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
            float(central[pos]),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_i = i
    return best_i

def choose_canonical_freq(idxs, emb, orig_terms, token_lists, name_freq):
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
            float(central[pos]),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_i = i
    return best_i


# =========================
# Main pipeline
# =========================
def unify_once2(file_name: str, cfg: UnifyConfig = UnifyConfig()) -> None:
    print("unify_once開始", flush=True)

    # paths (delta.py の位置基準)
    base = Path(__file__).resolve().parent
    datasets = (base / ".." / "datasets").resolve()

    csv_path = datasets / f"synonym_ners{file_name}.csv"
    labels_pkl = datasets / f"labels{file_name}.pickle"

    print("読み込みパス:", str(csv_path), flush=True)
    print("labels pickle:", str(labels_pkl), flush=True)

    # labels
    labels_df = pd.read_pickle(labels_pkl).copy()
    labels_df["name"] = labels_df["name"].astype(str)
    labels_df["labels"] = labels_df["labels"].apply(lambda x: x if isinstance(x, list) else ensure_list(x))

    # syno
    if csv_path.exists():
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

    # preprocess
    tok, mode = get_sudachi_tokenizer()

    orig_terms0 = syno["name"].tolist()
    terms0 = [normalize_base(t) for t in orig_terms0]
    readings0 = [get_reading(tok, mode, t) for t in terms0]
    term_labels0 = syno["labels"].tolist()
    terms_for_bi0 = [build_bi_input(t, labs) for t, labs in zip(terms0, term_labels0)]

    # suffix/core (build on pre-compression vocab)
    suffixes = build_suffixes(readings0, cfg)
    token_lists0 = build_token_lists(readings0, suffixes)
    idf_map = build_idf_map(token_lists0)
    core_feature, core_norm = make_core_feature(idf_map, cfg)

    # compress by exact reading
    orig_terms, terms, readings, term_labels, name2rep_base = compress_by_reading(
        orig_terms0, terms0, readings0, term_labels0
    )
    terms_for_bi = [build_bi_input(t, labs) for t, labs in zip(terms, term_labels)]

    token_lists = build_token_lists(readings, suffixes)

    # bucket by first char of reading
    bucket = defaultdict(list)
    for idx, r in enumerate(readings):
        if r:
            bucket[r[0]].append(idx)

    # Bi encode + topk
    bi = SentenceTransformer(cfg.bi_model)
    print("Bi-Encoder ベクトル化開始", flush=True)
    emb = bi.encode(terms_for_bi, normalize_embeddings=True, batch_size=256)
    print("terms のベクトル化完了", flush=True)

    cos_topk = util.semantic_search(emb, emb, top_k=cfg.topk_bi)

    # pairs
    pair_set = set()

    def add_pair(i, j):
        if i == j:
            return
        a, b = (i, j) if i < j else (j, i)
        if terms[a] == terms[b]:
            return
        pair_set.add((a, b))

    # from bi
    for i, neighs in enumerate(cos_topk):
        for cand in neighs:
            add_pair(i, cand["corpus_id"])

    # extra from reading/core
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
            if (s_jw >= cfg.str_min) or (s_core >= cfg.core_cand_min):
                best.append((max(s_jw, s_core), j))
        best.sort(reverse=True)
        for _, j in best[: cfg.top_str]:
            add_pair(i, j)

    # core inverted index candidates
    core2idxs = defaultdict(list)
    for idx, toks in enumerate(token_lists):
        for core in toks[1:]:
            core2idxs[core].append(idx)

    for i in range(len(terms)):
        for core in token_lists[i][1:]:
            js = core2idxs.get(core, [])
            for j in js[: cfg.top_core]:
                if i != j:
                    add_pair(i, j)

    index_pairs = list(pair_set)
    print("pair_set size:", len(index_pairs), flush=True)

    # Cross-Encoder
    ce = CrossEncoder(cfg.ce_model)
    pairs = [(terms_for_bi[i], terms_for_bi[j]) for i, j in index_pairs]

    ce_scores = []
    for st in range(0, len(pairs), cfg.ce_batch):
        batch = pairs[st : st + cfg.ce_batch]
        ce_scores.extend(ce.predict(batch))
        if st % (cfg.ce_batch * 200) == 0:
            print(f"Cross-Encoder {st}/{len(pairs)}", flush=True)

    # edges + scores (same fusion!)
    edges = []
    scores_list = []

    for (i, j), ce_s in zip(index_pairs, ce_scores):
        cos_s = float(util.cos_sim(emb[i], emb[j]).item())
        jw_s = jw(readings[i], readings[j])
        core_s = core_norm(core_feature(token_lists[i], token_lists[j]))
        s_str = max(jw_s, core_s)

        total = cfg.alpha * float(ce_s) + cfg.beta * cos_s + cfg.gamma * float(s_str)

        if total >= cfg.thresh:
            edges.append((i, j, total))

        scores_list.append({
            "term1": terms[i],
            "term2": terms[j],
            "ce_score": float(ce_s),
            "cos_score": cos_s,
            "jw_score": float(jw_s),
            "core_score": float(core_s),
            "str_score": float(s_str),
            "total_score": float(total),
        })

    print("edges:", len(edges), flush=True)

    # graph clustering
    G = nx.Graph()
    G.add_nodes_from(range(len(terms)))
    for i, j, w in edges:
        G.add_edge(i, j, weight=w)

    clusters = [sorted(list(comp)) for comp in nx.connected_components(G)]
    print("clusters:", len(clusters), flush=True)

    # canonical maps
    name_freq = Counter(orig_terms)

    base2idxs = defaultdict(list)
    for i, t in enumerate(terms):
        base2idxs[t].append(i)

    canonical_map_strict = {i: terms[i] for i in range(len(terms))}
    canonical_map_freq = {i: terms[i] for i in range(len(terms))}

    for idxs in clusters:
        root_s = choose_canonical_strict(idxs, emb, orig_terms, name_freq)
        root_f = choose_canonical_freq(idxs, emb, orig_terms, token_lists, name_freq)
        root_base_s = terms[root_s]
        root_base_f = terms[root_f]
        for i in idxs:
            canonical_map_strict[i] = root_base_s
            canonical_map_freq[i] = root_base_f

    def apply_cluster_canon(mapper, nm):
        rep_base = name2rep_base.get(nm, None)
        if rep_base is None:
            return "none"
        idxs = base2idxs.get(rep_base, [])
        if not idxs:
            return rep_base
        return mapper[idxs[0]]

    syno["canonical_strict"] = syno["name"].apply(lambda nm: apply_cluster_canon(canonical_map_strict, nm))
    syno["canonical_freq"] = syno["name"].apply(lambda nm: apply_cluster_canon(canonical_map_freq, nm))

    # final canonical policy (今は strict を採用)
    syno["canonical"] = syno["canonical_strict"]

    # save
    out_csv = datasets / f"synonym_ners{file_name}.csv"
    out_pkl = datasets / f"synonym_ners{file_name}.pickle"
    syno.to_csv(out_csv, index=False, encoding="utf-8-sig")
    with open(out_pkl, "wb") as f:
        pickle.dump(syno, f)
    print("syno saved:", str(out_csv), flush=True)

    df_scores = pd.DataFrame(scores_list)
    df_scores.to_csv(datasets / f"scores_{file_name}.csv", index=False, encoding="utf-8-sig")
    with open(datasets / f"scores_{file_name}.pickle", "wb") as f:
        pickle.dump(scores_list, f)

    print("スコア保存完了", flush=True)
    print("unify_once終了", flush=True)
