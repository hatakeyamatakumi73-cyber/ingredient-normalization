from sentence_transformers import SentenceTransformer, util, CrossEncoder
from rapidfuzz.distance import JaroWinkler
import unicodedata, re, networkx as nx
import numpy as np
import pickle
import pandas as pd
import codecs
import os
def unify_once(file_name):
    print("unify_once開始", flush=True)

    base = os.path.dirname(__file__)
    csv_path = os.path.join(base, "..", "datasets", f"synonym_ners{file_name}.csv")
    csv_path = os.path.abspath(csv_path)
    pickle_path = os.path.join(base, "..", "datasets", f"ners{file_name}.pickle")
    pickle_path = os.path.abspath(pickle_path)

    print("読み込みパス:", csv_path)

    syno = pd.read_csv(csv_path, encoding="utf-8-sig")
    print('read_syno')
    print(len(syno))

    # 0) 正規化と簡易前処理

    def normalize(s: str) -> str:
        s = unicodedata.normalize("NFKC", s)
        s = re.split(r"[・/／,、\s]+", s)[0]
        s = s.replace("ｰ","ー").replace("‐","ー").replace("–","ー").replace("—","ー")
        s = re.sub(r"[『』「」\"'`]", "", s)

        return s



    with open(pickle_path, 'rb') as f:
        ners_by_recipe = pickle.load(f)

    ners_by_recipe = ners_by_recipe[ners_by_recipe['ok/ng']=='ok']
    ners_by_recipe.set_index('ner_name', inplace=True)
    ingred_names = list(set(ners_by_recipe.index.tolist()))
    terms = [normalize(t) for t in ingred_names]

    # 未統一材料
    none_terms = syno[syno['canonical']=='none']['ner_name'].tolist()
    none_terms_norm = [normalize(t) for t in none_terms]

    # 1) Bi-Encoderでベクトル化
    bi = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    print("Bi-Encoder ベクトル化開始", flush=True)
    emb = bi.encode(terms, normalize_embeddings=True, batch_size=256)
    print("terms のベクトル化完了", flush=True)
    emb_none = bi.encode(none_terms_norm, normalize_embeddings=True, batch_size=256)
    print("none_terms のベクトル化完了", flush=True)
    
    # 2) Bi-Encoderによる候補探索

    k = 10
    # 元のterms同士
    try:
        cos_topk = util.semantic_search(emb, emb, top_k=k)
    except Exception as e:
        print("cos_topk 作成中に例外:", e)
        cos_topk = []

    # none_termsと元termsの組み合わせ
    try:
        cos_topk_none = util.semantic_search(emb_none, emb, top_k=k)
    except Exception as e:
        print("cos_topk_none 作成中に例外:", e)
        cos_topk_none = []
    print("cos_topk の長さ:", len(cos_topk))
    print("cos_topk_none の長さ:", len(cos_topk_none))

    # Cross-Encoderで再ランク

    print("Cross-Encoder ロード開始")
    ce = CrossEncoder("BAAI/bge-reranker-v2-m3")
    print("Cross-Encoder ロード完了")

    pairs, index_pairs = [], []
    for i, neighs in enumerate(cos_topk):
        for cand in neighs:
            j = cand['corpus_id']
            if i >= j: continue
            if terms[i] == terms[j]: continue
            pairs.append((terms[i], terms[j]))
            index_pairs.append((i, j))

    # none_termsとtermsのペア
    for i, neighs in enumerate(cos_topk_none):
        for cand in neighs:
            j = cand['corpus_id']
            pairs.append((none_terms_norm[i], terms[j]))
            index_pairs.append((len(terms)+i, j))  # ノード番号をずらす

    ce_scores = []
    for i in range(0, len(pairs), 16):
        batch = pairs[i:i+16]
        batch_scores = ce.predict(batch)
        ce_scores.extend(batch_scores)
        print(f"Cross-Encoder {i}/{len(pairs)} 完了", flush=True)
    

    # スコア融合 + エッジ作成
    def string_sim(a,b):
        return JaroWinkler.normalized_similarity(a,b)

    alpha, beta, gamma = 0.450, 0.500, 0.050
    edges = []

    for (i,j), ce_s in zip(index_pairs, ce_scores):
        if i < len(terms):
            cos_s = util.cos_sim(emb[i], emb[j]).item()
            t_i = terms[i]
        else:
            idx_none = i - len(terms)
            cos_s = util.cos_sim(emb_none[idx_none], emb[j]).item()
            t_i = none_terms_norm[idx_none]
        t_j = terms[j]
        s_str = string_sim(t_i, t_j)
        score = alpha*ce_s + beta*cos_s + gamma*s_str
        if score >= 0.7420:
            edges.append((i, j, score))

    # グラフ化 → クラスタ化
    G = nx.Graph()
    G.add_nodes_from(range(len(terms)+len(none_terms)))
    for i,j,w in edges:
        G.add_edge(i,j,weight=w)

    clusters = []
    all_terms = terms + none_terms_norm
    for comp in nx.connected_components(G):
        idxs = sorted(list(comp))
        clusters.append([all_terms[i] for i in idxs])

    term2idx = {t:i for i,t in enumerate(all_terms)}

    def choose_canonical(cluster_terms):
        idxs = [term2idx[t] for t in cluster_terms]
        sub_emb = np.vstack([emb[idx] if idx < len(terms) else emb_none[idx-len(terms)] for idx in idxs])
        sims = util.cos_sim(sub_emb, sub_emb)
        scores = sims.sum(dim=1).cpu().numpy()
        return cluster_terms[int(scores.argmax())]

    # canonical列作成
    ners_by_recipe.insert(1, 'canonical', 'none')

    for c in clusters:
        root = choose_canonical(c)
        for t in c:
            if t in ners_by_recipe.index:
                ners_by_recipe.loc[t,'canonical'] = root

    ners_by_recipe.reset_index(inplace=True)

    # 保存
    ners_by_recipe.to_csv(f"../datasets/synonym_ners{file_name}.csv", index=False, encoding="utf-8-sig", mode='w', header=True)
    with open(f'../datasets/synonym_ners{file_name}.pickle','wb') as f:
        pickle.dump(ners_by_recipe,f)

    scores_list = []
    for (i,j), ce_s in zip(index_pairs, ce_scores):
        cos_s = util.cos_sim(emb[i], emb[j]).item()
        s_str = string_sim(terms[i], terms[j])
        total = alpha*ce_s + beta*cos_s + gamma*s_str
        scores_list.append({
            "term1": terms[i],
            "term2": terms[j],
            "ce_score": ce_s,
            "cos_score": cos_s,
            "string_score": s_str,
            "total_score": total
        })

    # CSVに保存
    df_scores = pd.DataFrame(scores_list)
    df_scores.to_csv(f"../datasets/scores_{file_name}.csv", index=False, encoding="utf-8-sig")

    with open(f"../datasets/scores_{file_name}.pickle", "wb") as f:
        pickle.dump(scores_list, f)
    print(f"スコア保存完了: ../datasets/scores_{file_name}.csv")
    print("unify_once終了", flush=True)
pass
