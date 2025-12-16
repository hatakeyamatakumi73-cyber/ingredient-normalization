import numpy as np
import pickle
import networkx as nx
from sentence_transformers import util, SentenceTransformer


file_name = input("dish name?")

with open(f"../datasets/scores_{file_name}.pickle", "rb") as f:
    scores_list = pickle.load(f)

terms = list({s['term1'] for s in scores_list} | {s['term2'] for s in scores_list})
term2idx = {t: i for i, t in enumerate(terms)}


bi = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
emb = bi.encode(terms, normalize_embeddings=True, batch_size=256)


alphas = np.arange(0.4, 0.7, 0.05)
betas  = np.arange(0.25, 0.7, 0.05)
gammas = np.arange(0.05, 0.25, 0.05)

QUANTILE = 0.90 

# クラスタ評価 
def cluster_score(edges):
    G = nx.Graph()
    G.add_nodes_from(terms)
    for t1, t2, w in edges:
        G.add_edge(t1, t2, weight=w)

    clusters = list(nx.connected_components(G))
    score = 0.0

    for c in clusters:
        if len(c) <= 1:
            continue
        idxs = [term2idx[t] for t in c]
        sub = emb[idxs]
        sims = util.cos_sim(sub, sub)
        score += sims.mean().item()

    return score / max(len(clusters), 1)


best_score = -1
best_params = None

#  グリッド探索
for a in alphas:
    for b in betas:
        for g in gammas:
            if abs(a + b + g - 1.0) > 1e-6:
                continue

            # total_score 再計算
            totals = [
                a*s['ce_score'] + b*s['cos_score'] + g*s['string_score']
                for s in scores_list
            ]

            #  分位点で threshold 決定
            threshold = np.quantile(totals, QUANTILE)

            edges = []
            for s, total in zip(scores_list, totals):
                if total >= threshold:
                    edges.append((s['term1'], s['term2'], total))

            score = cluster_score(edges)

            if score > best_score:
                best_score = score
                best_params = (a, b, g, threshold)

#  出力 
print(" 最適パラメータ（quantile-based）")
print(f"alpha     : {best_params[0]:.3f}")
print(f"beta      : {best_params[1]:.3f}")
print(f"gamma     : {best_params[2]:.3f}")
print(f"threshold : {best_params[3]:.4f}")
print(f"cluster score : {best_score:.4f}")
