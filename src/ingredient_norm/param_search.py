import numpy as np
import pickle
import networkx as nx
from sentence_transformers import util, SentenceTransformer

file_name = input("dish name? ").strip()

with open(f"../datasets/scores_{file_name}.pickle", "rb") as f:
    scores_list = pickle.load(f)

# terms
terms = list({s["term1"] for s in scores_list} | {s["term2"] for s in scores_list})
term2idx = {t: i for i, t in enumerate(terms)}

# embed
bi = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
emb = bi.encode(terms, normalize_embeddings=True, batch_size=256)

alphas = np.arange(0.4, 0.7, 0.05)
betas  = np.arange(0.25, 0.7, 0.05)
gammas = np.arange(0.05, 0.25, 0.05)

# クラスタ評価
def cluster_score(edges, size_soft_cap=25, lambda_size=0.03):
    G = nx.Graph()
    G.add_nodes_from(terms)
    for t1, t2, w in edges:
        G.add_edge(t1, t2, weight=w)

    clusters = list(nx.connected_components(G))
    total = 0.0
    n_used = 0

    for c in clusters:
        if len(c) <= 1:
            continue

        idxs = [term2idx[t] for t in c]
        sub = emb[idxs]
        sims = util.cos_sim(sub, sub).cpu().numpy()

        n = len(idxs)
        intra = (sims.sum() - np.trace(sims)) / (n * (n - 1))

        over = max(0, n - size_soft_cap)
        penalty = lambda_size * (over ** 2)

        total += intra - penalty
        n_used += 1

    return total / max(n_used, 1)

quantiles = np.arange(0.92, 0.99, 0.01)

best_score = -1
best_params = None

# ここが変更点：mixed_string を作って使う
# ※ core_score_dbg が無い古いpickleでも落ちないように 0.0 をデフォルトに
jw_arr   = np.array([float(s.get("string_score", 0.0)) for s in scores_list], dtype=np.float32)
core_arr = np.array([float(s.get("core_score_dbg", 0.0)) for s in scores_list], dtype=np.float32)
mix_arr  = np.maximum(jw_arr, core_arr)  # ★混合（max）

ce_arr  = np.array([float(s["ce_score"])  for s in scores_list], dtype=np.float32)
cos_arr = np.array([float(s["cos_score"]) for s in scores_list], dtype=np.float32)

for a in alphas:
    for b in betas:
        for g in gammas:
            if abs(a + b + g - 1.0) > 1e-6:
                continue

            totals = a * ce_arr + b * cos_arr + g * mix_arr  # ★ここも混合へ

            for q in quantiles:
                threshold = float(np.quantile(totals, q))

                edges = []
                for s, total in zip(scores_list, totals):
                    if total >= threshold:
                        edges.append((s["term1"], s["term2"], float(total)))

                score = cluster_score(edges)

                if score > best_score:
                    best_score = score
                    best_params = (a, b, g, q, threshold)

print("✅ 最適パラメータ（quantile-grid, mixed string=max(jw,core)）")
print(f"alpha     : {best_params[0]:.3f}")
print(f"beta      : {best_params[1]:.3f}")
print(f"gamma     : {best_params[2]:.3f}")
print(f"quantile  : {best_params[3]:.2f}")
print(f"threshold : {best_params[4]:.4f}")
print(f"cluster score : {best_score:.4f}")

# 参考：混合がどれくらい効いてるか（任意）
print(f"avg jw   : {jw_arr.mean():.4f}")
print(f"avg core : {core_arr.mean():.4f}")
print(f"avg mix  : {mix_arr.mean():.4f}")
win = (core_arr > jw_arr)
print("core wins ratio:", win.mean())
print("core wins count:", win.sum(), "/", len(win))
idx = np.where(core_arr > jw_arr)[0]
# core が勝ってる度合いでソート
top = idx[np.argsort((core_arr[idx] - jw_arr[idx]))[::-1]][:50]

for k in top[:20]:
    s = scores_list[k]
    print(s["term1"], " / ", s["term2"],
          " jw=", float(s.get("string_score",0)),
          " core=", float(s.get("core_score_dbg",0)),
          " mix=", float(max(s.get("string_score",0), s.get("core_score_dbg",0))))
a,b,g,q,threshold = best_params  # さっきの結果を使う想定

tot_jw  = a*ce_arr + b*cos_arr + g*jw_arr
tot_mix = a*ce_arr + b*cos_arr + g*mix_arr

rescued = (tot_mix >= threshold) & (tot_jw < threshold)
print("rescued count:", rescued.sum())

# rescued の中から醤油っぽいのを抽出
for k in np.where(rescued)[0][:200]:
    t1 = scores_list[k]["term1"]
    t2 = scores_list[k]["term2"]
    if ("醤油" in t1) or ("醤油" in t2) or ("正油" in t1) or ("正油" in t2) or ("しょうゆ" in t1) or ("しょうゆ" in t2):
        print("RESCUED:", t1, "/", t2, " tot_jw=", float(tot_jw[k]), " tot_mix=", float(tot_mix[k]),
              " jw=", float(jw_arr[k]), " core=", float(core_arr[k]))
