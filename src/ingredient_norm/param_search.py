# src/ingredient_norm/param_search.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pickle
import numpy as np
import networkx as nx

from sentence_transformers import util, SentenceTransformer


# =========================
# Config
# =========================

@dataclass(frozen=True)
class SearchConfig:

    # model
    bi_model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

    # grid
    alphas = np.arange(0.4, 0.7, 0.05)
    betas  = np.arange(0.25, 0.7, 0.05)
    gammas = np.arange(0.05, 0.25, 0.05)

    quantiles = np.arange(0.92, 0.99, 0.01)

    # cluster eval
    size_soft_cap: int = 25
    lambda_size: float = 0.03


# =========================
# IO
# =========================

def load_scores(datasets: Path, name: str) -> list[dict]:

    p = datasets / f"scores_{name}.pickle"

    with open(p, "rb") as f:
        scores = pickle.load(f)

    return scores


# =========================
# Preprocess
# =========================

def build_terms(scores):

    terms = list(
        {s["term1"] for s in scores} |
        {s["term2"] for s in scores}
    )

    term2idx = {t: i for i, t in enumerate(terms)}

    return terms, term2idx


def build_embeddings(terms, model_name: str):

    bi = SentenceTransformer(model_name)

    emb = bi.encode(
        terms,
        normalize_embeddings=True,
        batch_size=256
    )

    return emb


# =========================
# Scoring
# =========================

def cluster_score(
    edges,
    emb,
    term2idx,
    size_soft_cap=25,
    lambda_size=0.03
):

    G = nx.Graph()

    G.add_nodes_from(term2idx.keys())

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


# =========================
# Grid search
# =========================

def build_score_arrays(scores):

    jw = np.array(
        [float(s.get("string_score", 0.0)) for s in scores],
        dtype=np.float32
    )

    core = np.array(
        [float(s.get("core_score_dbg", 0.0)) for s in scores],
        dtype=np.float32
    )

    mix = np.maximum(jw, core)

    ce = np.array(
        [float(s["ce_score"]) for s in scores],
        dtype=np.float32
    )

    cos = np.array(
        [float(s["cos_score"]) for s in scores],
        dtype=np.float32
    )

    return ce, cos, jw, core, mix


def grid_search(
    scores,
    emb,
    term2idx,
    cfg: SearchConfig
):

    ce, cos, jw, core, mix = build_score_arrays(scores)

    best_score = -1.0
    best = None

    for a in cfg.alphas:
        for b in cfg.betas:
            for g in cfg.gammas:

                if abs(a + b + g - 1.0) > 1e-6:
                    continue

                totals = a * ce + b * cos + g * mix

                for q in cfg.quantiles:

                    th = float(np.quantile(totals, q))

                    edges = []

                    for s, t in zip(scores, totals):

                        if t >= th:
                            edges.append(
                                (s["term1"], s["term2"], float(t))
                            )

                    sc = cluster_score(
                        edges,
                        emb,
                        term2idx,
                        cfg.size_soft_cap,
                        cfg.lambda_size
                    )

                    if sc > best_score:
                        best_score = sc
                        best = (a, b, g, q, th)

    return best, best_score


# =========================
# Report
# =========================

def report_result(best, score, scores):

    a, b, g, q, th = best

    print("\n✅ Best parameters")
    print(f"alpha     : {a:.3f}")
    print(f"beta      : {b:.3f}")
    print(f"gamma     : {g:.3f}")
    print(f"quantile  : {q:.2f}")
    print(f"threshold : {th:.4f}")
    print(f"score     : {score:.4f}")

    jw = np.array([float(s.get("string_score", 0.0)) for s in scores])
    core = np.array([float(s.get("core_score_dbg", 0.0)) for s in scores])
    mix = np.maximum(jw, core)

    print("\n[debug]")
    print("avg jw   :", jw.mean())
    print("avg core :", core.mean())
    print("avg mix  :", mix.mean())


# =========================
# Main
# =========================

def main():

    name = input("dish name? ").strip()

    base = Path(__file__).resolve().parent
    datasets = (base / ".." / "datasets").resolve()

    cfg = SearchConfig()

    print("loading scores...")
    scores = load_scores(datasets, name)

    print("building terms...")
    terms, term2idx = build_terms(scores)

    print("embedding...")
    emb = build_embeddings(terms, cfg.bi_model)

    print("grid search...")
    best, score = grid_search(scores, emb, term2idx, cfg)

    report_result(best, score, scores)


if __name__ == "__main__":
    main()
