"""
Fixed-k root-cause-localization metrics (notebook §0.4/§0.8): Hit@k, Precision@k, Recall@k,
MRR, NDCG@k — evaluated at pre-registered k in {1, 3, 5, 10}, independent of |G_e|, unlike the
k=|G_e| precision=recall view in the main notebook's §8 (kept there only as a secondary diagnostic).
"""
from __future__ import annotations

import numpy as np

K_VALUES = (1, 3, 5, 10)


def hit_at_k(ranking: list[int], gt: set[int], k: int) -> int:
    return int(len(set(ranking[:k]) & gt) > 0)


def precision_at_k(ranking: list[int], gt: set[int], k: int) -> float:
    return len(set(ranking[:k]) & gt) / k


def recall_at_k(ranking: list[int], gt: set[int], k: int) -> float:
    return len(set(ranking[:k]) & gt) / len(gt)


def reciprocal_rank(ranking: list[int], gt: set[int]) -> float:
    for i, d in enumerate(ranking, start=1):
        if d in gt:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranking: list[int], gt: set[int], k: int) -> float:
    dcg = sum(1.0 / np.log2(i + 1) for i, d in enumerate(ranking[:k], start=1) if d in gt)
    ideal_hits = min(len(gt), k)
    idcg = sum(1.0 / np.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def score_ranking(ranking: list[int], gt: set[int], k_values=K_VALUES) -> dict:
    """All fixed-k metrics for one event's ranking vs. its ground-truth set."""
    out = {"mrr": reciprocal_rank(ranking, gt)}
    for k in k_values:
        out[f"hit@{k}"] = hit_at_k(ranking, gt, k)
        out[f"precision@{k}"] = precision_at_k(ranking, gt, k)
        out[f"recall@{k}"] = recall_at_k(ranking, gt, k)
        out[f"ndcg@{k}"] = ndcg_at_k(ranking, gt, k)
    return out


def random_ranking_expected_scores(n_dims: int, gt: set[int], k_values=K_VALUES,
                                    n_repeats: int = 2000, seed: int = 0) -> dict:
    """Monte-Carlo expected value of score_ranking under a uniform-random permutation —
    the chance floor a real localization method must beat (governing prompt §43)."""
    rng = np.random.default_rng(seed)
    dims = np.arange(1, n_dims + 1)
    accum = None
    for _ in range(n_repeats):
        perm = rng.permutation(dims).tolist()
        s = score_ranking(perm, gt, k_values)
        accum = s if accum is None else {k: accum[k] + v for k, v in s.items()}
    return {k: v / n_repeats for k, v in accum.items()}


def correlation_ranking(window: np.ndarray, target_col: int | None = None) -> list[int]:
    """Correlation-only localization baseline (governing prompt §44): rank the 38 KPI dims by
    |Pearson correlation| with a target anomaly-severity series, within the given window
    (T, feat) array. Returns a 1-indexed ranking, most-implicated first.

    If `target_col` is given, correlate every *other* dim against that one fixed target dim
    (excluded from its own ranking). If not given, each dim i is scored against a **leave-one-out**
    severity series -- the mean absolute deviation of the *other* feat-1 dims -- rather than a
    target that includes dim i itself. An earlier version used a target built from *all* dims
    (including the candidate), which let a dimension trivially "correlate" with an aggregate it was
    itself a large part of, inflating this baseline to near-perfect scores -- exactly the kind of
    self-referential leakage this project's leakage-audit discipline (notebook 01, governing
    prompt Section 31) exists to catch, so it is fixed here rather than left in.
    """
    feat = window.shape[1]
    corrs = np.zeros(feat)
    if target_col is not None:
        target = window[:, target_col]
        for i in range(feat):
            if i == target_col:
                corrs[i] = -np.inf
                continue
            c = np.corrcoef(target, window[:, i])[0, 1]
            corrs[i] = 0.0 if np.isnan(c) else abs(c)
    else:
        dev = np.abs(window - window.mean(axis=0))  # (T, feat)
        col_sum = dev.sum(axis=1)
        for i in range(feat):
            target = (col_sum - dev[:, i]) / (feat - 1)  # leave-one-out mean deviation of the rest
            c = np.corrcoef(target, window[:, i])[0, 1]
            corrs[i] = 0.0 if np.isnan(c) else abs(c)
    order = np.argsort(-corrs)
    return (order + 1).tolist()
