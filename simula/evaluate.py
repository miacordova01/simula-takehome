"""Evaluation metrics chosen for how the model is actually used.

Why these and not "whatever maximises the leaderboard":

- **AUC** is the usual headline but it is invariant to monotone rescaling, so a
  badly calibrated model can post a great AUC and still lose money. Reported,
  but not trusted alone.

- **Normalized entropy (NE)** -- log loss divided by the log loss of the base-rate
  predictor. This is the metric Facebook's ad ranking papers optimise, because it
  penalises miscalibration directly. NE < 1 means better than always predicting
  the base rate; the improvement over 1.0 is the number that matters.
  RIG (relative information gain) = 1 - NE.

- **Calibration**. In an auction, bid = predicted_CTR x value, so a systematic
  20% overprediction spends 20% too much. We report the overall ratio and a
  decile calibration curve, and expected calibration error.

- **Per-opportunity ranking metrics**. Global AUC pools across contexts, but the
  production question is narrower: given ONE impression opportunity and N
  candidate ads, does the top-ranked one get clicked? `group_ranking_metrics`
  builds candidate slates and reports recall@1 and NDCG, which is the metric
  the candidate-ranking deliverable is actually judged on.

- **Sliced metrics**. Cold entities are a large fraction of traffic, so an
  aggregate number hides the failure mode that matters. Everything is reported
  sliced by whether the entity was seen in history.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score


def normalized_entropy(y: np.ndarray, p: np.ndarray) -> float:
    """log loss / log loss of the base-rate predictor. Lower is better."""
    p = np.clip(p, 1e-7, 1 - 1e-7)
    base = float(np.mean(y))
    ll = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    ll_base = -(base * np.log(base) + (1 - base) * np.log(1 - base))
    return float(ll / ll_base)


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 20) -> float:
    """Mean |predicted - actual| across equal-count bins, weighted by bin size.

    A constant predictor (e.g. the base-rate baseline) cannot be split into
    quantile bins -- `qcut` returns all-NaN, which would silently divide 0/0.
    ECE is still well defined there: it collapses to the single-bin case,
    |mean(p) - mean(y)|. Same fallback covers a predictor with fewer distinct
    values than requested bins.
    """
    df = pd.DataFrame({"y": np.asarray(y, dtype=float), "p": np.asarray(p, dtype=float)})
    if df["p"].nunique() < 2:
        return float(abs(df["p"].mean() - df["y"].mean()))

    df["b"] = pd.qcut(df["p"], bins, labels=False, duplicates="drop")
    df = df.dropna(subset=["b"])
    if df.empty:
        return float(abs(np.mean(p) - np.mean(y)))

    g = df.groupby("b").agg(n=("y", "size"), act=("y", "mean"), pred=("p", "mean"))
    total = float(g["n"].sum())
    if total == 0:
        return float(abs(np.mean(p) - np.mean(y)))
    return float((g["n"] * (g["pred"] - g["act"]).abs()).sum() / total)


def calibration_table(y: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"y": y, "p": p})
    df["decile"] = pd.qcut(df["p"], bins, labels=False, duplicates="drop")
    g = df.groupby("decile").agg(
        n=("y", "size"), pred_ctr=("p", "mean"), actual_ctr=("y", "mean")
    )
    g["ratio"] = g["pred_ctr"] / g["actual_ctr"].replace(0, np.nan)
    return g


def core_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    ne = normalized_entropy(y, p)
    return {
        "n": int(len(y)),
        "actual_ctr": float(y.mean()),
        "pred_ctr": float(p.mean()),
        "auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "log_loss": float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7))),
        "norm_entropy": ne,
        "rig": 1.0 - ne,
        "calib_ratio": float(p.mean() / y.mean()) if y.mean() > 0 else np.nan,
        "ece": expected_calibration_error(y, p),
    }


def sliced_metrics(
    y: np.ndarray, p: np.ndarray, slices: dict[str, np.ndarray]
) -> pd.DataFrame:
    """Metrics on each boolean slice, plus overall."""
    rows = {"overall": core_metrics(y, p)}
    for name, mask in slices.items():
        m = np.asarray(mask, dtype=bool)
        if m.sum() < 100 or len(np.unique(y[m])) < 2:
            continue
        rows[name] = core_metrics(y[m], p[m])
    return pd.DataFrame(rows).T


def group_ranking_metrics(
    y: np.ndarray,
    p: np.ndarray,
    group_id: np.ndarray,
    min_group: int = 2,
) -> dict[str, float]:
    """Ranking quality *within* an ad opportunity.

    Each group is one slate of candidates competing for the same slot. We only
    score groups that contain at least one click and at least one non-click,
    since otherwise the ordering cannot be right or wrong.
    """
    df = pd.DataFrame({"y": y, "p": p, "g": group_id})
    g = df.groupby("g")["y"].agg(["size", "sum"])
    usable = g[(g["size"] >= min_group) & (g["sum"] > 0) & (g["sum"] < g["size"])].index
    df = df[df["g"].isin(usable)]
    if df.empty:
        return {"n_groups": 0}

    df = df.sort_values(["g", "p"], ascending=[True, False])
    df["rank"] = df.groupby("g").cumcount() + 1

    # recall@1: fraction of slates whose top-ranked candidate was clicked
    top1 = df[df["rank"] == 1]["y"].mean()

    # mean reciprocal rank of the first clicked candidate
    clicked = df[df["y"] == 1].groupby("g")["rank"].min()
    mrr = float((1.0 / clicked).mean())

    # NDCG with binary gains
    def _ndcg(sub: pd.DataFrame) -> float:
        gains = sub["y"].to_numpy()
        disc = 1.0 / np.log2(np.arange(2, len(gains) + 2))
        dcg = float((gains * disc).sum())
        idcg = float((np.sort(gains)[::-1] * disc).sum())
        return dcg / idcg if idcg > 0 else np.nan

    ndcg = float(df.groupby("g")[["y"]].apply(_ndcg).mean())

    # within-group AUC, averaged -- the cleanest "did we order this slate right"
    def _auc(sub: pd.DataFrame) -> float:
        return roc_auc_score(sub["y"], sub["p"])

    gauc = float(df.groupby("g")[["y", "p"]].apply(_auc).mean())

    return {
        "n_groups": int(df["g"].nunique()),
        "recall@1": float(top1),
        "mrr": mrr,
        "ndcg": ndcg,
        "group_auc": gauc,
    }
