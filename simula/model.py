"""Model definitions.

Two models, deliberately:

- **LightGBM** on time-aware target encodings. This is the accuracy workhorse.
  GBMs dominate tabular CTR when the categorical explosion is handled by
  encoding rather than one-hot, and they give a cheap, inspectable feature
  importance that we use to justify the feature set.

- **Hashed logistic regression** (the "FTRL-style" baseline). Every large ad
  system has one of these because it retrains in minutes, handles brand-new
  feature values by construction, and gives a floor to compare against. If the
  GBM cannot clearly beat it, the extra serving complexity is not worth it.

The GBM is trained with a recency weight: a row from 6 days ago counts less than
one from yesterday. Given how fast the character mix churns (top-200 Jaccard
falls from 1.0 to 0.14 across the log) uniform weighting over-fits stale mix.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

LGB_PARAMS: dict = {
    "objective": "binary",
    "metric": ["binary_logloss", "auc"],
    "learning_rate": 0.05,
    "num_leaves": 127,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 5.0,
    "max_bin": 255,
    "verbosity": -1,
    "num_threads": 0,
    "seed": 17,
}


def recency_weights(hour_index: pd.Series, half_life_hours: float = 72.0) -> np.ndarray:
    """Exponential decay so recent rows dominate. half_life=3 days."""
    h = hour_index.to_numpy(dtype=float)
    age = h.max() - h
    return np.power(0.5, age / half_life_hours)


@dataclass
class TrainedGBM:
    booster: object
    features: list[str]
    best_iteration: int

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.booster.predict(  # type: ignore[attr-defined]
            X[self.features], num_iteration=self.best_iteration
        )


def train_gbm(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_va: pd.DataFrame,
    y_va: pd.Series,
    w_tr: np.ndarray | None = None,
    params: dict | None = None,
    num_boost_round: int = 3000,
    early_stopping: int = 100,
    verbose_eval: int = 200,
) -> TrainedGBM:
    import lightgbm as lgb

    p = dict(LGB_PARAMS)
    if params:
        p.update(params)

    feats = list(X_tr.columns)
    dtr = lgb.Dataset(X_tr[feats], label=y_tr, weight=w_tr, free_raw_data=False)
    dva = lgb.Dataset(X_va[feats], label=y_va, reference=dtr, free_raw_data=False)

    booster = lgb.train(
        p,
        dtr,
        num_boost_round=num_boost_round,
        valid_sets=[dtr, dva],
        valid_names=["train", "valid"],
        callbacks=[
            lgb.early_stopping(early_stopping, verbose=True),
            lgb.log_evaluation(verbose_eval),
        ],
    )
    return TrainedGBM(booster=booster, features=feats, best_iteration=booster.best_iteration)


# ----------------------------------------------------------------------
# Hashed logistic baseline
# ----------------------------------------------------------------------

HASH_COLS = [
    "site_id", "site_domain", "site_category",
    "app_id", "app_domain", "app_category",
    "device_model", "device_type", "device_conn_type", "device_id", "device_ip",
    "banner_pos", "C1", "C14", "C15", "C16", "C17", "C18", "C19", "C20", "C21",
    "character_id", "safety_tier", "creator_type",
    "hour_of_day", "turn_bucket", "smc_bucket",
]

HASH_CROSSES = [
    ("site_id", "banner_pos"),
    ("app_id", "C18"),
    ("site_category", "C15", "C16"),
    ("safety_tier", "C18"),
    ("character_id", "banner_pos"),
]


def _bucketize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["turn_bucket"] = np.minimum(df["conversation_turn"], 20).astype(str)
    df["smc_bucket"] = pd.cut(
        df["session_msg_count"], [0, 2, 5, 10, 20, 40, 1000], labels=False
    ).astype(str)
    return df


def hash_matrix(df: pd.DataFrame, n_bits: int = 20):
    """Feature-hash the raw categoricals into a sparse CSR matrix.

    The hashing trick is what makes this baseline cold-start-proof: an unseen
    site_id still lands in a bucket, it just shares that bucket with others.
    """
    from scipy.sparse import csr_matrix

    df = _bucketize(df)
    dim = 1 << n_bits
    n = len(df)

    pieces: list[np.ndarray] = []
    for c in HASH_COLS:
        if c not in df.columns:
            continue
        s = (c + "=" + df[c].astype(str)).to_numpy()
        pieces.append(s)
    for cols in HASH_CROSSES:
        if not all(c in df.columns for c in cols):
            continue
        s = "x_" + "_".join(cols) + "="
        vals = df[cols[0]].astype(str)
        for c in cols[1:]:
            vals = vals + "_" + df[c].astype(str)
        pieces.append((s + vals).to_numpy())

    n_feat = len(pieces)
    cols_idx = np.empty(n * n_feat, dtype=np.int64)
    for i, arr in enumerate(pieces):
        h = pd.util.hash_array(arr.astype(object)) % dim
        cols_idx[i * n : (i + 1) * n] = h

    # reorder into row-major
    cols_idx = cols_idx.reshape(n_feat, n).T.reshape(-1)
    indptr = np.arange(0, n * n_feat + 1, n_feat, dtype=np.int64)
    data = np.ones(n * n_feat, dtype=np.float32)
    return csr_matrix((data, cols_idx, indptr), shape=(n, dim))


def train_hashed_logreg(X_tr, y_tr, C: float = 1.0):
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(
        solver="liblinear", C=C, max_iter=200, tol=1e-4,
    )
    clf.fit(X_tr, y_tr)
    return clf
