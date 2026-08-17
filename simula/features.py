"""Feature engineering.

Design constraints that drove the choices here:

1. **Everything is high-cardinality and categorical.** site_id has 2.6k values,
   device_ip 547k. One-hot is out; LightGBM's native categorical handling degrades
   badly past a few hundred levels. So the workhorse is *target encoding*.

2. **Target encoding leaks unless it is time-aware.** The standard K-fold OOF
   trick removes within-row leakage but still lets the encoder see the future,
   which flatters offline metrics and breaks in production. Instead this module
   uses an **expanding time window**: a row on day D is encoded using clicks
   observed strictly before day D. That is exactly what a feature store
   refreshed on a daily batch job would serve, so offline numbers transfer.

3. **Ranking needs context x ad crosses.** The ranker varies only the ad-side
   fields (banner_pos, C14-C21) while context is fixed. A model with no cross
   terms produces the same ad ordering for every context, which defeats the
   point. Crosses are target-encoded like any other column.

4. **Cold start is the common case, not an edge case.** 44% of device_ips are
   singletons and 7% of characters have <10 impressions, so the encoder must
   degrade gracefully: shrink toward a parent prior rather than toward the
   global mean. `hierarchical_te` does that.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Columns target-encoded on their own.
TE_COLS = [
    "site_id",
    "site_domain",
    "site_category",
    "app_id",
    "app_domain",
    "app_category",
    "device_model",
    "device_ip",
    "device_id",
    "C14",
    "C17",
    "C19",
    "C20",
    "C21",
    "character_id",
]

# Cross features. Context x ad terms let the model reorder candidates per context.
CROSS_COLS: list[tuple[str, ...]] = [
    ("site_id", "banner_pos"),          # surface x slot
    ("site_category", "C15", "C16"),    # surface x creative size
    ("app_id", "C18"),                  # host app x creative type
    ("app_category", "C21"),
    ("safety_tier", "C18"),             # character tone x creative type
    ("safety_tier", "C21"),
    ("character_id", "banner_pos"),
    ("device_model", "C15", "C16"),     # device x creative size (does it fit?)
    ("site_id", "C17"),
    ("hour_bucket", "site_category"),
]

# Low-cardinality columns passed straight through as small ints.
#
# `day_of_week` and `is_weekend` are deliberately EXCLUDED. Two reasons, and the
# first one bit hard before it was caught:
#
#   1. Trees extrapolate catastrophically on an ordinal whose test values fall
#      outside the training range. The log spans 10 days, so any training window
#      shorter than a week misses some weekdays entirely. A model trained on
#      Wed-Fri (dow 2-4) and scored on Monday (dow 0) sent every row to the
#      leftmost bin: calibration went from 0.98 to 2.18 and RIG went negative.
#      That looked exactly like violent concept drift and was in fact a feature
#      encoding bug. See scripts/debug_staleness.py for the isolation.
#   2. Even done correctly (cyclic encoding or categorical), day-of-week is not
#      identifiable here. Each weekday appears once or twice, so its effect is
#      perfectly confounded with that specific date's traffic mix. There is no
#      way to separate "Mondays are good" from "the 27th was good".
#
# hour_of_day is kept: it recurs 10 times and its range is fully covered.
PASSTHROUGH_COLS = [
    "banner_pos",
    "device_type",
    "device_conn_type",
    "C1",
    "C15",
    "C16",
    "C18",
    "hour_of_day",
    "conversation_turn",
    "session_msg_count",
]

# Columns we build frequency (count) features for. Count features are a cheap,
# very robust proxy for "how much do we know about this entity".
COUNT_COLS = [
    "site_id",
    "app_id",
    "device_ip",
    "device_id",
    "device_model",
    "character_id",
    "C14",
    "C17",
]


@dataclass
class EncoderState:
    """Fitted per-day encoders. Keyed by the day they are *valid for*."""

    prior: float
    te_maps: dict[str, pd.Series] = field(default_factory=dict)
    count_maps: dict[str, pd.Series] = field(default_factory=dict)
    # parent prior used when a value is unseen, e.g. character -> safety_tier CTR
    parent_maps: dict[str, pd.Series] = field(default_factory=dict)


def _key(df: pd.DataFrame, cols: tuple[str, ...]) -> pd.Series:
    """Join several columns into one string key for crossing."""
    if len(cols) == 1:
        return df[cols[0]].astype(str)
    out = df[cols[0]].astype(str)
    for c in cols[1:]:
        out = out + "\x1f" + df[c].astype(str)
    return out


def cross_name(cols: tuple[str, ...]) -> str:
    return "x_" + "__".join(cols)


def add_base_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Derived columns needed before encoding. Safe to call twice."""
    if "hour_bucket" not in df.columns:
        # 4 coarse dayparts -- keeps the cross cardinality sane.
        df["hour_bucket"] = (df["hour_of_day"] // 6).astype("int8")
    if "session_progress" not in df.columns:
        df["session_progress"] = (
            df["conversation_turn"] / df["session_msg_count"].clip(lower=1)
        ).clip(0, 1).astype("float32")
    if "turns_remaining" not in df.columns:
        df["turns_remaining"] = (
            df["session_msg_count"] - df["conversation_turn"]
        ).astype("int32")
    if "log_interactions" not in df.columns and "num_interactions" in df.columns:
        df["log_interactions"] = np.log1p(df["num_interactions"]).astype("float32")
    if "char_age_days" not in df.columns and "created_at" in df.columns:
        df["char_age_days"] = (df["dt"] - df["created_at"]).dt.days.astype("float32")
    if "is_unknown_device" not in df.columns:
        df["is_unknown_device"] = (df["device_id"].astype(str) == "a99f214a").astype("int8")
    return df


def fit_encoders(
    hist: pd.DataFrame,
    prior_weight: float = 25.0,
    cross_prior_weight: float = 50.0,
) -> EncoderState:
    """Fit target-encoding and count maps on a history slice.

    prior_weight is the number of pseudo-observations pulled toward the prior.
    Higher = more shrinkage = safer for rare values. 25 was chosen so a value
    needs roughly 25 impressions before its own rate dominates the prior, which
    matches the point where a binomial CTR estimate at p~0.18 gets a standard
    error below the between-entity spread we measured (sd ~0.024).
    """
    prior = float(hist["click"].mean())
    st = EncoderState(prior=prior)

    for c in TE_COLS:
        g = hist.groupby(c, observed=True)["click"].agg(["sum", "size"])
        st.te_maps[c] = (g["sum"] + prior * prior_weight) / (g["size"] + prior_weight)

    for cols in CROSS_COLS:
        name = cross_name(cols)
        k = _key(hist, cols)
        g = hist.groupby(k, observed=True)["click"].agg(["sum", "size"])
        st.te_maps[name] = (g["sum"] + prior * cross_prior_weight) / (
            g["size"] + cross_prior_weight
        )

    for c in COUNT_COLS:
        st.count_maps[c] = hist.groupby(c, observed=True).size()

    # Parent priors for hierarchical backoff on cold entities.
    if "safety_tier" in hist.columns:
        g = hist.groupby("safety_tier", observed=True)["click"].agg(["sum", "size"])
        st.parent_maps["character_id__safety_tier"] = (
            (g["sum"] + prior * prior_weight) / (g["size"] + prior_weight)
        )
    g = hist.groupby("site_category", observed=True)["click"].agg(["sum", "size"])
    st.parent_maps["site_id__site_category"] = (
        (g["sum"] + prior * prior_weight) / (g["size"] + prior_weight)
    )
    g = hist.groupby("app_category", observed=True)["click"].agg(["sum", "size"])
    st.parent_maps["app_id__app_category"] = (
        (g["sum"] + prior * prior_weight) / (g["size"] + prior_weight)
    )
    return st


def apply_encoders(df: pd.DataFrame, st: EncoderState) -> pd.DataFrame:
    """Transform rows with a fitted EncoderState. Returns a numeric frame."""
    out = pd.DataFrame(index=df.index)

    for c in PASSTHROUGH_COLS:
        if c in df.columns:
            out[c] = df[c].astype("float32")

    for c in ("session_progress", "turns_remaining", "log_interactions",
              "char_age_days", "is_unknown_device"):
        if c in df.columns:
            out[c] = df[c].astype("float32")

    # safety_tier / creator_type as ordinal ints (only 3 and 2 levels).
    if "safety_tier" in df.columns:
        out["safety_tier"] = (
            df["safety_tier"].astype(str).map({"sfw": 0, "suggestive": 1, "mature": 2})
            .fillna(-1).astype("float32")
        )
    if "creator_type" in df.columns:
        out["creator_type"] = (
            df["creator_type"].astype(str).map({"community": 0, "official": 1})
            .fillna(-1).astype("float32")
        )

    # --- target encodings -------------------------------------------------
    for c in TE_COLS:
        m = st.te_maps[c]
        vals = df[c].map(m).astype("float32")
        # hierarchical backoff for unseen values
        parent_key = next(
            (k for k in st.parent_maps if k.startswith(f"{c}__")), None
        )
        if parent_key is not None:
            parent_col = parent_key.split("__", 1)[1]
            fallback = df[parent_col].map(st.parent_maps[parent_key]).astype("float32")
            vals = vals.fillna(fallback)
        out[f"te_{c}"] = vals.fillna(st.prior)
        # explicit "was this value known?" flag -- lets the tree learn a
        # different response surface on the cold-start path.
        out[f"seen_{c}"] = (~df[c].map(m).isna()).astype("float32")

    for cols in CROSS_COLS:
        name = cross_name(cols)
        k = _key(df, cols)
        out[f"te_{name}"] = k.map(st.te_maps[name]).astype("float32").fillna(st.prior)

    # --- counts -----------------------------------------------------------
    for c in COUNT_COLS:
        out[f"cnt_{c}"] = np.log1p(
            df[c].map(st.count_maps[c]).astype("float32").fillna(0.0)
        )

    return out


def build_time_aware_matrix(
    df: pd.DataFrame,
    min_history_days: int = 1,
    prior_weight: float = 25.0,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Encode every row using only data strictly before its own day.

    Returns (X, y, meta). Rows in the first `min_history_days` days are dropped
    because they have no history to encode against.

    This simulates a feature store rebuilt once per day, which is both realistic
    and strictly leakage-free: no row is ever encoded with a click that had not
    happened yet.
    """
    df = add_base_columns(df)
    days = sorted(df["day"].unique())

    X_parts, y_parts, meta_parts = [], [], []
    for i, d in enumerate(days):
        if i < min_history_days:
            continue
        hist = df[df["day"] < d]
        cur = df[df["day"] == d]
        st = fit_encoders(hist, prior_weight=prior_weight)
        X_parts.append(apply_encoders(cur, st))
        y_parts.append(cur["click"])
        meta_parts.append(cur[["day", "hour_index", "character_id", "device_ip", "device_id"]])
        if verbose:
            print(f"  encoded day {d}: {len(cur):,} rows from {len(hist):,} history rows")

    X = pd.concat(X_parts).astype("float32")
    y = pd.concat(y_parts)
    meta = pd.concat(meta_parts)
    return X, y, meta
