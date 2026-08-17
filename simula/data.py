"""Loading and joining the raw CSVs.

The impressions file is ~170MB / 1M rows. Reading it with the right dtypes
(categoricals for the hashed id columns) keeps it around 200MB in RAM instead
of ~1.5GB of Python strings, and makes repeat runs cheap via a parquet cache.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import CACHE, CHARACTERS_CSV, IMPRESSIONS_CSV

# Hashed 8-char hex ids -- read as str, converted to category after load.
_STR_COLS = [
    "site_id",
    "site_domain",
    "site_category",
    "app_id",
    "app_domain",
    "app_category",
    "device_id",
    "device_ip",
    "device_model",
    "character_id",
]

_INT_COLS = {
    "hour": "int32",
    "click": "int8",
    "banner_pos": "int16",
    "device_type": "int16",
    "device_conn_type": "int16",
    "C1": "int16",
    "C14": "int32",
    "C15": "int32",
    "C16": "int32",
    "C17": "int32",
    "C18": "int16",
    "C19": "int32",
    "C20": "int32",
    "C21": "int32",
    "conversation_turn": "int32",
    "session_msg_count": "int32",
}


def load_impressions(path: Path | None = None, use_cache: bool = True) -> pd.DataFrame:
    """Load impressions.csv with compact dtypes, caching to parquet."""
    path = path or IMPRESSIONS_CSV
    cache = CACHE / "impressions.parquet"
    if use_cache and cache.exists() and cache.stat().st_mtime > path.stat().st_mtime:
        return pd.read_parquet(cache)

    dtypes: dict[str, str] = dict(_INT_COLS)
    for c in _STR_COLS:
        dtypes[c] = "str"
    # id is a 20-digit unsigned integer -- keep as string, it is only a key.
    dtypes["id"] = "str"

    df = pd.read_csv(path, dtype=dtypes)
    for c in _STR_COLS:
        df[c] = df[c].astype("category")

    if use_cache:
        df.to_parquet(cache, index=False)
    return df


def load_characters(path: Path | None = None) -> pd.DataFrame:
    """Load characters.csv."""
    path = path or CHARACTERS_CSV
    df = pd.read_csv(
        path,
        dtype={
            "character_id": "str",
            "character_name": "str",
            "character_description": "str",
            "safety_tier": "category",
            "creator_type": "category",
            "num_interactions": "int64",
        },
        parse_dates=["created_at"],
    )
    return df


def add_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Expand the YYMMDDHH `hour` int into usable time columns (in place)."""
    s = df["hour"].astype("int64")
    df["hour_of_day"] = (s % 100).astype("int8")
    df["day"] = (s // 100).astype("int32")  # YYMMDD
    dt = pd.to_datetime(df["hour"].astype(str), format="%y%m%d%H")
    df["dt"] = dt
    df["day_of_week"] = dt.dt.dayofweek.astype("int8")
    df["is_weekend"] = (df["day_of_week"] >= 5).astype("int8")
    # Continuous hour index from the start of the log -- used for time splits
    # and for recency weighting.
    df["hour_index"] = ((dt - dt.min()) / pd.Timedelta("1h")).astype("int32")
    return df


def join_characters(imp: pd.DataFrame, chars: pd.DataFrame) -> pd.DataFrame:
    """Left-join character metadata onto impressions."""
    keep = [
        "character_id",
        "character_description",
        "safety_tier",
        "creator_type",
        "num_interactions",
        "created_at",
    ]
    out = imp.merge(chars[keep], on="character_id", how="left", validate="m:1")
    return out
