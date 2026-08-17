"""Online scoring path.

The offline pipeline builds features with pandas groupbys over a million rows.
That is the wrong shape for a 50ms request budget. This module is the online
mirror of `features.py`, with three changes that matter for latency:

1. **Context/ad split.** Every feature is either context-only (same for all N
   candidates), ad-only, or a cross. Context features are computed ONCE per
   request and broadcast down the slate. With N=50 candidates that removes ~60%
   of the per-candidate work, because most of the expensive target-encoding
   lookups are on context columns (site_id, app_id, device_ip, character_id).

2. **Dict lookups, not pandas.** The fitted encoders are converted from pandas
   Series to plain Python dicts at load time. A `dict.get` is ~40ns; a
   `Series.map` on a 50-row frame is ~50us of overhead. At this batch size
   pandas is pure loss.

3. **One batched LightGBM call.** N candidates go through `predict` as a single
   (N, F) float32 array. Per-call overhead dominates at this size, so batching
   the slate is worth roughly an order of magnitude over per-candidate calls.

`ModelScorer.score_slate` is the function the ranker calls.
"""

from __future__ import annotations

import pickle
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ARTIFACTS
from .features import CROSS_COLS, PASSTHROUGH_COLS, TE_COLS, cross_name

# Which target-encoded columns come from the context vs the candidate.
_CTX_TE = [
    "site_id", "site_domain", "site_category",
    "app_id", "app_domain", "app_category",
    "device_model", "device_ip", "device_id", "character_id",
]
_AD_TE = ["C14", "C17", "C19", "C20", "C21"]

_CTX_COUNTS = ["site_id", "app_id", "device_ip", "device_id", "device_model", "character_id"]
_AD_COUNTS = ["C14", "C17"]

_SAFETY_MAP = {"sfw": 0, "suggestive": 1, "mature": 2}
_CREATOR_MAP = {"community": 0, "official": 1}


def _to_dict(s: pd.Series) -> dict:
    """pandas Series -> plain dict with str keys (fast online lookups)."""
    return {str(k): float(v) for k, v in s.items()}


@dataclass
class ScorerStats:
    n_calls: int = 0
    total_us: float = 0.0
    feature_us: float = 0.0
    predict_us: float = 0.0


class ModelScorer:
    """Loads the trained artifacts and scores a slate."""

    def __init__(self, artifacts_dir: Path | None = None) -> None:
        import lightgbm as lgb

        d = artifacts_dir or ARTIFACTS
        self.booster = lgb.Booster(model_file=str(d / "model.txt"))
        with open(d / "encoders.pkl", "rb") as f:
            blob = pickle.load(f)
        self.state = blob["state"]
        self.features: list[str] = blob["features"]
        self.calibrator = blob.get("calibrator")

        self.prior = float(self.state.prior)
        # Convert every encoder to a plain dict once, at load.
        self.te: dict[str, dict] = {k: _to_dict(v) for k, v in self.state.te_maps.items()}
        self.counts: dict[str, dict] = {
            k: {str(a): float(b) for a, b in v.items()}
            for k, v in self.state.count_maps.items()
        }
        self.parents: dict[str, dict] = {
            k: _to_dict(v) for k, v in self.state.parent_maps.items()
        }
        self.fidx = {f: i for i, f in enumerate(self.features)}
        self.n_features = len(self.features)
        self.stats = ScorerStats()

    # ------------------------------------------------------------------
    def context_vector(self, ctx) -> np.ndarray:
        """Compute the context-only slice of the feature vector. Cacheable."""
        v = np.full(self.n_features, np.nan, dtype=np.float32)

        def put(name: str, val: float) -> None:
            i = self.fidx.get(name)
            if i is not None:
                v[i] = val

        hour_s = str(ctx.hour)
        hour_of_day = int(hour_s[6:8])
        dt = pd.Timestamp(f"20{hour_s[0:2]}-{hour_s[2:4]}-{hour_s[4:6]} {hour_of_day}:00")
        dow = dt.dayofweek

        put("hour_of_day", hour_of_day)
        put("day_of_week", dow)
        put("is_weekend", 1.0 if dow >= 5 else 0.0)
        put("device_type", ctx.device_type)
        put("device_conn_type", ctx.device_conn_type)
        put("C1", ctx.C1)
        put("conversation_turn", ctx.conversation_turn)
        put("session_msg_count", ctx.session_msg_count)
        put("session_progress",
            min(ctx.conversation_turn / max(ctx.session_msg_count, 1), 1.0))
        put("turns_remaining", ctx.session_msg_count - ctx.conversation_turn)
        put("safety_tier", _SAFETY_MAP.get(ctx.safety_tier, -1))
        put("creator_type", _CREATOR_MAP.get(ctx.creator_type, -1))
        put("log_interactions", float(np.log1p(ctx.num_interactions)))
        put("char_age_days", float((dt - pd.Timestamp(ctx.created_at)).days))
        put("is_unknown_device", 1.0 if ctx.device_id == "a99f214a" else 0.0)

        # context target encodings + hierarchical backoff
        parent_of = {
            "character_id": ("character_id__safety_tier", ctx.safety_tier),
            "site_id": ("site_id__site_category", ctx.site_category),
            "app_id": ("app_id__app_category", ctx.app_category),
        }
        for c in _CTX_TE:
            raw = str(getattr(ctx, c))
            m = self.te[c]
            val = m.get(raw)
            seen = 1.0 if val is not None else 0.0
            if val is None and c in parent_of:
                pk, pv = parent_of[c]
                val = self.parents.get(pk, {}).get(str(pv))
            put(f"te_{c}", self.prior if val is None else val)
            put(f"seen_{c}", seen)

        for c in _CTX_COUNTS:
            n = self.counts[c].get(str(getattr(ctx, c)), 0.0)
            put(f"cnt_{c}", float(np.log1p(n)))

        self._ctx_cache_hour_bucket = hour_of_day // 6
        return v

    # ------------------------------------------------------------------
    def score_slate(self, ctx, candidates: list) -> np.ndarray:
        """pCTR for every candidate against one context."""
        t0 = time.perf_counter()
        n = len(candidates)
        if n == 0:
            return np.zeros(0)

        base = self.context_vector(ctx)
        M = np.repeat(base[None, :], n, axis=0)
        hb = self._ctx_cache_hour_bucket

        fidx = self.fidx
        prior = self.prior
        te = self.te
        counts = self.counts

        # Pre-resolve the column indices we write per candidate.
        i_bp = fidx.get("banner_pos")
        i_c15, i_c16, i_c18 = fidx.get("C15"), fidx.get("C16"), fidx.get("C18")

        for r, c in enumerate(candidates):
            row = M[r]
            if i_bp is not None:
                row[i_bp] = c.banner_pos
            if i_c15 is not None:
                row[i_c15] = c.C15
            if i_c16 is not None:
                row[i_c16] = c.C16
            if i_c18 is not None:
                row[i_c18] = c.C18

            for col in _AD_TE:
                key = str(getattr(c, col))
                m = te[col]
                val = m.get(key)
                i = fidx.get(f"te_{col}")
                if i is not None:
                    row[i] = prior if val is None else val
                i = fidx.get(f"seen_{col}")
                if i is not None:
                    row[i] = 0.0 if val is None else 1.0

            for col in _AD_COUNTS:
                i = fidx.get(f"cnt_{col}")
                if i is not None:
                    row[i] = float(np.log1p(counts[col].get(str(getattr(c, col)), 0.0)))

            # crosses
            for cols in CROSS_COLS:
                name = cross_name(cols)
                i = fidx.get(f"te_{name}")
                if i is None:
                    continue
                parts = []
                for col in cols:
                    if col == "hour_bucket":
                        parts.append(str(hb))
                    elif hasattr(c, col):
                        parts.append(str(getattr(c, col)))
                    else:
                        parts.append(str(getattr(ctx, col)))
                key = "\x1f".join(parts)
                val = te[name].get(key)
                row[i] = prior if val is None else val

        t1 = time.perf_counter()
        p = self.booster.predict(M, num_iteration=self.booster.best_iteration)
        p = np.asarray(p, dtype=float)
        if self.calibrator is not None:
            p = self.calibrator.predict(p)
        t2 = time.perf_counter()

        self.stats.n_calls += 1
        self.stats.feature_us += (t1 - t0) * 1e6
        self.stats.predict_us += (t2 - t1) * 1e6
        self.stats.total_us += (t2 - t0) * 1e6
        return p

    def __call__(self, ctx, candidates: list) -> np.ndarray:
        return self.score_slate(ctx, candidates)
