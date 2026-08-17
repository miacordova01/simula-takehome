"""Why does the frozen model collapse at day 141027 instead of decaying smoothly?"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from simula.data import add_time_columns, join_characters, load_characters, load_impressions
from simula.evaluate import core_metrics
from simula.features import add_base_columns, apply_encoders, fit_encoders
from simula.model import recency_weights, train_gbm

df = add_base_columns(join_characters(add_time_columns(load_impressions()), load_characters()))

print("day -> dayofweek mapping:")
print(df.groupby("day")["day_of_week"].agg(["first", "size"]))

train_days = [141022, 141023, 141024]
state = fit_encoders(df[df["day"].isin([141021, 141022, 141023])])


def build(days):
    sub = df[df["day"].isin(days)]
    return apply_encoders(sub, state).astype("float32"), sub["click"], sub


Xtr, ytr, str_ = build(train_days)
Xva, yva, _ = build([141025])
w = recency_weights(str_["hour_index"], 72.0)

print("\ndow values in train:", sorted(str_["day_of_week"].unique()))

for label, drop in [("with day_of_week", []), ("WITHOUT day_of_week", ["day_of_week", "is_weekend"])]:
    feats = [c for c in Xtr.columns if c not in drop]
    g = train_gbm(Xtr[feats], ytr, Xva[feats], yva, w_tr=w, verbose_eval=0, early_stopping=60)
    print(f"\n=== {label} ({g.best_iteration} trees) ===")
    rows = []
    for d in [141025, 141026, 141027, 141028, 141029, 141030]:
        Xd, yd, _s = build([d])
        m = core_metrics(yd.to_numpy(), g.predict(Xd[feats]))
        rows.append({"day": d, "dow": int(_s["day_of_week"].iloc[0]),
                     "auc": m["auc"], "rig": m["rig"], "calib": m["calib_ratio"]})
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: "%.4f" % x))

# how much of the feature space goes stale?
print("\n=== share of rows falling back to the prior (unseen value) ===")
rows = []
for d in [141025, 141027, 141030]:
    Xd, _y, _s = build([d])
    r = {"day": d}
    for c in ["seen_character_id", "seen_device_ip", "seen_site_id", "seen_C14", "seen_C17"]:
        if c in Xd:
            r[c] = float(1 - Xd[c].mean())
    rows.append(r)
print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: "%.3f" % x))

print("\n=== char_age_days range (monotone in time -> extrapolation risk) ===")
for d in [141022, 141024, 141027, 141030]:
    s = df[df["day"] == d]["char_age_days"]
    print(f"  day {d}: min {s.min():.0f} max {s.max():.0f} mean {s.mean():.1f}")
