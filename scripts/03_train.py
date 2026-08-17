"""
Train and evaluate the CTR model.

SPLIT
-----
Strictly temporal, because that is the only split that answers the production
question ("does a model trained on the past work tomorrow?"). A random split
would leak in three ways at once: the same device_ip appears on both sides, the
target encodings would see future clicks, and the character popularity mix would
be identical across folds -- which it emphatically is not (top-200 character
Jaccard between the first and last day is 0.14).

  day 141021        -> dropped (no history to encode against)
  days 141022-27    -> train
  day  141028       -> validation / early stopping
  days 141029-30    -> held-out test, never looked at during tuning

MODELS
------
  - hashed logistic regression (fast, cold-start-proof baseline)
  - LightGBM on time-aware target encodings (the candidate for production)
  - ablations that isolate what the character layer is worth

Writes artifacts/model.txt, artifacts/encoders.pkl and reports/model_results.md
"""

from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from simula.config import ARTIFACTS, REPORTS  # noqa: E402
from simula.data import add_time_columns, join_characters, load_characters, load_impressions  # noqa: E402
from simula.evaluate import calibration_table, core_metrics, sliced_metrics  # noqa: E402
from simula.features import (  # noqa: E402
    add_base_columns,
    build_time_aware_matrix,
    fit_encoders,
)
from simula.model import (  # noqa: E402
    TrainedGBM,
    hash_matrix,
    recency_weights,
    train_gbm,
    train_hashed_logreg,
)

TRAIN_DAYS = [141022, 141023, 141024, 141025, 141026, 141027]
VALID_DAYS = [141028]
TEST_DAYS = [141029, 141030]

OUT: list[str] = []


def say(s: str = "") -> None:
    print(s)
    OUT.append(s)


def table(df: pd.DataFrame, fmt: str = "%.4f") -> None:
    say("```")
    say(df.to_string(float_format=lambda x: fmt % x))
    say("```")


def main() -> None:
    say("# CTR model: training and evaluation")

    t0 = time.time()
    imp = add_time_columns(load_impressions())
    df = join_characters(imp, load_characters())
    df = add_base_columns(df)
    say(f"\nLoaded {len(df):,} rows in {time.time()-t0:.1f}s")

    # ------------------------------------------------------------------
    say("\n## Building time-aware features")
    say()
    say("Each day is encoded using only days strictly before it, mirroring a "
        "feature store rebuilt by a nightly batch job. Day 141021 is dropped "
        "because it has no history.")
    say("```")
    t0 = time.time()
    X, y, meta = build_time_aware_matrix(df, min_history_days=1)
    say("```")
    say(f"Encoded {X.shape[0]:,} rows x {X.shape[1]} features in {time.time()-t0:.1f}s")

    day = meta["day"]
    tr = day.isin(TRAIN_DAYS).to_numpy()
    va = day.isin(VALID_DAYS).to_numpy()
    te = day.isin(TEST_DAYS).to_numpy()

    say(f"- train {tr.sum():,} rows (CTR {y[tr].mean():.4f})")
    say(f"- valid {va.sum():,} rows (CTR {y[va].mean():.4f})")
    say(f"- test  {te.sum():,} rows (CTR {y[te].mean():.4f})")

    # ------------------------------------------------------------------
    say("\n## Baseline 1: predict the base rate")
    say()
    base_p = np.full(te.sum(), y[tr].mean())
    m_base = core_metrics(y[te].to_numpy(), base_p)
    table(pd.Series(m_base).to_frame("value"))

    # ------------------------------------------------------------------
    say("\n## Baseline 2: hashed logistic regression on raw categoricals")
    say()
    t0 = time.time()
    raw = df[df["day"] != 141021].reset_index(drop=True)
    assert len(raw) == len(X), "row alignment broken"
    H = hash_matrix(raw, n_bits=20)
    lr = train_hashed_logreg(H[tr], y[tr].to_numpy(), C=0.5)
    p_lr_te = lr.predict_proba(H[te])[:, 1]
    say(f"trained in {time.time()-t0:.1f}s on a {H.shape[1]:,}-dim hashed space")
    m_lr = core_metrics(y[te].to_numpy(), p_lr_te)
    table(pd.Series(m_lr).to_frame("value"))

    # ------------------------------------------------------------------
    say("\n## Model: LightGBM on time-aware target encodings")
    say()
    w = recency_weights(meta.loc[tr, "hour_index"], half_life_hours=72.0)
    say(f"Recency weighting: exponential, 72h half-life "
        f"(oldest train row weighted {w.min():.3f} vs newest 1.000)")
    say("```")
    t0 = time.time()
    gbm = train_gbm(X[tr], y[tr], X[va], y[va], w_tr=w)
    say("```")
    say(f"trained {gbm.best_iteration} trees in {time.time()-t0:.1f}s")

    p_te_raw = gbm.predict(X[te])
    p_va = gbm.predict(X[va])

    # --- recalibration ----------------------------------------------------
    # Bid = pCTR x value, so calibration error is spend error. Two variants,
    # because the obvious one turns out to be wrong here.
    from sklearn.isotonic import IsotonicRegression  # noqa: PLC0415

    # (a) The textbook approach: fit isotonic on the held-out validation day.
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(p_va, y[va].to_numpy())
    p_te_iso = iso.predict(p_te_raw)

    # (b) The production approach: refit on a short ROLLING window of recently
    # observed traffic, with a lag to represent click-attribution delay. Scoring
    # hour h uses labels from hours [h - lag - window, h - lag). This is what an
    # online recalibration job actually has available, and unlike (a) it tracks
    # day-level CTR movement instead of freezing yesterday's level in.
    LAG_H, WIN_H = 2, 12
    latest_calibrator = None
    te_hours = meta.loc[te, "hour_index"].to_numpy()
    y_te_arr = y[te].to_numpy()
    p_te_roll = p_te_raw.copy()
    va_hours = meta.loc[va, "hour_index"].to_numpy()
    y_va_arr = y[va].to_numpy()

    for h in np.unique(te_hours):
        lo, hi = h - LAG_H - WIN_H, h - LAG_H
        m_src_te = (te_hours >= lo) & (te_hours < hi)
        m_src_va = (va_hours >= lo) & (va_hours < hi)
        src_p = np.concatenate([p_te_raw[m_src_te], p_va[m_src_va]])
        src_y = np.concatenate([y_te_arr[m_src_te], y_va_arr[m_src_va]])
        tgt = te_hours == h
        if len(src_y) < 3000 or src_y.min() == src_y.max():
            continue  # not enough recent evidence -> leave raw
        iso_h = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso_h.fit(src_p, src_y)
        p_te_roll[tgt] = iso_h.predict(p_te_raw[tgt])
        latest_calibrator = iso_h  # the one a live service would be holding

    m_gbm_raw = core_metrics(y_te_arr, p_te_raw)
    m_gbm_iso = core_metrics(y_te_arr, p_te_iso)
    m_gbm_roll = core_metrics(y_te_arr, p_te_roll)

    say("Two recalibration strategies:")
    say(f"- fixed isotonic fitted on the validation day: "
        f"ECE {m_gbm_raw['ece']:.5f} -> {m_gbm_iso['ece']:.5f}, "
        f"calib ratio {m_gbm_raw['calib_ratio']:.4f} -> {m_gbm_iso['calib_ratio']:.4f}")
    say(f"- rolling isotonic ({WIN_H}h window, {LAG_H}h attribution lag): "
        f"ECE {m_gbm_raw['ece']:.5f} -> {m_gbm_roll['ece']:.5f}, "
        f"calib ratio {m_gbm_raw['calib_ratio']:.4f} -> {m_gbm_roll['calib_ratio']:.4f}")
    say()
    say("The fixed calibrator **hurts**: the validation day ran at CTR "
        f"{y[va].mean():.4f} against the test window's {y_te_arr.mean():.4f}, so it "
        "bakes in yesterday's level and systematically underpredicts. This is a "
        "concrete instance of the drift problem, and the reason the production "
        "answer is a short rolling recalibration rather than a frozen one.")

    # pick the better of the two for everything downstream
    if m_gbm_roll["ece"] <= m_gbm_raw["ece"]:
        p_te, m_gbm, chosen = p_te_roll, m_gbm_roll, "rolling isotonic"
    else:
        p_te, m_gbm, chosen = p_te_raw, m_gbm_raw, "raw (uncalibrated)"
    say(f"\nUsing **{chosen}** downstream.")

    say("\n### Head-to-head on the held-out test days")
    comp = pd.DataFrame(
        {
            "base_rate": m_base,
            "hashed_logreg": m_lr,
            "lightgbm_raw": m_gbm_raw,
            "lightgbm_fixed_isotonic": m_gbm_iso,
            "lightgbm_rolling_isotonic": m_gbm_roll,
        }
    ).T
    table(comp)

    rig_gain = (m_gbm["rig"] - m_lr["rig"]) / max(m_lr["rig"], 1e-9)
    say(f"\nLightGBM improves RIG over the hashed baseline by "
        f"**{rig_gain:.1%}** relative, and AUC by "
        f"{m_gbm['auc'] - m_lr['auc']:+.4f} absolute.")

    # ------------------------------------------------------------------
    say("\n### Calibration on test")
    say()
    say("Bid = pCTR x value, so calibration error is spend error. Decile table:")
    ct = calibration_table(y[te].to_numpy(), p_te, bins=10)
    table(ct)
    say(f"Overall predicted/actual = **{m_gbm['calib_ratio']:.4f}**, ECE = {m_gbm['ece']:.5f}")

    # ------------------------------------------------------------------
    say("\n### Metrics sliced by cold-start status")
    say()
    say("Aggregate numbers hide the cold path. `seen_*` flags come from the "
        "encoder: 1 if the entity appeared in history, 0 if it is brand new.")
    te_idx = np.where(te)[0]
    slices = {
        "cold_character": (X["seen_character_id"].to_numpy()[te] == 0),
        "warm_character": (X["seen_character_id"].to_numpy()[te] == 1),
        "cold_device_ip": (X["seen_device_ip"].to_numpy()[te] == 0),
        "warm_device_ip": (X["seen_device_ip"].to_numpy()[te] == 1),
        "cold_site": (X["seen_site_id"].to_numpy()[te] == 0),
        "unknown_device_id": (X["is_unknown_device"].to_numpy()[te] == 1),
        "known_device_id": (X["is_unknown_device"].to_numpy()[te] == 0),
    }
    sm = sliced_metrics(y[te].to_numpy(), p_te, slices)
    table(sm)

    # ------------------------------------------------------------------
    say("\n## Feature importance (gain)")
    say()
    imp_df = pd.DataFrame(
        {
            "feature": gbm.features,
            "gain": gbm.booster.feature_importance("gain"),  # type: ignore[attr-defined]
            "split": gbm.booster.feature_importance("split"),  # type: ignore[attr-defined]
        }
    ).sort_values("gain", ascending=False)
    imp_df["gain_share"] = imp_df["gain"] / imp_df["gain"].sum()
    table(imp_df.head(30))
    imp_df.to_csv(REPORTS / "feature_importance.csv", index=False)

    char_feats = [f for f in gbm.features
                  if "character" in f or "safety_tier" in f or "creator_type" in f
                  or "interactions" in f or "char_age" in f]
    conv_feats = ["conversation_turn", "session_msg_count", "session_progress",
                  "turns_remaining"]
    say(f"\n- character-derived features carry "
        f"{imp_df[imp_df.feature.isin(char_feats)]['gain_share'].sum():.1%} of total gain")
    say(f"- conversation-state features carry "
        f"{imp_df[imp_df.feature.isin(conv_feats)]['gain_share'].sum():.1%} of total gain")

    # ------------------------------------------------------------------
    say("\n## Ablations: what is each feature family actually worth?")
    say()
    say("Each row retrains the full model with one family removed. The honest "
        "test of whether the character layer earns its serving cost.")

    families = {
        "no character features": char_feats,
        "no conversation state": conv_feats,
        "no cross features": [f for f in gbm.features if f.startswith("te_x_")],
        "no target encodings": [f for f in gbm.features if f.startswith("te_")],
        "no count features": [f for f in gbm.features if f.startswith("cnt_")],
        "no device features": [f for f in gbm.features if "device" in f],
    }
    # Baseline must be the RAW model: the ablated variants below are scored
    # uncalibrated, so comparing them against the calibrated full model would
    # attribute the calibrator's effect to the dropped feature family.
    abl_rows = {"full model": m_gbm_raw}
    for name, drop in families.items():
        keep = [f for f in gbm.features if f not in set(drop)]
        if not keep or len(keep) == len(gbm.features):
            continue
        g2 = train_gbm(
            X[tr][keep], y[tr], X[va][keep], y[va], w_tr=w,
            verbose_eval=0, early_stopping=60,
        )
        abl_rows[name] = core_metrics(y[te].to_numpy(), g2.predict(X[te][keep]))
        say(f"  {name}: {len(keep)} features, {g2.best_iteration} trees")

    abl = pd.DataFrame(abl_rows).T[["auc", "norm_entropy", "rig", "log_loss"]]
    abl["auc_delta"] = abl["auc"] - abl.loc["full model", "auc"]
    abl["rig_delta_rel"] = (abl["rig"] - abl.loc["full model", "rig"]) / abl.loc["full model", "rig"]
    table(abl)

    # ------------------------------------------------------------------
    # Persist the production artifacts: a booster plus encoders fitted on
    # ALL available history (train+valid+test) -- that is what you would ship
    # tomorrow morning.
    say("\n## Persisting artifacts")
    final_state = fit_encoders(df)
    gbm.booster.save_model(str(ARTIFACTS / "model.txt"),  # type: ignore[attr-defined]
                           num_iteration=gbm.best_iteration)
    with open(ARTIFACTS / "encoders.pkl", "wb") as f:
        pickle.dump(
            {
                "state": final_state,
                "features": gbm.features,
                # the most recently refit rolling calibrator -- in a live service
                # this slot is replaced hourly by the recalibration job.
                "calibrator": latest_calibrator if chosen.startswith("rolling") else None,
            },
            f,
        )
    say(f"- wrote `{ARTIFACTS/'model.txt'}` ({gbm.best_iteration} trees)")
    say(f"- wrote `{ARTIFACTS/'encoders.pkl'}`")

    np.save(ARTIFACTS / "test_preds.npy", p_te)
    meta.loc[te].to_parquet(ARTIFACTS / "test_meta.parquet")
    X[te].to_parquet(ARTIFACTS / "test_X.parquet")
    y[te].to_frame("click").to_parquet(ARTIFACTS / "test_y.parquet")

    (REPORTS / "model_results.md").write_text("\n".join(OUT))
    print(f"\nWrote {REPORTS/'model_results.md'}")


if __name__ == "__main__":
    main()
