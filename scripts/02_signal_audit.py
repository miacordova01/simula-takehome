"""
Signal audit: which columns actually carry information about `click`?

Marginal CTR-by-bucket tables are misleading on 1M rows -- with enough data a
purely random column produces "significant" looking spread, and a genuinely
predictive column can be a proxy for something else. Three sharper tests:

  T1. Out-of-time transfer. Compute per-value CTR on an early window, score a
      later window with it, and measure AUC. A column only has *usable* signal
      if its per-value CTR estimated in the past predicts the future. This is
      exactly how the feature would be used in production.

  T2. Permutation control. Shuffle the column, redo T1. The gap between real
      and shuffled tells you how much of the apparent spread is just binomial
      noise plus the estimator's own overfitting.

  T3. Conditional contribution. Add the column to a gradient-boosted model that
      already has the strong context features, and measure the delta in
      held-out AUC / log loss. This catches confounding: safety_tier could look
      predictive only because mature characters live on different sites.

Writes reports/signal_audit.md
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from simula.config import RANDOM_SEED, REPORTS  # noqa: E402
from simula.data import add_time_columns, join_characters, load_characters, load_impressions  # noqa: E402

OUT: list[str] = []
rng = np.random.default_rng(RANDOM_SEED)

# Train on days 21-27, test on 28-30. Same split the real model will use.
SPLIT_DAY = 141028


def say(s: str = "") -> None:
    print(s)
    OUT.append(s)


def table(df: pd.DataFrame, fmt: str = "%.4f") -> None:
    say("```")
    say(df.to_string(float_format=lambda x: fmt % x))
    say("```")


def smoothed_te(
    train: pd.DataFrame, test: pd.DataFrame, col: str, prior_w: float = 50.0
) -> np.ndarray:
    """Per-value smoothed CTR learned on train, applied to test."""
    prior = train["click"].mean()
    g = train.groupby(col, observed=True)["click"].agg(["sum", "size"])
    te = (g["sum"] + prior * prior_w) / (g["size"] + prior_w)
    return test[col].map(te).astype(float).fillna(prior).to_numpy()


def transfer_auc(train: pd.DataFrame, test: pd.DataFrame, col: str) -> float:
    """T1: AUC of a past-estimated per-value CTR on future data."""
    return roc_auc_score(test["click"], smoothed_te(train, test, col))


def main() -> None:
    say("# Signal audit")
    say()
    say("Train window: days 141021-141027. Test window: days 141028-141030.")

    imp = load_impressions()
    chars = load_characters()
    imp = add_time_columns(imp)
    df = join_characters(imp, chars)

    df["char_age_days"] = (df["dt"] - df["created_at"]).dt.days
    df["session_progress"] = (
        df["conversation_turn"] / df["session_msg_count"].clip(lower=1)
    ).clip(0, 1)
    df["is_first_turn"] = (df["conversation_turn"] == 1).astype("int8")
    df["log_interactions"] = np.log1p(df["num_interactions"])

    train = df[df["day"] < SPLIT_DAY]
    test = df[df["day"] >= SPLIT_DAY]
    say(f"train rows {len(train):,} (CTR {train['click'].mean():.4f}), "
        f"test rows {len(test):,} (CTR {test['click'].mean():.4f})")

    # ------------------------------------------------------------------
    say()
    say("## T1/T2: out-of-time transfer AUC vs a shuffled control")
    say()
    say("`auc` = past-estimated per-value CTR scoring the future.")
    say("`auc_shuffled` = same after permuting the column (pure-noise floor, ~0.500).")
    say("`edge` = auc - auc_shuffled. Anything under ~0.005 is not usable signal.")
    say()

    cols = [
        # context / publisher / device
        "site_id", "site_domain", "site_category",
        "app_id", "app_domain", "app_category",
        "device_id", "device_ip", "device_model", "device_type", "device_conn_type",
        # anonymized
        "C1", "C14", "C15", "C16", "C17", "C18", "C19", "C20", "C21", "banner_pos",
        # time
        "hour_of_day", "day_of_week",
        # character / conversation
        "character_id", "safety_tier", "creator_type",
        "conversation_turn", "session_msg_count", "session_progress",
        "is_first_turn", "num_interactions", "char_age_days",
    ]

    rows = []
    test_shuf = test.copy()
    for c in cols:
        try:
            a = transfer_auc(train, test, c)
        except Exception as e:  # noqa: BLE001
            say(f"  (skipped {c}: {e})")
            continue
        # permutation control: shuffle within the test set so the mapping is broken
        # but the marginal distribution and cardinality are preserved.
        test_shuf[c] = rng.permutation(test[c].to_numpy())
        a_sh = transfer_auc(train, test_shuf, c)
        rows.append({
            "column": c,
            "n_unique": int(df[c].nunique()),
            "auc": a,
            "auc_shuffled": a_sh,
            "edge": a - a_sh,
        })

    res = pd.DataFrame(rows).sort_values("edge", ascending=False).reset_index(drop=True)
    table(res)

    res.to_csv(REPORTS / "signal_audit.csv", index=False)

    strong = res[res["edge"] >= 0.01]["column"].tolist()
    weak = res[res["edge"] < 0.005]["column"].tolist()
    say()
    say(f"**Carries signal (edge >= 0.010):** {', '.join(strong)}")
    say()
    say(f"**No usable standalone signal (edge < 0.005):** {', '.join(weak)}")

    # ------------------------------------------------------------------
    say()
    say("## Is `character_id` more than binomial noise?")
    say()
    # Variance decomposition on the train window: compare observed between-character
    # variance in CTR to what pure binomial sampling would produce.
    g = train.groupby("character_id", observed=True)["click"].agg(["sum", "size"])
    g = g[g["size"] >= 100]
    p_hat = g["sum"] / g["size"]
    p_bar = train["click"].mean()
    observed_var = p_hat.var(ddof=1)
    expected_noise_var = float((p_bar * (1 - p_bar) / g["size"]).mean())
    excess = observed_var - expected_noise_var
    say(f"- characters with >=100 train impressions: {len(g):,}")
    say(f"- observed variance of per-character CTR: {observed_var:.6f}")
    say(f"- variance expected from binomial noise alone: {expected_noise_var:.6f}")
    say(f"- excess (true between-character variance): {excess:.6f}")
    if excess > 0:
        say(f"- implied true between-character CTR sd: {np.sqrt(max(excess,0)):.4f} "
            f"(vs base CTR {p_bar:.4f})")
    icc = max(excess, 0) / observed_var
    say(f"- share of observed spread that is real (intra-class correlation): **{icc:.1%}**")

    # ------------------------------------------------------------------
    say()
    say("## Is the `safety_tier` lift real or confounded?")
    say()
    base = df.groupby("safety_tier", observed=True)["click"].mean()
    say("Raw CTR by tier:")
    table(base.to_frame("ctr"))

    # Stratify by the strongest context feature (C15/C16 = creative size, and site_id).
    for strat in ["C15", "site_id", "app_id", "C18"]:
        # weight each tier's CTR by the *overall* strata distribution -> direct
        # standardisation, removes composition effects.
        cell = df.groupby([strat, "safety_tier"], observed=True)["click"].agg(["sum", "size"])
        cell = cell.reset_index()
        w = df.groupby(strat, observed=True).size().rename("w")
        cell = cell.merge(w, on=strat)
        cell["ctr"] = cell["sum"] / cell["size"]
        # only strata where all tiers appear with enough support
        ok = cell.groupby(strat, observed=True)["size"].transform("min") >= 30
        cell = cell[ok]
        std = (
            cell.assign(num=lambda d: d["ctr"] * d["w"])
            .groupby("safety_tier", observed=True)
            .apply(lambda d: d["num"].sum() / d["w"].sum(), include_groups=False)
        )
        say(f"CTR standardised over `{strat}` strata:")
        table(std.to_frame("ctr_adj"))

    # Does character->tier assignment look independent of context?
    say()
    say("If characters were assigned to traffic at random, tier should be "
        "independent of the publisher/creative columns. Cramer's V:")
    from itertools import product  # noqa: PLC0415

    def cramers_v(a: pd.Series, b: pd.Series) -> float:
        ct = pd.crosstab(a, b)
        chi2 = ((ct - np.outer(ct.sum(1), ct.sum(0)) / ct.values.sum()) ** 2
                / (np.outer(ct.sum(1), ct.sum(0)) / ct.values.sum())).values.sum()
        n = ct.values.sum()
        return float(np.sqrt((chi2 / n) / (min(ct.shape) - 1)))

    sample = df.sample(200_000, random_state=RANDOM_SEED)
    vs = {c: cramers_v(sample["safety_tier"], sample[c])
          for c in ["C15", "C18", "site_category", "app_category", "banner_pos", "C1"]}
    table(pd.Series(vs).sort_values(ascending=False).to_frame("cramers_v"))

    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "signal_audit.md").write_text("\n".join(OUT))
    print(f"\nWrote {REPORTS/'signal_audit.md'}")


if __name__ == "__main__":
    main()
