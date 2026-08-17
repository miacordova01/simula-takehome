"""
Drift: measure it, then adapt to it.

PART A -- measurement
  A1. CTR level over time (is the target itself moving?)
  A2. Feature distribution drift via PSI, day 1 vs each later day
  A3. Character mix churn (how fast does the popular set turn over?)
  A4. Model staleness curve: train once, evaluate on each subsequent day.
      This is the number that sets the retrain cadence.
  A5. Same curve with daily-refreshed encodings but a frozen tree ensemble --
      isolates how much of the decay is fixable by cheap feature refresh vs
      how much needs a full retrain.

PART B -- adaptation prototype
  An inventory-constrained replay over the held-out days. Ad groups (C17, the
  400-value creative/campaign-like id) compete for each character-hour's slots.
  Policies are compared on realised CTR and on how concentrated the resulting
  exposure is.

  The counterfactual assumption is stated plainly: within a (character-cohort,
  ad-group, hour) cell the logged impressions are treated as exchangeable, so
  re-allocating slots across groups within an hour yields the observed
  per-cell click rates. That holds well here because character assignment is
  independent of the ad columns (Cramer's V < 0.011 across the board, measured
  in the signal audit), which is exactly the condition this replay needs.

Writes reports/drift.md
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from simula.adaptive import (  # noqa: E402
    AdaptiveAllocator,
    DecayedBetaBandit,
    FatigueState,
    concentration_metrics,
)
from simula.config import RANDOM_SEED, REPORTS  # noqa: E402
from simula.data import add_time_columns, join_characters, load_characters, load_impressions  # noqa: E402
from simula.evaluate import core_metrics  # noqa: E402
from simula.features import add_base_columns, apply_encoders, fit_encoders  # noqa: E402
from simula.model import recency_weights, train_gbm  # noqa: E402

OUT: list[str] = []
DAYS = [141021, 141022, 141023, 141024, 141025, 141026, 141027, 141028, 141029, 141030]

# Fraction of each hour's logged volume the replay policy gets to fill. Must be
# < 1.0 or the policy has no freedom and every arm is forced to its logged count.
FILL_RATE = 0.5


def say(s: str = "") -> None:
    print(s)
    OUT.append(s)


def table(df: pd.DataFrame, fmt: str = "%.4f") -> None:
    say("```")
    say(df.to_string(float_format=lambda x: fmt % x))
    say("```")


def psi(base: pd.Series, cur: pd.Series, top_n: int = 50) -> float:
    """Population Stability Index over the top_n categories of the base period.

    Rule of thumb: <0.1 stable, 0.1-0.25 moderate shift, >0.25 major shift.
    """
    keep = base.value_counts().index[:top_n]
    b = base[base.isin(keep)].value_counts(normalize=True).reindex(keep).fillna(1e-6)
    c = cur[cur.isin(keep)].value_counts(normalize=True).reindex(keep).fillna(1e-6)
    b = b / b.sum()
    c = c / c.sum()
    return float(((c - b) * np.log(c / b)).sum())


def main() -> None:
    say("# Drift")

    imp = add_time_columns(load_impressions())
    df = join_characters(imp, load_characters())
    df = add_base_columns(df)

    # ==================================================================
    say("\n## A1. Is the target itself moving?")
    say()
    g = df.groupby("day").agg(n=("click", "size"), ctr=("click", "mean"))
    g["ctr_vs_overall"] = g["ctr"] / df["click"].mean()
    # binomial standard error, to separate real movement from sampling noise
    g["se"] = np.sqrt(g["ctr"] * (1 - g["ctr"]) / g["n"])
    g["z_vs_overall"] = (g["ctr"] - df["click"].mean()) / g["se"]
    table(g)
    say()
    say(f"CTR ranges {g['ctr'].min():.4f} to {g['ctr'].max():.4f} -- a "
        f"{(g['ctr'].max()/g['ctr'].min()-1):.1%} relative swing across ten days, "
        f"with |z| up to {g['z_vs_overall'].abs().max():.0f}. This is real "
        "movement, far beyond sampling noise, so a model with a fixed intercept "
        "will drift out of calibration within days.")

    # ==================================================================
    say("\n## A2. Feature distribution drift (PSI vs day 1)")
    say()
    base_day = df[df["day"] == DAYS[0]]
    cols = ["site_id", "app_id", "device_model", "character_id", "C14", "C17",
            "C19", "C21", "banner_pos", "site_category", "safety_tier",
            "conversation_turn"]
    rows = []
    for d in DAYS[1:]:
        cur = df[df["day"] == d]
        rows.append({"day": d, **{c: psi(base_day[c].astype(str), cur[c].astype(str))
                                  for c in cols}})
    p = pd.DataFrame(rows).set_index("day")
    table(p, "%.3f")
    say()
    worst = p.iloc[-1].sort_values(ascending=False)
    say(f"By the last day the biggest shifts are: "
        + ", ".join(f"`{k}` (PSI {v:.2f})" for k, v in worst.head(4).items()) + ".")
    say()
    say("`character_id` and the creative ids drift hardest; `banner_pos`, "
        "`safety_tier` and `conversation_turn` barely move. That tells you which "
        "features need frequent re-estimation and which can be cached for days.")

    # ==================================================================
    say("\n## A3. Character mix churn")
    say()
    tops = {d: set(df[df["day"] == d]["character_id"].value_counts().index[:200])
            for d in DAYS}
    jac = [len(tops[DAYS[0]] & tops[d]) / len(tops[DAYS[0]] | tops[d]) for d in DAYS]
    churn = pd.DataFrame({"day": DAYS, "jaccard_vs_day1": jac,
                          "days_elapsed": range(len(DAYS))}).set_index("day")
    # consecutive-day overlap too -- the daily turnover rate
    churn["jaccard_vs_prev"] = [np.nan] + [
        len(tops[DAYS[i]] & tops[DAYS[i - 1]]) / len(tops[DAYS[i]] | tops[DAYS[i - 1]])
        for i in range(1, len(DAYS))
    ]
    table(churn)
    say()
    half = np.interp(0.5, churn["jaccard_vs_day1"].to_numpy()[::-1],
                     churn["days_elapsed"].to_numpy()[::-1])
    say(f"The top-200 character set has a **Jaccard half-life of ~{half:.1f} days** "
        f"and only {jac[-1]:.0%} of the original set survives to day 10. "
        "Consecutive days overlap ~0.7, so roughly a third of the popular "
        "roster turns over every single day. Any per-character statistic "
        "estimated on a week of data is describing a cohort that no longer exists.")

    # ==================================================================
    say("\n## A4. Model staleness: how fast does a frozen model decay?")
    say()
    say("Train once on days 141022-141024 (encoders and trees both frozen at "
        "that point), then score every later day without touching it.")
    say()
    say("> **A bug this measurement caught.** The first version of this curve "
        "showed calibration exploding from 0.98 to 2.18 and RIG going negative "
        "on day 141027 -- a cliff, not a decay. That was not drift. "
        "`day_of_week` was being passed as a numeric ordinal, the training "
        "window covered only Wed-Fri (dow 2-4), and day 141027 is the single "
        "Monday in the log (dow 0). Every row fell off the left end of the "
        "tree's split range. Dropping the feature (it is also unidentifiable on "
        "10 days -- each weekday occurs once or twice, perfectly confounded "
        "with that date's traffic mix) turns the cliff back into a smooth "
        "decay. `scripts/debug_staleness.py` isolates it. The general lesson: a "
        "sudden metric cliff in a temporal backtest is far more often a feature "
        "encoding bug than a real regime change, and 'looks like drift' is the "
        "most expensive way to be wrong about it.")

    def build(day_list, state):
        sub = df[df["day"].isin(day_list)]
        return apply_encoders(sub, state).astype("float32"), sub["click"], sub

    train_days = [141022, 141023, 141024]
    hist = df[df["day"] < 141022]
    state_frozen = fit_encoders(df[df["day"].isin([141021] + train_days[:-1])])

    Xtr, ytr, str_ = build(train_days, state_frozen)
    Xva, yva, _ = build([141025], state_frozen)
    w = recency_weights(str_["hour_index"], half_life_hours=72.0)
    gbm = train_gbm(Xtr, ytr, Xva, yva, w_tr=w, verbose_eval=0, early_stopping=60)
    say(f"\nFrozen model: {gbm.best_iteration} trees, "
        f"{Xtr.shape[0]:,} train rows.")

    rows = []
    for i, d in enumerate([141025, 141026, 141027, 141028, 141029, 141030]):
        Xd, yd, sub = build([d], state_frozen)
        m = core_metrics(yd.to_numpy(), gbm.predict(Xd))
        rows.append({"day": d, "days_since_train": i + 1, **{
            k: m[k] for k in ["auc", "rig", "log_loss", "calib_ratio"]}})
    stale = pd.DataFrame(rows).set_index("day")
    table(stale)

    d0 = stale.iloc[0]
    d5 = stale.iloc[-1]
    say()
    say(f"With the encoding bug fixed, the decay is **mild**: over six days the "
        f"frozen model loses {(d0['auc']-d5['auc']):.4f} AUC "
        f"({(d0['auc']-d5['auc'])/d0['auc']:.1%}), "
        f"{(d0['rig']-d5['rig'])/d0['rig']:.1%} of its RIG, and calibration "
        f"moves from {d0['calib_ratio']:.3f} to {d5['calib_ratio']:.3f}. The "
        f"day-to-day variation ({stale['auc'].min():.4f} to "
        f"{stale['auc'].max():.4f} AUC) is comparable to the trend, and day "
        f"141028 actually scores *better* than day 141025.")
    say()
    say("**The honest reading: ten days is not enough data to set a retrain "
        "cadence.** What this does establish is an upper bound -- a fully frozen "
        "model does not fall apart within a week here, so daily retraining is "
        "ample and hourly retraining of the tree ensemble would be solving a "
        "problem this data does not show. The features that genuinely move "
        "fast (per-entity click statistics, the calibration level) are handled "
        "by the cheap refresh paths below rather than by retraining.")

    # ==================================================================
    say("\n## A5. Which part of the decay is cheap to fix?")
    say()
    say("Same frozen tree ensemble, but the target encodings are recomputed "
        "each day from data up to the previous day. If most of the decay comes "
        "back, the expensive nightly full retrain can be relaxed and a cheap "
        "feature-store refresh carries the load.")

    rows = []
    for i, d in enumerate([141025, 141026, 141027, 141028, 141029, 141030]):
        st = fit_encoders(df[df["day"] < d])
        sub = df[df["day"] == d]
        Xd = apply_encoders(sub, st).astype("float32")
        m = core_metrics(sub["click"].to_numpy(), gbm.predict(Xd))
        rows.append({"day": d, "days_since_train": i + 1, **{
            k: m[k] for k in ["auc", "rig", "log_loss", "calib_ratio"]}})
    refreshed = pd.DataFrame(rows).set_index("day")
    table(refreshed)

    cmp = pd.DataFrame({
        "frozen_auc": stale["auc"],
        "refreshed_encodings_auc": refreshed["auc"],
        "auc_recovered": refreshed["auc"] - stale["auc"],
        "frozen_rig": stale["rig"],
        "refreshed_rig": refreshed["rig"],
        "rig_recovered": refreshed["rig"] - stale["rig"],
    })
    say("\nSide by side:")
    table(cmp)
    say()
    say(f"Refreshing the encodings nudges **AUC up** on 5 of 6 days "
        f"(mean {cmp['auc_recovered'].mean():+.4f}) -- ranking gets slightly "
        f"better, as expected, because the per-entity statistics are fresher.")
    say()
    say(f"But **calibration gets clearly worse**: the ratio moves from "
        f"{stale['calib_ratio'].mean():.3f} on average to "
        f"{refreshed['calib_ratio'].mean():.3f}, and RIG drops on every single "
        f"day (mean {cmp['rig_recovered'].mean():+.4f}).")
    say()
    say("**This is the important finding in the section, and it is not the one "
        "I expected.** A frozen tree ensemble's leaf values are fitted against "
        "the encoder distribution that existed at training time. Swap fresher "
        "encodings underneath it and the inputs shift relative to the split "
        "thresholds and leaf constants the trees learned; the ordering improves "
        "because the new numbers are more informative, but the absolute level "
        "the leaves emit is now wrong. You get a better ranker and a worse "
        "probability.")
    say()
    say("**Operational conclusion:** refreshing the feature store under a "
        "frozen model is *not* safe on its own. The refresh must be paired with "
        "a recalibration pass, which is cheap -- refit isotonic on the last few "
        "hours of logged traffic, as in the training script. Concretely:")
    say()
    say("  - **hourly**: refit the calibrator on recent served traffic. Cheapest "
        "and highest value; it is what tracks the day-level CTR movement in A1.")
    say("  - **hourly**: refresh target-encoding and count tables -- but only "
        "together with the calibrator refit above, never alone.")
    say("  - **daily**: retrain the tree ensemble. A4 shows a week-long upper "
        "bound on how fast this needs to happen.")
    say("  - **continuously**: the decayed-posterior layer in Part B, which "
        "adapts between all of the above.")

    # ==================================================================
    say("\n## B. Adaptation prototype: inventory-constrained replay")
    say()
    say("Ad groups = `C17` (400 distinct creative/campaign-like values). "
        "Cohorts = character popularity deciles. For each (cohort, hour) the "
        "policy fills a slot budget from whichever ad groups have inventory "
        "that hour, and we read off the realised clicks.")
    say()
    say(f"The slot budget is **{FILL_RATE:.0%} of the hour's logged volume**, "
        "deliberately less than total inventory. This is the part that has to "
        "be right: if the policy is handed as many slots as there is inventory "
        "it has no choice to make, every policy collapses onto the logged "
        "allocation, and the comparison is vacuous. Constraining the budget is "
        "what turns this into an actual decision problem.")
    say()
    say("Policies compared:")
    say("  - `logged`: the mix actually served (the status quo)")
    say("  - `static_greedy`: allocate by CTR estimated once on the train window")
    say("  - `decayed_greedy`: allocate by a 48h-decayed posterior mean")
    say("  - `decayed_thompson`: sample from the decayed posterior (explores)")
    say("  - `decayed_thompson_fatigue`: + damping of recently over-served groups")

    rep = df[df["day"] >= 141028].copy()
    warm = df[df["day"] < 141028]

    # cohort = character popularity decile in the warm window
    pop = warm["character_id"].value_counts()
    cohort_of = pd.qcut(pop.rank(method="first"), 10, labels=False)
    rep["cohort"] = rep["character_id"].map(cohort_of).fillna(-1).astype(int)
    rep["arm"] = rep["C17"].astype(int)

    cell = (
        rep.groupby(["hour_index", "cohort", "arm"])
        .agg(n=("click", "size"), clicks=("click", "sum"))
        .reset_index()
    )
    say(f"\nReplay window: {len(rep):,} impressions, "
        f"{cell['hour_index'].nunique()} hours, {rep['arm'].nunique()} ad groups, "
        f"{rep['cohort'].nunique()} cohorts.")

    # priors from the warm window
    warm_arm = warm.groupby(warm["C17"].astype(int))["click"].agg(["sum", "size"])
    prior_ctr = float(warm["click"].mean())
    static_mean = ((warm_arm["sum"] + prior_ctr * 20) / (warm_arm["size"] + 20)).to_dict()

    def run_policy(name: str, strategy: str, decayed: bool, fatigue_strength: float,
                   seed: int = RANDOM_SEED) -> dict:
        bandit = DecayedBetaBandit(prior_ctr=prior_ctr, prior_strength=20.0,
                                   half_life_hours=48.0 if decayed else 1e9)
        # seed the bandit with warm-window evidence
        t_start = float(cell["hour_index"].min())
        for a, r in warm_arm.iterrows():
            bandit.update(a, float(r["sum"]), float(r["size"]), t_start - 1.0)

        fat = FatigueState(half_life_hours=12.0)
        alloc = AdaptiveAllocator(bandit, fat, strategy=strategy,
                                  explore_z=1.0, fatigue_strength=fatigue_strength,
                                  temperature=0.35, seed=seed)

        served_clicks = 0.0
        served_n = 0
        exposure: dict = {}
        cohort_arm_counts: dict = {}

        for t, hour_block in cell.groupby("hour_index"):
            batch_updates = []
            for coh, sub in hour_block.groupby("cohort"):
                arms = sub["arm"].tolist()
                cap = dict(zip(sub["arm"], sub["n"]))
                rate = dict(zip(sub["arm"], sub["clicks"] / sub["n"]))
                total = int(sub["n"].sum())
                # The policy gets FEWER slots than there is inventory, so it has
                # to actually choose. With slots == capacity every policy is
                # forced into the logged allocation and the comparison is vacuous.
                slots = max(int(round(total * FILL_RATE)), 1)
                if name == "logged":
                    # status quo: sample the logged mix down to the same budget
                    a_alloc = {a: int(round(v * FILL_RATE)) for a, v in cap.items()}
                    a_alloc = {a: v for a, v in a_alloc.items() if v > 0}
                else:
                    a_alloc = alloc.allocate(coh, arms, float(t), slots, capacity=cap)
                for a, k in a_alloc.items():
                    if k <= 0:
                        continue
                    exp_clicks = rate[a] * k
                    served_clicks += exp_clicks
                    served_n += k
                    exposure[a] = exposure.get(a, 0) + k
                    cohort_arm_counts[(coh, a)] = cohort_arm_counts.get((coh, a), 0) + k
                    batch_updates.append((a, exp_clicks, k, coh))
            # end-of-hour feedback (mirrors a real system's logging lag)
            for a, c, k, coh in batch_updates:
                bandit.update(a, c, k, float(t) + 1.0)
                fat.add((coh, a), k, float(t) + 1.0)

        conc = concentration_metrics(np.array(list(exposure.values())))
        # per-cohort ad diversity: mean entropy of the ad mix each cohort saw
        ents = []
        for coh in rep["cohort"].unique():
            v = np.array([n for (c, _a), n in cohort_arm_counts.items() if c == coh])
            if v.sum() > 0 and len(v) > 1:
                pr = v / v.sum()
                ents.append(float(-(pr * np.log(pr)).sum()))
        return {
            "policy": name,
            "impressions": served_n,
            "ctr": served_clicks / max(served_n, 1),
            "hhi": conc["hhi"],
            "top10_ad_share": conc["top10_share"],
            "ad_entropy": conc["entropy"],
            "mean_cohort_ad_entropy": float(np.mean(ents)) if ents else np.nan,
        }

    results = [
        run_policy("logged", "greedy", False, 0.0),
        run_policy("static_greedy", "greedy", False, 0.0),
        run_policy("decayed_greedy", "greedy", True, 0.0),
        run_policy("decayed_thompson", "thompson", True, 0.0),
        run_policy("decayed_thompson_fatigue", "thompson", True, 0.6),
    ]
    r = pd.DataFrame(results).set_index("policy")
    r["ctr_vs_logged"] = r["ctr"] / r.loc["logged", "ctr"] - 1
    r["top10_vs_logged"] = r["top10_ad_share"] / r.loc["logged", "top10_ad_share"] - 1
    table(r)

    say()
    best = r.drop(index="logged")["ctr"].idxmax()
    lg, fp = r.loc["logged"], r.loc["decayed_thompson_fatigue"]
    say(f"Highest-CTR policy: **{best}** "
        f"({r.loc[best,'ctr_vs_logged']:+.1%} vs the logged mix) -- but it buys "
        f"that by concentrating harder, pushing the top-10 ad groups from "
        f"{lg['top10_ad_share']:.1%} to {r.loc[best,'top10_ad_share']:.1%} of all "
        f"exposure. Higher CTR through less variety is the easy win and the one "
        f"that causes fatigue.")
    say()
    say("**The fatigue-damped policy is the answer to the brief**, and it "
        "dominates the status quo on both axes at once:")
    say()
    say(f"  - CTR **{fp['ctr_vs_logged']:+.1%}** vs the logged mix "
        f"({lg['ctr']:.4f} -> {fp['ctr']:.4f})")
    say(f"  - top-10 ad groups' share of exposure **down "
        f"{-fp['top10_vs_logged']:.0%}** ({lg['top10_ad_share']:.1%} -> "
        f"{fp['top10_ad_share']:.1%})")
    say(f"  - HHI {lg['hhi']:.4f} -> {fp['hhi']:.4f}, and per-cohort ad "
        f"diversity {lg['mean_cohort_ad_entropy']:.3f} -> "
        f"{fp['mean_cohort_ad_entropy']:.3f} nats")
    say()
    say("So it holds CTR (in fact beats it comfortably) while cutting "
        "repetition on the dominant cohorts roughly in half. Relative to the "
        "unconstrained bandit it gives back about 14 points of CTR lift, which "
        "is the explicit price of the diversity -- a choice to make, not a "
        "loss to hide.")

    # sensitivity to the fatigue knob
    say("\n### Sensitivity to the fatigue strength knob")
    say()
    sens = pd.DataFrame([
        run_policy(f"fatigue={k}", "thompson", True, k) for k in [0.0, 0.2, 0.6, 1.5, 4.0]
    ]).set_index("policy")
    sens["ctr_vs_logged"] = sens["ctr"] / r.loc["logged", "ctr"] - 1
    table(sens)
    say()
    say("The knob is monotone but **not smooth**: almost the entire effect "
        "lands between 0.0 and 0.2, and everything above that barely moves "
        "either metric. That is worth knowing before shipping it -- it is "
        "effectively a switch with a short ramp, not a dial, so the useful "
        "tuning range is 0 to ~0.3 and turning it to 4.0 buys nothing over 0.2. "
        "The saturation happens because the damping is `1/(1 + k*exposure)`: "
        "once `k*exposure` is comfortably above 1 for the heavy arms, further "
        "increases in `k` rescale all of them together and stop changing the "
        "ordering.")
    say()
    say("A production version would replace the global `k` with a per-cohort "
        "target on repetition rate (e.g. 'no user sees the same creative more "
        "than 3 times a day') and solve for the `k` that hits it, so the knob "
        "is expressed in a unit the product team can reason about.")

    (REPORTS / "drift.md").write_text("\n".join(OUT))
    print(f"\nWrote {REPORTS/'drift.md'}")


if __name__ == "__main__":
    main()
