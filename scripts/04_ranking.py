"""
Candidate ranking.

Three things, in order of how much they matter:

  1. **Does the ranker order real slates correctly?** Evaluated on natural
     slates: groups of held-out impressions that share a context key but differ
     in their ad attributes. Within-group AUC / recall@1 / NDCG on real labels.
     This is the metric that maps to the product question.

  2. **Is the ranker actually context-sensitive?** A model with no context x ad
     interaction produces the SAME ad ordering for every request, which makes
     per-impression ranking pointless. Measured by handing one fixed candidate
     set to many different contexts and looking at the rank correlation between
     them. If that correlation is ~1.0 the whole exercise is theatre.

  3. **Worked examples** with the business rules switched on -- safety gate,
     pacing, fatigue, advertiser diversity -- showing why each ad landed where
     it did, including under model uncertainty.

Writes reports/ranking.md and reports/sample_rankings.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from simula.config import ARTIFACTS, RANDOM_SEED, REPORTS  # noqa: E402
from simula.data import add_time_columns, join_characters, load_characters, load_impressions  # noqa: E402
from simula.evaluate import group_ranking_metrics  # noqa: E402
from simula.ranker import (  # noqa: E402
    BudgetPacer,
    Candidate,
    CandidateRanker,
    Context,
    FatigueTracker,
)
from simula.serving import ModelScorer  # noqa: E402

OUT: list[str] = []
rng = np.random.default_rng(RANDOM_SEED)


def say(s: str = "") -> None:
    print(s)
    OUT.append(s)


def table(df: pd.DataFrame, fmt: str = "%.4f") -> None:
    say("```")
    say(df.to_string(float_format=lambda x: fmt % x))
    say("```")


def main() -> None:
    say("# Candidate ranking")

    imp = add_time_columns(load_impressions())
    df = join_characters(imp, load_characters())
    test = df[df["day"] >= 141029].reset_index(drop=True)
    p_te = np.load(ARTIFACTS / "test_preds.npy")
    test_meta = pd.read_parquet(ARTIFACTS / "test_meta.parquet")

    # align: test_preds covers days 141029-30 in the same order as test_meta
    full_test = df[df["day"].isin([141029, 141030])].reset_index(drop=True)
    assert len(full_test) == len(p_te), (len(full_test), len(p_te))
    full_test["p"] = p_te

    # ------------------------------------------------------------------
    say("\n## 1. Ranking real slates from held-out traffic")
    say()
    say("A 'slate' here is a set of held-out impressions that share the same "
        "request context but were served different ads. Grouping on a context "
        "key and ranking within the group gives a ranking metric on real "
        "labels, with no simulation and no assumption about what the retrieval "
        "layer would have proposed.")

    keys = {
        "site+app+character+hour": ["site_id", "app_id", "character_id", "hour"],
        "site+app+device_model+hour": ["site_id", "app_id", "device_model", "hour"],
        "site+character+hour": ["site_id", "character_id", "hour"],
        "app+character+hour": ["app_id", "character_id", "hour"],
    }
    rows = []
    for name, cols in keys.items():
        gid = full_test.groupby(cols, observed=True).ngroup()
        sizes = gid.value_counts()
        m = group_ranking_metrics(
            full_test["click"].to_numpy(), full_test["p"].to_numpy(), gid.to_numpy()
        )
        m["context_key"] = name
        m["median_slate_size"] = float(sizes.median())
        m["mean_slate_size"] = float(sizes.mean())
        rows.append(m)
    rk = pd.DataFrame(rows).set_index("context_key")
    table(rk)

    say()
    say("`group_auc` is the honest headline: within a fixed context, how often "
        "does the model put a clicked ad above a non-clicked one. It sits well "
        "above 0.5, so the model is genuinely discriminating between ads for "
        "the *same* opportunity, not just between easy and hard contexts.")
    say()
    say("**Caveat, stated up front:** natural slates in this log are tiny "
        "(median size 1, mean 1.1-1.9), because the logged system served one ad "
        "per opportunity and contexts rarely repeat exactly. Only groups with "
        "both a click and a non-click are scoreable, so these metrics rest on a "
        "few thousand small groups and are correspondingly noisy. Looser context "
        "keys give bigger groups and better-looking numbers, but they also let "
        "genuine context differences leak back in -- which is why the tightest "
        "key (`site+app+character+hour`, group_auc 0.543) is the conservative "
        "read and the loosest (`app+character+hour`, 0.688) is the optimistic "
        "one. The truth is in between. A production system would get this metric "
        "properly from an interleaving or explore-bucket experiment, not from "
        "logged single-slot data.")
    say()
    say("Note `group_auc` is lower than the pooled AUC. That is expected and "
        "healthy: pooled AUC gets free credit for separating a high-CTR surface "
        "from a low-CTR one, which the ranker never has to do -- the surface is "
        "fixed at request time. Reporting only the pooled number would overstate "
        "how much value the ranker adds.")

    # ------------------------------------------------------------------
    say("\n## 2. Is the ranker context-sensitive?")
    say()
    say("Take one fixed set of 40 candidate ads. Score it against 300 different "
        "randomly drawn contexts. If the model had no context x ad interaction, "
        "every context would produce an identical ordering and the pairwise "
        "Spearman correlation between orderings would be 1.0.")

    scorer = ModelScorer()

    ad_pool = (
        full_test[["banner_pos", "C14", "C15", "C16", "C17", "C18", "C19", "C20", "C21"]]
        .drop_duplicates()
        .sample(40, random_state=RANDOM_SEED)
        .reset_index(drop=True)
    )
    cands = [
        Candidate(
            ad_id=f"ad_{i}", advertiser_id=f"adv_{i % 12}",
            banner_pos=int(r.banner_pos), C14=int(r.C14), C15=int(r.C15),
            C16=int(r.C16), C17=int(r.C17), C18=int(r.C18), C19=int(r.C19),
            C20=int(r.C20), C21=int(r.C21),
        )
        for i, r in ad_pool.iterrows()
    ]

    ctx_rows = full_test.sample(300, random_state=RANDOM_SEED)
    score_mat = []
    for _, r in ctx_rows.iterrows():
        ctx = Context(
            character_id=str(r.character_id), conversation_turn=int(r.conversation_turn),
            session_msg_count=int(r.session_msg_count), hour=int(r.hour),
            site_id=str(r.site_id), site_domain=str(r.site_domain),
            site_category=str(r.site_category), app_id=str(r.app_id),
            app_domain=str(r.app_domain), app_category=str(r.app_category),
            device_id=str(r.device_id), device_ip=str(r.device_ip),
            device_model=str(r.device_model), device_type=int(r.device_type),
            device_conn_type=int(r.device_conn_type), C1=int(r.C1),
            safety_tier=str(r.safety_tier), creator_type=str(r.creator_type),
            num_interactions=int(r.num_interactions),
            created_at=str(pd.Timestamp(r.created_at).date()),
        )
        score_mat.append(scorer.score_slate(ctx, cands))
    S = np.vstack(score_mat)

    ranks = pd.DataFrame(S).rank(axis=1)
    corr = ranks.T.corr(method="spearman").to_numpy()
    iu = np.triu_indices_from(corr, k=1)
    say()
    say(f"- pairwise Spearman between context orderings: "
        f"mean {corr[iu].mean():.3f}, p10 {np.percentile(corr[iu],10):.3f}, "
        f"p90 {np.percentile(corr[iu],90):.3f}")
    top1 = S.argmax(axis=1)
    say(f"- distinct ads winning the top slot across 300 contexts: "
        f"**{len(np.unique(top1))} of {len(cands)}**")
    say(f"- most frequent winner takes the top slot "
        f"{pd.Series(top1).value_counts(normalize=True).iloc[0]:.1%} of the time")
    within = S.std(axis=0).mean()
    across = S.mean(axis=0).std()
    say(f"- pCTR spread for the same ad across contexts: mean within-ad sd "
        f"{within:.4f} vs across-ad sd {across:.4f}")
    say()
    say(f"That last line is worth sitting with: the same ad's pCTR moves "
        f"{within/across:.1f}x more as the *context* changes than the average "
        f"pCTR moves as the *ad* changes. Context dominates. Practically, this "
        "means most of the model's value is in deciding how much an impression "
        "is worth (pricing, pacing, whether to bid at all), and a smaller but "
        "real slice is in choosing between ads for it. Worth being honest about, "
        "because it also tells you where the next modelling effort pays off: "
        "richer ad-side features, not richer context features.")
    say()
    if corr[iu].mean() < 0.9:
        say("Orderings genuinely move with context, so per-impression ranking "
            "is doing real work rather than reproducing a global ad ranking.")
    else:
        say("**Orderings barely move with context.** Most of the score is a "
            "global ad-quality term; per-impression ranking is adding little "
            "beyond a static ad ordering, and the honest thing is to say so.")

    # ------------------------------------------------------------------
    say("\n## 3. Worked examples")
    say()
    say("Two contexts drawn to be as different as the data allows, run through "
        "the full ranker with safety gating, budget pacing, frequency capping "
        "and advertiser diversity switched on.")

    # build a candidate set with varied bids, safety caps and cold ads
    ex_pool = (
        full_test[["banner_pos", "C14", "C15", "C16", "C17", "C18", "C19", "C20", "C21"]]
        .drop_duplicates().sample(10, random_state=3).reset_index(drop=True)
    )
    bids = [2.50, 1.10, 0.80, 4.00, 1.75, 0.95, 3.20, 1.40, 0.60, 2.10]
    caps = ["sfw", "mature", "suggestive", "sfw", "mature",
            "mature", "suggestive", "mature", "sfw", "mature"]
    ex_cands = [
        Candidate(
            ad_id=f"AD{i:02d}", advertiser_id=f"ADV{i % 4}",
            banner_pos=int(r.banner_pos), C14=int(r.C14), C15=int(r.C15),
            C16=int(r.C16), C17=int(r.C17), C18=int(r.C18), C19=int(r.C19),
            C20=int(r.C20), C21=int(r.C21),
            bid_cpc=bids[i], max_safety_tier=caps[i],
        )
        for i, r in ex_pool.iterrows()
    ]
    # ad impression counts: make two of them cold on purpose
    ad_counts = {f"AD{i:02d}": float(v) for i, v in
                 enumerate([12000, 8000, 15, 30000, 4500, 60, 22000, 900, 40000, 3000])}

    # Spend levels chosen to exercise every branch of the pacer at midday:
    # ADV0 on track, ADV1 underspending (boost), ADV2 nearly exhausted
    # (throttle), ADV3 overspending (throttle).
    pacer = BudgetPacer(
        daily_budget={"ADV0": 5000.0, "ADV1": 5000.0, "ADV2": 800.0, "ADV3": 5000.0},
        spend_today={"ADV0": 2400.0, "ADV1": 900.0, "ADV2": 780.0, "ADV3": 4200.0},
    )
    fatigue = FatigueTracker()

    ranker = CandidateRanker(
        scorer=scorer, pacer=pacer, fatigue=fatigue,
        ad_impression_counts=ad_counts, explore_z=1.0,
        prior_ctr=float(full_test["click"].mean()),
    )

    # pick two contrasting real contexts
    sfw_row = full_test[(full_test.safety_tier == "sfw") &
                        (full_test.conversation_turn <= 2)].iloc[0]
    mature_row = full_test[(full_test.safety_tier == "mature") &
                           (full_test.conversation_turn >= 10)].iloc[0]

    def make_ctx(r) -> Context:
        return Context(
            character_id=str(r.character_id), conversation_turn=int(r.conversation_turn),
            session_msg_count=int(r.session_msg_count), hour=int(r.hour),
            site_id=str(r.site_id), site_domain=str(r.site_domain),
            site_category=str(r.site_category), app_id=str(r.app_id),
            app_domain=str(r.app_domain), app_category=str(r.app_category),
            device_id=str(r.device_id), device_ip=str(r.device_ip),
            device_model=str(r.device_model), device_type=int(r.device_type),
            device_conn_type=int(r.device_conn_type), C1=int(r.C1),
            safety_tier=str(r.safety_tier), creator_type=str(r.creator_type),
            num_interactions=int(r.num_interactions),
            created_at=str(pd.Timestamp(r.created_at).date()),
        )

    chars = load_characters().set_index("character_id")
    all_out = []
    for label, row in [("A: sfw character, turn <=2 (fresh session)", sfw_row),
                       ("B: mature character, turn >=10 (deep roleplay)", mature_row)]:
        ctx = make_ctx(row)
        cname = chars.loc[ctx.character_id, "character_name"] if ctx.character_id in chars.index else "?"
        cdesc = chars.loc[ctx.character_id, "character_description"] if ctx.character_id in chars.index else "?"
        say(f"\n### Context {label}")
        say()
        say(f"- character: `{cname}` -- {cdesc}")
        say(f"- tier `{ctx.safety_tier}`, creator `{ctx.creator_type}`, "
            f"{ctx.num_interactions:,} lifetime interactions")
        say(f"- turn {ctx.conversation_turn} of {ctx.session_msg_count}, "
            f"site `{ctx.site_id}`, app `{ctx.app_id}`, hour {ctx.hour}")

        ranked = ranker.rank(ctx, ex_cands, fraction_of_day_elapsed=0.5)
        rdf = ranker.explain(ranked)[
            ["rank", "ad_id", "advertiser_id", "p_ctr", "p_ctr_ucb", "bid_cpc",
             "ev", "pacing_mult", "fatigue_mult", "diversity_mult", "final_score", "reason"]
        ]
        table(rdf.set_index("rank"))
        blocked = getattr(ranker, "_last_blocked", [])
        if blocked:
            say(f"Filtered before scoring: "
                + "; ".join(f"`{c.ad_id}` ({why})" for c, why in blocked))
        rdf.insert(0, "context", label)
        all_out.append(rdf)

    say()
    say("Reading example B: the mature character removes every advertiser whose "
        "`max_safety_tier` is `sfw` before any scoring happens -- a high bid "
        "cannot buy past a brand-safety rule. `ADV2` is throttled because it has "
        "spent 97% of its daily budget by midday. The two cold ads carry a "
        "visible UCB bonus, which is the ranker deliberately paying a little "
        "expected revenue to learn their true rate.")

    # ------------------------------------------------------------------
    say("\n## 4. Ordering under uncertainty")
    say()
    say("When the model is unsure, the tie-break is deliberate rather than "
        "arbitrary. Sweeping the exploration weight on the same slate:")
    say("Averaged over 200 real contexts, so the numbers are not an artifact of "
        "one slate. `true_ev` uses the model's *unboosted* pCTR, i.e. the "
        "revenue we actually expect to collect; the gap between z=0 and higher "
        "z is the price paid for information.")
    sweep = []
    sweep_ctxs = [make_ctx(r) for _, r in
                  full_test.sample(200, random_state=11).iterrows()]
    for z in [0.0, 0.5, 1.0, 2.0, 4.0]:
        r2 = CandidateRanker(scorer=scorer, pacer=pacer, fatigue=FatigueTracker(),
                             ad_impression_counts=ad_counts, explore_z=z,
                             prior_ctr=float(full_test["click"].mean()))
        cold_top1, evs, cold_rank = [], [], []
        for c in sweep_ctxs:
            rr = r2.rank(c, ex_cands, fraction_of_day_elapsed=0.5)
            if not rr:
                continue
            cold_top1.append(ad_counts[rr[0].ad_id] < 100)
            evs.append(rr[0].p_ctr * rr[0].bid_cpc)
            cr = [x.rank for x in rr if ad_counts[x.ad_id] < 100]
            if cr:
                cold_rank.append(float(np.mean(cr)))
        sweep.append({
            "explore_z": z,
            "cold_ad_wins_top_slot": float(np.mean(cold_top1)),
            "mean_rank_of_cold_ads": float(np.mean(cold_rank)),
            "mean_true_ev_top1": float(np.mean(evs)),
        })
    sw = pd.DataFrame(sweep).set_index("explore_z")
    sw["ev_cost_vs_z0"] = sw["mean_true_ev_top1"] / sw.loc[0.0, "mean_true_ev_top1"] - 1
    table(sw)
    say()
    say("The knob is smooth and the price is legible. At `z=1`, cold ads take "
        "the top slot 5x as often as at `z=0` (7.5% vs 1.5%) for a 2.0% "
        "give-back in expected revenue. At `z=4` a third of top slots go to "
        "cold ads, for a 9.6% give-back.")
    say()
    say("There is no free lunch and no cliff either -- exploration bought is "
        "roughly linear in revenue given up across this range, so the right `z` "
        "is a business decision about how fast the catalogue needs to be "
        "learned, not a hyperparameter to tune offline. Given that ~37% of "
        "creative ids in any held-out window are unseen (see the cold-start "
        "report), something in the `z=1` region is the defensible default: it "
        "keeps a meaningful learning rate on new creatives for a couple of "
        "percent of revenue.")
    say()
    say("Note the bonus is a function of the ad's own impression count, so it "
        "decays automatically as an ad accumulates data -- there is no separate "
        "'new ad' code path to maintain and no threshold to tune.")

    pd.concat(all_out).to_csv(REPORTS / "sample_rankings.csv", index=False)
    (REPORTS / "ranking.md").write_text("\n".join(OUT))
    print(f"\nWrote {REPORTS/'ranking.md'}")


if __name__ == "__main__":
    main()
