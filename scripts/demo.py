"""
Live demo: rank a slate of candidate ads for two contrasting chat contexts.

Runs in a few seconds and needs only artifacts/ (no CSV load), so it is safe to
run on camera or in front of an interviewer.

    python scripts/demo.py

If artifacts/ is missing, run scripts/03_train.py first (or ./run_all.sh).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from simula.config import ARTIFACTS  # noqa: E402
from simula.ranker import (  # noqa: E402
    BudgetPacer,
    Candidate,
    CandidateRanker,
    Context,
    FatigueTracker,
)


def rule(title: str = "") -> None:
    print("\n" + "=" * 78)
    if title:
        print(f"  {title}")
        print("=" * 78)


# ----------------------------------------------------------------------
# A slate of candidate ads. Bids, brand-safety caps and impression counts are
# set to exercise every branch of the ranker: cold ads, a budget-throttled
# advertiser, and creatives that may not run next to mature personas.
# ----------------------------------------------------------------------
CANDIDATES = [
    #        id      advertiser  pos   C14   C15  C16   C17  C18 C19   C20  C21   bid  max_tier
    ("AD_MOBILE_GAME", "ADV_GAMES",  1, 15708, 300, 250, 1722, 2,  35, 100083,  79, 3.20, "mature"),
    ("AD_DATING_APP",  "ADV_DATE",   1, 20362, 320,  50, 2333, 0,  39,     -1, 157, 4.50, "mature"),
    ("AD_ENERGY_DRNK", "ADV_BEV",    0, 17753, 320,  50, 1993, 0,  35,     -1,  23, 1.80, "sfw"),
    ("AD_BANK_CARD",   "ADV_FIN",    0,  4687, 320,  50,  423, 3,  39,     -1,  32, 5.00, "sfw"),
    ("AD_NEW_RPG",     "ADV_GAMES",  1, 21611, 320,  50, 2480, 2,  35, 100084,  33, 2.40, "mature"),
    ("AD_STREAMING",   "ADV_MEDIA",  0, 18993, 320,  50, 2161, 0, 167,     -1,  71, 1.20, "suggestive"),
]

# How many impressions each ad has accumulated. Two are deliberately cold.
IMPRESSIONS = {
    "AD_MOBILE_GAME": 45_000,
    "AD_DATING_APP": 30_000,
    "AD_ENERGY_DRNK": 12_000,
    "AD_BANK_CARD": 22_000,
    "AD_NEW_RPG": 40,        # cold -- launched this morning
    "AD_STREAMING": 15,      # cold
}


def build_candidates() -> list[Candidate]:
    out = []
    for (aid, adv, pos, c14, c15, c16, c17, c18, c19, c20, c21, bid, cap) in CANDIDATES:
        out.append(
            Candidate(
                ad_id=aid, advertiser_id=adv, banner_pos=pos,
                C14=c14, C15=c15, C16=c16, C17=c17, C18=c18,
                C19=c19, C20=c20, C21=c21,
                bid_cpc=bid, max_safety_tier=cap,
            )
        )
    return out


CONTEXTS = {
    "A. SFW hero character, 2 messages into a fresh session": Context(
        character_id="baf21a857d", conversation_turn=2, session_msg_count=4,
        hour=14102914, site_id="8fda644b", site_domain="25d4cfcd",
        site_category="f028772b", app_id="ecad2386", app_domain="7801e8d9",
        app_category="07d7df22", device_id="a99f214a", device_ip="b264c159",
        device_model="be6db1d7", device_type=1, device_conn_type=0, C1=1005,
        safety_tier="sfw", creator_type="official", num_interactions=12000,
        created_at="2014-03-01",
    ),
    "B. Mature companion character, 18 messages into a roleplay": Context(
        character_id="c3607b676d", conversation_turn=18, session_msg_count=22,
        hour=14102923, site_id="1fbe01fe", site_domain="f3845767",
        site_category="28905ebd", app_id="ecad2386", app_domain="7801e8d9",
        app_category="07d7df22", device_id="a99f214a", device_ip="a4459495",
        device_model="517bef98", device_type=1, device_conn_type=0, C1=1005,
        safety_tier="mature", creator_type="community", num_interactions=496,
        created_at="2014-06-14",
    ),
}


def main() -> None:
    if not (ARTIFACTS / "model.txt").exists():
        print("No trained model found. Run scripts/03_train.py first "
              "(or ./run_all.sh).")
        sys.exit(1)

    rule("Loading model")
    t = time.perf_counter()
    from simula.serving import ModelScorer  # imported here so the timing is honest

    scorer = ModelScorer()
    print(f"LightGBM ({scorer.booster.num_trees()} trees) + "
          f"{sum(len(v) for v in scorer.te.values()):,} encoder keys "
          f"loaded in {time.perf_counter()-t:.2f}s")

    cands = build_candidates()

    # ADV_FIN has nearly exhausted today's budget -> should be throttled.
    pacer = BudgetPacer(
        daily_budget={"ADV_GAMES": 8000.0, "ADV_DATE": 8000.0, "ADV_BEV": 8000.0,
                      "ADV_FIN": 1000.0, "ADV_MEDIA": 8000.0},
        spend_today={"ADV_GAMES": 3200.0, "ADV_DATE": 1800.0, "ADV_BEV": 3900.0,
                     "ADV_FIN": 970.0, "ADV_MEDIA": 4100.0},
    )
    fatigue = FatigueTracker()
    ranker = CandidateRanker(
        scorer=scorer, pacer=pacer, fatigue=fatigue,
        ad_impression_counts={k: float(v) for k, v in IMPRESSIONS.items()},
        explore_z=1.0, prior_ctr=0.18,
    )

    for label, ctx in CONTEXTS.items():
        rule(label)
        print(f"character tier : {ctx.safety_tier}")
        print(f"conversation   : turn {ctx.conversation_turn} of "
              f"{ctx.session_msg_count}")
        print(f"surface        : site {ctx.site_id} / app {ctx.app_id}")

        t = time.perf_counter()
        ranked = ranker.rank(ctx, cands, fraction_of_day_elapsed=0.6)
        elapsed_ms = (time.perf_counter() - t) * 1000

        print(f"\nranked {len(cands)} candidates in {elapsed_ms:.2f}ms\n")
        hdr = f"{'#':>2}  {'ad':<16}{'pCTR':>7}{'+UCB':>7}{'bid':>6}" \
              f"{'pace':>6}{'fatig':>6}{'score':>8}   why"
        print(hdr)
        print("-" * len(hdr))
        for r in ranked:
            print(f"{r.rank:>2}  {r.ad_id:<16}{r.p_ctr:>7.4f}{r.p_ctr_ucb:>7.4f}"
                  f"{r.bid_cpc:>6.2f}{r.pacing_mult:>6.2f}{r.fatigue_mult:>6.2f}"
                  f"{r.final_score:>8.3f}   {r.reason}")

        blocked = getattr(ranker, "_last_blocked", [])
        if blocked:
            print("\nblocked before scoring (brand safety is a filter, not a "
                  "score penalty):")
            for c, why in blocked:
                print(f"    {c.ad_id:<16} {why}")

    # ------------------------------------------------------------------
    rule("Frequency capping across repeated requests")
    ctx = list(CONTEXTS.values())[0]
    user = ctx.device_ip
    print(f"Same user ({user}) sees the same slate 5 times in a row.")
    print("Watch the winner rotate as the fatigue multiplier decays:\n")
    for i in range(1, 6):
        ranked = ranker.rank(ctx, cands, fraction_of_day_elapsed=0.6)
        top = ranked[0]
        ranker.fatigue.record(user, top.ad_id, top.advertiser_id)
        print(f"  request {i}: serve {top.ad_id:<16} "
              f"(fatigue mult {top.fatigue_mult:.2f}, score {top.final_score:.3f})")

    rule("Latency under load")
    slate = cands * 9  # 54 candidates
    times = []
    for _ in range(200):
        t = time.perf_counter()
        ranker.rank(list(CONTEXTS.values())[1], slate, fraction_of_day_elapsed=0.6)
        times.append((time.perf_counter() - t) * 1000)
    a = np.array(times)
    print(f"{len(slate)}-candidate slate, 200 requests:")
    print(f"  p50 {np.percentile(a,50):.2f}ms   "
          f"p95 {np.percentile(a,95):.2f}ms   "
          f"p99 {np.percentile(a,99):.2f}ms   (budget: 50ms)")
    print()


if __name__ == "__main__":
    main()
