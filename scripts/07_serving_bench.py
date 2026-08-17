"""
Serving latency benchmark.

The brief asks for <50ms p99. Rather than assert an architecture, this measures
the actual scoring path on this machine and reports where the time goes, so the
architecture section is grounded in numbers.

Measures, for slate sizes 1..100:
  - feature assembly time (context vector + per-candidate fill)
  - LightGBM batched predict time
  - end-to-end ranker time including safety gate, UCB, pacing, fatigue, sort

Also measures the win from caching the context vector across candidates, which
is the single biggest structural optimisation available.

Writes reports/latency.md
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from simula.config import RANDOM_SEED, REPORTS  # noqa: E402
from simula.data import add_time_columns, join_characters, load_characters, load_impressions  # noqa: E402
from simula.ranker import BudgetPacer, Candidate, CandidateRanker, Context, FatigueTracker  # noqa: E402
from simula.serving import ModelScorer  # noqa: E402

OUT: list[str] = []


def say(s: str = "") -> None:
    print(s)
    OUT.append(s)


def table(df: pd.DataFrame, fmt: str = "%.3f") -> None:
    say("```")
    say(df.to_string(float_format=lambda x: fmt % x))
    say("```")


def main() -> None:
    say("# Serving latency")

    imp = add_time_columns(load_impressions())
    df = join_characters(imp, load_characters())
    sample = df[df["day"] >= 141029].sample(600, random_state=RANDOM_SEED)

    t0 = time.perf_counter()
    scorer = ModelScorer()
    say(f"\nModel + encoder load: {time.perf_counter()-t0:.2f}s "
        f"(one-time, at process start)")
    say(f"- trees: {scorer.booster.num_trees()}, features: {scorer.n_features}")
    say(f"- encoder tables held in memory: {len(scorer.te)} target-encoding maps, "
        f"{sum(len(v) for v in scorer.te.values()):,} total keys")

    ad_pool = (
        df[["banner_pos", "C14", "C15", "C16", "C17", "C18", "C19", "C20", "C21"]]
        .drop_duplicates().sample(200, random_state=RANDOM_SEED).reset_index(drop=True)
    )

    def make_cands(n: int) -> list[Candidate]:
        return [
            Candidate(
                ad_id=f"ad_{i}", advertiser_id=f"adv_{i%15}",
                banner_pos=int(r.banner_pos), C14=int(r.C14), C15=int(r.C15),
                C16=int(r.C16), C17=int(r.C17), C18=int(r.C18), C19=int(r.C19),
                C20=int(r.C20), C21=int(r.C21),
                bid_cpc=1.0 + (i % 7) * 0.4,
            )
            for i, r in ad_pool.head(n).iterrows()
        ]

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

    contexts = [make_ctx(r) for _, r in sample.iterrows()]

    # ------------------------------------------------------------------
    say("\n## Scoring path, by slate size")
    say()
    say("Each row is 400 requests against real held-out contexts. Times are "
        "per request (the whole slate), single-threaded process, warm cache.")

    rows = []
    for n in [1, 5, 10, 20, 50, 100]:
        cands = make_cands(n)
        for c in contexts[:20]:
            scorer.score_slate(c, cands)
        lat, feat, pred = [], [], []
        for c in contexts[:400]:
            s0 = scorer.stats.feature_us, scorer.stats.predict_us
            t = time.perf_counter()
            scorer.score_slate(c, cands)
            lat.append((time.perf_counter() - t) * 1000)
            feat.append(scorer.stats.feature_us - s0[0])
            pred.append(scorer.stats.predict_us - s0[1])
        a = np.array(lat)
        rows.append({
            "slate_size": n,
            "p50_ms": np.percentile(a, 50),
            "p95_ms": np.percentile(a, 95),
            "p99_ms": np.percentile(a, 99),
            "max_ms": a.max(),
            "feature_us_mean": float(np.mean(feat)),
            "predict_us_mean": float(np.mean(pred)),
        })
    lat_df = pd.DataFrame(rows).set_index("slate_size")
    table(lat_df)

    # ------------------------------------------------------------------
    say("\n## Full ranker (scoring + safety gate + UCB + pacing + fatigue + sort)")
    say()
    ranker = CandidateRanker(
        scorer=scorer,
        pacer=BudgetPacer(daily_budget={f"adv_{i}": 10000.0 for i in range(15)},
                          spend_today={f"adv_{i}": 1000.0 * i for i in range(15)}),
        fatigue=FatigueTracker(),
        ad_impression_counts={f"ad_{i}": float(10 ** (i % 5)) for i in range(200)},
        explore_z=1.0,
    )
    rows = []
    for n in [10, 20, 50, 100]:
        cands = make_cands(n)
        for c in contexts[:20]:
            ranker.rank(c, cands)
        lat = []
        for c in contexts[:400]:
            t = time.perf_counter()
            ranker.rank(c, cands, fraction_of_day_elapsed=0.5)
            lat.append((time.perf_counter() - t) * 1000)
        a = np.array(lat)
        rows.append({"slate_size": n, "p50_ms": np.percentile(a, 50),
                     "p95_ms": np.percentile(a, 95), "p99_ms": np.percentile(a, 99),
                     "max_ms": a.max()})
    full_df = pd.DataFrame(rows).set_index("slate_size")
    table(full_df)

    p99_50 = full_df.loc[50, "p99_ms"]
    say()
    say(f"**p99 at a 50-candidate slate: {p99_50:.2f}ms** against a 50ms budget, "
        f"leaving {50 - p99_50:.0f}ms for network, retrieval, auction and logging.")

    # ------------------------------------------------------------------
    say("\n## Where the time goes")
    say()
    n = 50
    cands = make_cands(n)
    for c in contexts[:20]:
        scorer.score_slate(c, cands)

    t = time.perf_counter()
    for c in contexts[:400]:
        scorer.context_vector(c)
    ctx_us = (time.perf_counter() - t) / 400 * 1e6

    row = lat_df.loc[n]
    breakdown = pd.DataFrame({
        "stage": ["context vector (once per request)",
                  "per-candidate feature fill",
                  "LightGBM batched predict",
                  "ranker business logic"],
        "microseconds": [
            ctx_us,
            row["feature_us_mean"] - ctx_us,
            row["predict_us_mean"],
            max(full_df.loc[n, "p50_ms"] * 1000 - row["feature_us_mean"]
                - row["predict_us_mean"], 0.0),
        ],
    })
    breakdown["share"] = breakdown["microseconds"] / breakdown["microseconds"].sum()
    table(breakdown.set_index("stage"))

    # ------------------------------------------------------------------
    say("\n## Value of the context/ad split")
    say()
    say("The context vector is computed once and broadcast across the slate. "
        "If it were recomputed per candidate instead:")
    naive_us = ctx_us * n + (row["feature_us_mean"] - ctx_us) + row["predict_us_mean"]
    actual_us = row["feature_us_mean"] + row["predict_us_mean"]
    cmp = pd.DataFrame({
        "approach": ["context computed once (implemented)",
                     "context recomputed per candidate"],
        "us_per_request": [actual_us, naive_us],
        "ms_per_request": [actual_us / 1000, naive_us / 1000],
    }).set_index("approach")
    table(cmp)
    say(f"\nThe split is worth **{naive_us/actual_us:.1f}x** at a 50-candidate "
        f"slate, and the gap widens linearly with slate size.")

    # ------------------------------------------------------------------
    say("\n## Throughput")
    say()
    qps = 1000.0 / full_df.loc[50, "p50_ms"]
    say(f"- single core, 50-candidate slates: **~{qps:,.0f} requests/sec**")
    say(f"- a 16-core box at 60% utilisation: ~{qps*16*0.6:,.0f} req/s")
    say("- LightGBM releases the GIL during predict, so a thread pool scales "
        "close to linearly until memory bandwidth binds.")

    (REPORTS / "latency.md").write_text("\n".join(OUT))
    print(f"\nWrote {REPORTS/'latency.md'}")


if __name__ == "__main__":
    main()
