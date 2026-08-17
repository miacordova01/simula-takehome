"""Candidate ranking.

The retrieval layer hands us N candidate ads for one impression opportunity.
Each candidate is a (banner_pos, C14..C21) tuple describing creative and
advertiser attributes; the context (character, conversation, site, app, device,
time) is fixed across the slate.

Ranking is NOT just "sort by pCTR". A production ranker has to reconcile four
things that pull in different directions:

  1. **Expected value.** pCTR x bid. Advertisers pay per click, so the quantity
     to maximise is revenue, not click probability. A 2% CTR ad at a $5 CPC
     beats a 6% CTR ad at $1.

  2. **Uncertainty.** pCTR from a cold advertiser is a guess. Sorting purely by
     the point estimate means we never learn -- the ranker keeps serving what it
     already believes is good and never discovers what is actually good. The
     score therefore carries an explicit uncertainty term and we optimise an
     upper confidence bound, which buys exploration in proportion to how little
     we know. See `ThompsonExplorer` for the sampling alternative.

  3. **Hard constraints.** A `mature` creative on an `sfw` character is a brand
     safety incident, not a ranking mistake. These are filters applied BEFORE
     scoring, never soft penalties -- a big enough pCTR must not be able to buy
     its way past a safety gate.

  4. **Cross-request state.** Budget pacing and frequency capping depend on what
     we served other users a minute ago. These are multiplicative modifiers on
     the score rather than filters, so a paced advertiser degrades gracefully
     instead of dropping off a cliff.

Final score:

    score = pCTR_adj * bid * pacing * fatigue * diversity

where pCTR_adj is the UCB-adjusted click estimate, and everything else is a
multiplier in (0, 1].
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# Domain objects
# ----------------------------------------------------------------------

SAFETY_ORDER = {"sfw": 0, "suggestive": 1, "mature": 2}


@dataclass
class Candidate:
    """One ad the retrieval layer proposes."""

    ad_id: str
    advertiser_id: str
    banner_pos: int
    # anonymized creative / advertiser attributes
    C14: int
    C15: int
    C16: int
    C17: int
    C18: int
    C19: int
    C20: int
    C21: int
    bid_cpc: float = 1.0
    # highest character safety tier this creative may be shown against
    max_safety_tier: str = "mature"
    # set by the advertiser: don't show me next to spicier personas than this
    min_safety_tier: str = "sfw"

    def as_row(self) -> dict:
        return {
            "banner_pos": self.banner_pos,
            "C14": self.C14, "C15": self.C15, "C16": self.C16, "C17": self.C17,
            "C18": self.C18, "C19": self.C19, "C20": self.C20, "C21": self.C21,
        }


@dataclass
class Context:
    """The fixed part of an ad opportunity."""

    character_id: str
    conversation_turn: int
    session_msg_count: int
    hour: int  # YYMMDDHH
    site_id: str
    site_domain: str
    site_category: str
    app_id: str
    app_domain: str
    app_category: str
    device_id: str
    device_ip: str
    device_model: str
    device_type: int
    device_conn_type: int
    C1: int
    # joined from characters.csv
    safety_tier: str = "sfw"
    creator_type: str = "community"
    num_interactions: int = 0
    created_at: str = "2014-01-01"

    def as_row(self) -> dict:
        d = dict(self.__dict__)
        return d


# ----------------------------------------------------------------------
# Cross-request state
# ----------------------------------------------------------------------


@dataclass
class BudgetPacer:
    """Keeps an advertiser's spend on an even glide path through the day.

    Without pacing, high-pCTR advertisers exhaust their daily budget in the first
    hours and the evening auction gets thin. The multiplier is the ratio of
    "where spend should be by now" to "where it actually is", clipped so a
    single lagging advertiser cannot dominate.

    This is deliberately a *soft* multiplier. A hard cutoff at budget exhaustion
    creates a discontinuity that shows up as a CTR cliff at a fixed hour each
    day; a smooth throttle spreads the same budget without that artifact.
    """

    daily_budget: dict[str, float] = field(default_factory=dict)
    spend_today: dict[str, float] = field(default_factory=dict)
    floor: float = 0.1
    ceil: float = 2.0

    def multiplier(self, advertiser_id: str, fraction_of_day_elapsed: float) -> float:
        budget = self.daily_budget.get(advertiser_id)
        if not budget:
            return 1.0
        spent = self.spend_today.get(advertiser_id, 0.0)
        if spent >= budget:
            return self.floor  # throttled hard, but never fully removed
        target = budget * max(fraction_of_day_elapsed, 1e-3)
        if target <= 0:
            return 1.0
        ratio = target / max(spent, 1e-6)
        return float(np.clip(ratio, self.floor, self.ceil))

    def record_spend(self, advertiser_id: str, amount: float) -> None:
        self.spend_today[advertiser_id] = self.spend_today.get(advertiser_id, 0.0) + amount


@dataclass
class FatigueTracker:
    """Frequency capping per (user, advertiser) and per (user, creative).

    55.9% of impressions go to a device_ip we have seen before, so repeat
    exposure is the norm, not a corner case. Effective CTR on a creative decays
    roughly geometrically with repeat views; we model that decay explicitly
    rather than using a hard cap, and hard-cap only as a backstop.
    """

    seen_creative: dict[tuple[str, str], int] = field(default_factory=dict)
    seen_advertiser: dict[tuple[str, str], int] = field(default_factory=dict)
    decay: float = 0.65          # each repeat view is worth 65% of the previous
    hard_cap_creative: int = 4
    hard_cap_advertiser: int = 10

    def multiplier(self, user_key: str, ad_id: str, advertiser_id: str) -> float:
        nc = self.seen_creative.get((user_key, ad_id), 0)
        na = self.seen_advertiser.get((user_key, advertiser_id), 0)
        if nc >= self.hard_cap_creative or na >= self.hard_cap_advertiser:
            return 0.0
        return float(self.decay**nc)

    def record(self, user_key: str, ad_id: str, advertiser_id: str) -> None:
        self.seen_creative[(user_key, ad_id)] = self.seen_creative.get((user_key, ad_id), 0) + 1
        self.seen_advertiser[(user_key, advertiser_id)] = (
            self.seen_advertiser.get((user_key, advertiser_id), 0) + 1
        )


# ----------------------------------------------------------------------
# Uncertainty / exploration
# ----------------------------------------------------------------------


def beta_posterior(clicks: float, impressions: float, prior_ctr: float,
                   prior_strength: float = 30.0) -> tuple[float, float]:
    """Beta posterior over an entity's CTR, anchored on a prior."""
    a = clicks + prior_ctr * prior_strength
    b = (impressions - clicks) + (1 - prior_ctr) * prior_strength
    return max(a, 1e-6), max(b, 1e-6)


def ucb_adjust(p: float, impressions: float, z: float = 1.0,
               prior_ctr: float = 0.18, prior_strength: float = 30.0) -> float:
    """Optimism in the face of uncertainty.

    Adds z standard deviations of the Beta posterior to the model's estimate.
    The width shrinks as ~1/sqrt(n), so a well-observed ad is scored on its point
    estimate and a brand-new one gets a bonus that decays automatically as it
    accumulates impressions. No explicit "graduation" rule needed -- the maths
    handles the handoff.
    """
    a, b = beta_posterior(p * impressions, impressions, prior_ctr, prior_strength)
    n = a + b
    sd = math.sqrt(a * b / (n * n * (n + 1)))
    return float(min(p + z * sd, 1.0))


class ThompsonExplorer:
    """Thompson sampling alternative to UCB.

    Preferred when you care about regret over a long horizon and can tolerate
    per-request randomness; UCB is preferred when you need deterministic,
    reproducible, explainable rankings (easier to debug and to explain to an
    advertiser asking why they lost an auction). Both are provided; the ranker
    defaults to UCB for that explainability reason.
    """

    def __init__(self, seed: int = 17, prior_ctr: float = 0.18,
                 prior_strength: float = 30.0) -> None:
        self.rng = np.random.default_rng(seed)
        self.prior_ctr = prior_ctr
        self.prior_strength = prior_strength

    def sample(self, p: float, impressions: float) -> float:
        a, b = beta_posterior(p * impressions, impressions,
                              self.prior_ctr, self.prior_strength)
        return float(self.rng.beta(a, b))


# ----------------------------------------------------------------------
# The ranker
# ----------------------------------------------------------------------


@dataclass
class RankedAd:
    ad_id: str
    advertiser_id: str
    rank: int
    p_ctr: float
    p_ctr_ucb: float
    bid_cpc: float
    ev: float
    pacing_mult: float
    fatigue_mult: float
    diversity_mult: float
    final_score: float
    reason: str


class CandidateRanker:
    """Scores and orders a candidate slate for one opportunity."""

    def __init__(
        self,
        scorer,                      # callable: (context, candidates) -> np.ndarray of pCTR
        pacer: BudgetPacer | None = None,
        fatigue: FatigueTracker | None = None,
        ad_impression_counts: dict[str, float] | None = None,
        explore_z: float = 1.0,
        prior_ctr: float = 0.18,
        enforce_safety: bool = True,
    ) -> None:
        self.scorer = scorer
        self.pacer = pacer or BudgetPacer()
        self.fatigue = fatigue or FatigueTracker()
        self.ad_impression_counts = ad_impression_counts or {}
        self.explore_z = explore_z
        self.prior_ctr = prior_ctr
        self.enforce_safety = enforce_safety

    # -- hard gates ----------------------------------------------------
    def _safety_ok(self, ctx: Context, cand: Candidate) -> tuple[bool, str]:
        """Two-sided brand-safety gate.

        Runs before scoring. An advertiser declares the spiciest persona they
        will sit next to (`max_safety_tier`); a creative can also require a
        minimum tier if it is itself adult inventory that must not appear on
        sfw surfaces.
        """
        ctx_t = SAFETY_ORDER.get(ctx.safety_tier, 0)
        if ctx_t > SAFETY_ORDER.get(cand.max_safety_tier, 2):
            return False, f"blocked: character tier '{ctx.safety_tier}' exceeds advertiser cap"
        if ctx_t < SAFETY_ORDER.get(cand.min_safety_tier, 0):
            return False, f"blocked: creative requires >= '{cand.min_safety_tier}' surface"
        return True, ""

    # -- main entry point ----------------------------------------------
    def rank(
        self,
        ctx: Context,
        candidates: list[Candidate],
        fraction_of_day_elapsed: float = 0.5,
        top_k: int | None = None,
    ) -> list[RankedAd]:
        if not candidates:
            return []

        # 1. hard filters first -- safety is never tradeable against pCTR
        eligible: list[Candidate] = []
        blocked: list[tuple[Candidate, str]] = []
        for c in candidates:
            if self.enforce_safety:
                ok, why = self._safety_ok(ctx, c)
                if not ok:
                    blocked.append((c, why))
                    continue
            eligible.append(c)

        if not eligible:
            return []

        # 2. one batched model call for the whole slate
        p = np.asarray(self.scorer(ctx, eligible), dtype=float)

        user_key = ctx.device_ip if ctx.device_id == "a99f214a" else ctx.device_id

        rows: list[RankedAd] = []
        seen_advertisers: dict[str, int] = {}
        for c, pc in zip(eligible, p):
            n_imp = self.ad_impression_counts.get(c.ad_id, 0.0)
            p_ucb = ucb_adjust(
                pc, n_imp, z=self.explore_z, prior_ctr=self.prior_ctr
            )
            pacing = self.pacer.multiplier(c.advertiser_id, fraction_of_day_elapsed)
            fat = self.fatigue.multiplier(user_key, c.ad_id, c.advertiser_id)
            ev = p_ucb * c.bid_cpc

            reason_bits = []
            if n_imp < 100:
                reason_bits.append(f"cold ad (n={int(n_imp)}), +{p_ucb-pc:.4f} explore bonus")
            if pacing < 0.95:
                reason_bits.append(f"pacing throttle x{pacing:.2f}")
            elif pacing > 1.05:
                reason_bits.append(f"pacing boost x{pacing:.2f}")
            if fat == 0.0:
                reason_bits.append("frequency cap hit")
            elif fat < 1.0:
                reason_bits.append(f"fatigue x{fat:.2f}")
            if not reason_bits:
                reason_bits.append("scored on model estimate")

            rows.append(
                RankedAd(
                    ad_id=c.ad_id,
                    advertiser_id=c.advertiser_id,
                    rank=0,
                    p_ctr=float(pc),
                    p_ctr_ucb=p_ucb,
                    bid_cpc=c.bid_cpc,
                    ev=ev,
                    pacing_mult=pacing,
                    fatigue_mult=fat,
                    diversity_mult=1.0,
                    final_score=ev * pacing * fat,
                    reason="; ".join(reason_bits),
                )
            )

        # 3. advertiser diversity: demote the 2nd+ candidate from the same
        #    advertiser so one buyer cannot occupy the whole slate.
        rows.sort(key=lambda r: r.final_score, reverse=True)
        for r in rows:
            k = seen_advertisers.get(r.advertiser_id, 0)
            if k > 0:
                r.diversity_mult = 0.7**k
                r.final_score *= r.diversity_mult
                r.reason += f"; advertiser-diversity x{r.diversity_mult:.2f}"
            seen_advertisers[r.advertiser_id] = k + 1

        rows.sort(key=lambda r: r.final_score, reverse=True)
        for i, r in enumerate(rows, start=1):
            r.rank = i

        self._last_blocked = blocked
        return rows[:top_k] if top_k else rows

    def explain(self, ranked: list[RankedAd]) -> pd.DataFrame:
        return pd.DataFrame([r.__dict__ for r in ranked])
