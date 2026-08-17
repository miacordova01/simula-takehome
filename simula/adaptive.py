"""Adaptive serving layer.

The static model is retrained on a batch cadence (nightly, realistically). In
between refreshes the world moves: ad rotations change, characters churn in
popularity, novelty decays. This module is the thin online layer that tracks
those moves between retrains.

Core object is a **time-decayed Beta posterior** per arm. Instead of counting
(clicks, impressions) forever, both counts decay exponentially with a half-life.
That does three useful things at once:

  * it tracks drift -- an arm whose CTR fell last week stops being credited for
    click history from before the fall;
  * it re-inflates uncertainty over time, so an arm that has not been served
    recently automatically becomes explorable again (novelty/recovery), which a
    plain cumulative counter never does;
  * it bounds memory -- one float pair per arm, no event log.

`DecayedBetaBandit` is the state store. `AdaptiveAllocator` turns it into an
allocation over arms with an explicit fatigue penalty, which is what the drift
prototype evaluates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class DecayedBetaBandit:
    """Per-arm Beta posterior with exponentially decayed evidence.

    half_life_hours controls how fast old evidence is forgotten. 48h was picked
    from the observed drift rate: the top-200 character set turns over with a
    Jaccard half-life of roughly two days in this log, so evidence older than
    that is describing a different traffic mix.
    """

    prior_ctr: float = 0.18
    prior_strength: float = 20.0
    half_life_hours: float = 48.0
    clicks: dict = field(default_factory=dict)
    impressions: dict = field(default_factory=dict)
    last_update: dict = field(default_factory=dict)

    def _decay_to(self, arm, t: float) -> None:
        """Age an arm's counters forward to time t (hours)."""
        t0 = self.last_update.get(arm)
        if t0 is None:
            self.clicks[arm] = 0.0
            self.impressions[arm] = 0.0
            self.last_update[arm] = t
            return
        dt = t - t0
        if dt <= 0:
            return
        f = 0.5 ** (dt / self.half_life_hours)
        self.clicks[arm] = self.clicks.get(arm, 0.0) * f
        self.impressions[arm] = self.impressions.get(arm, 0.0) * f
        self.last_update[arm] = t

    def update(self, arm, clicks: float, impressions: float, t: float) -> None:
        self._decay_to(arm, t)
        self.clicks[arm] = self.clicks.get(arm, 0.0) + clicks
        self.impressions[arm] = self.impressions.get(arm, 0.0) + impressions

    def posterior(self, arm, t: float) -> tuple[float, float]:
        self._decay_to(arm, t)
        c = self.clicks.get(arm, 0.0)
        n = self.impressions.get(arm, 0.0)
        a = c + self.prior_ctr * self.prior_strength
        b = (n - c) + (1 - self.prior_ctr) * self.prior_strength
        return max(a, 1e-6), max(b, 1e-6)

    def mean(self, arm, t: float) -> float:
        a, b = self.posterior(arm, t)
        return a / (a + b)

    def sd(self, arm, t: float) -> float:
        a, b = self.posterior(arm, t)
        n = a + b
        return math.sqrt(a * b / (n * n * (n + 1)))

    def sample(self, arm, t: float, rng: np.random.Generator) -> float:
        a, b = self.posterior(arm, t)
        return float(rng.beta(a, b))

    def ucb(self, arm, t: float, z: float = 1.0) -> float:
        return self.mean(arm, t) + z * self.sd(arm, t)


@dataclass
class FatigueState:
    """Recent exposure of an arm within a cohort, exponentially decayed.

    Used to damp an arm that a cohort has already seen a lot of recently. This
    is the knob that trades a little CTR for a lot less repetition.
    """

    half_life_hours: float = 12.0
    exposure: dict = field(default_factory=dict)
    last_update: dict = field(default_factory=dict)

    def _decay_to(self, key, t: float) -> None:
        t0 = self.last_update.get(key)
        if t0 is None:
            self.exposure[key] = 0.0
            self.last_update[key] = t
            return
        dt = t - t0
        if dt > 0:
            self.exposure[key] = self.exposure.get(key, 0.0) * (
                0.5 ** (dt / self.half_life_hours)
            )
            self.last_update[key] = t

    def add(self, key, n: float, t: float) -> None:
        self._decay_to(key, t)
        self.exposure[key] = self.exposure.get(key, 0.0) + n

    def get(self, key, t: float) -> float:
        self._decay_to(key, t)
        return self.exposure.get(key, 0.0)


class AdaptiveAllocator:
    """Turns bandit state into an allocation over arms for one cohort-hour.

    strategy:
      'greedy'   -- all weight on the posterior-mean argmax (no exploration)
      'ucb'      -- weight proportional to the upper confidence bound
      'thompson' -- weight proportional to a posterior draw

    fatigue_strength scales how hard recently over-served arms are damped.
    0 disables it and recovers a pure CTR-maximising allocator.
    """

    def __init__(
        self,
        bandit: DecayedBetaBandit,
        fatigue: FatigueState | None = None,
        strategy: str = "thompson",
        explore_z: float = 1.0,
        fatigue_strength: float = 0.0,
        temperature: float = 1.0,
        seed: int = 17,
    ) -> None:
        self.bandit = bandit
        self.fatigue = fatigue or FatigueState()
        self.strategy = strategy
        self.explore_z = explore_z
        self.fatigue_strength = fatigue_strength
        self.temperature = temperature
        self.rng = np.random.default_rng(seed)

    def scores(self, cohort, arms: list, t: float) -> np.ndarray:
        b = self.bandit
        if self.strategy == "greedy":
            s = np.array([b.mean(a, t) for a in arms])
        elif self.strategy == "ucb":
            s = np.array([b.ucb(a, t, self.explore_z) for a in arms])
        elif self.strategy == "thompson":
            s = np.array([b.sample(a, t, self.rng) for a in arms])
        else:
            raise ValueError(self.strategy)

        if self.fatigue_strength > 0:
            exp = np.array([self.fatigue.get((cohort, a), t) for a in arms])
            # multiplicative damping: an arm seen `e` times recently is worth
            # 1/(1+k*e) of its score. Saturating, so it never fully bans an arm.
            s = s / (1.0 + self.fatigue_strength * exp)
        return s

    def allocate(self, cohort, arms: list, t: float, n_slots: int,
                 capacity: dict | None = None) -> dict:
        """Distribute n_slots across arms, respecting per-arm capacity."""
        if not arms:
            return {}
        s = self.scores(cohort, arms, t)
        s = np.maximum(s, 1e-9) ** (1.0 / max(self.temperature, 1e-6))
        w = s / s.sum()

        alloc = {a: 0 for a in arms}
        # proportional allocation, then greedily fix up against capacity
        want = np.floor(w * n_slots).astype(int)
        for a, k in zip(arms, want):
            cap = capacity.get(a, k) if capacity else k
            alloc[a] = int(min(k, cap))

        remaining = n_slots - sum(alloc.values())
        if remaining > 0:
            order = np.argsort(-w)
            for i in order:
                if remaining <= 0:
                    break
                a = arms[i]
                cap = capacity.get(a, remaining) if capacity else remaining
                room = cap - alloc[a]
                if room > 0:
                    take = int(min(room, remaining))
                    alloc[a] += take
                    remaining -= take
        return {a: k for a, k in alloc.items() if k > 0}


def concentration_metrics(counts: np.ndarray) -> dict[str, float]:
    """How concentrated is an exposure distribution?"""
    c = np.asarray(counts, dtype=float)
    c = c[c > 0]
    if c.size == 0:
        return {"hhi": np.nan, "top10_share": np.nan, "entropy": np.nan, "gini": np.nan}
    p = c / c.sum()
    hhi = float((p**2).sum())
    top10 = float(np.sort(p)[::-1][:10].sum())
    ent = float(-(p * np.log(p)).sum())
    srt = np.sort(p)
    n = len(srt)
    gini = float((2 * np.arange(1, n + 1) - n - 1).dot(srt) / n)
    return {"hhi": hhi, "top10_share": top10, "entropy": ent, "gini": gini}
