# Drift

## A1. Is the target itself moving?

```
             n    ctr  ctr_vs_overall     se  z_vs_overall
day                                                       
141021  111065 0.1792          0.9931 0.0012       -1.0851
141022  143836 0.1666          0.9235 0.0010      -14.0498
141023  104452 0.1916          1.0621 0.0012        9.2019
141024   89378 0.1843          1.0216 0.0013        3.0050
141025   90218 0.1925          1.0671 0.0013        9.2271
141026  103948 0.1950          1.0806 0.0012       11.8330
141027   86788 0.1955          1.0833 0.0013       11.1677
141028  142909 0.1654          0.9165 0.0010      -15.3400
141029  104450 0.1727          0.9572 0.0012       -6.6070
141030   22956 0.1655          0.9174 0.0025       -6.0733
```

CTR ranges 0.1654 to 0.1955 -- a 18.2% relative swing across ten days, with |z| up to 15. This is real movement, far beyond sampling noise, so a model with a fixed intercept will drift out of calibration within days.

## A2. Feature distribution drift (PSI vs day 1)

```
        site_id  app_id  device_model  character_id   C14   C17   C19   C21  banner_pos  site_category  safety_tier  conversation_turn
day                                                                                                                                   
141022    0.230   0.238         0.059         0.009 2.072 1.940 0.836 1.305       0.002          0.038        0.000              0.001
141023    0.217   0.331         0.028         0.009 4.472 3.841 0.625 1.718       0.005          0.026        0.000              0.001
141024    0.133   0.579         0.070         0.015 4.502 3.879 0.951 1.747       0.007          0.022        0.000              0.001
141025    0.234   0.688         0.031         0.028 5.945 5.865 1.362 3.028       0.016          0.049        0.000              0.001
141026    0.262   0.646         0.039         0.039 6.085 5.907 1.326 3.207       0.013          0.043        0.001              0.001
141027    0.240   0.677         0.053         0.051 6.080 5.948 1.469 3.115       0.004          0.042        0.001              0.001
141028    0.266   0.596         0.068         0.074 6.189 6.341 1.731 3.096       0.031          0.070        0.001              0.001
141029    0.264   0.440         0.078         0.112 6.475 6.482 1.503 1.981       0.031          0.049        0.001              0.001
141030    0.608   0.512         0.238         0.219 6.642 6.844 1.652 1.990       0.013          0.354        0.002              0.002
```

By the last day the biggest shifts are: `C17` (PSI 6.84), `C14` (PSI 6.64), `C21` (PSI 1.99), `C19` (PSI 1.65).

`character_id` and the creative ids drift hardest; `banner_pos`, `safety_tier` and `conversation_turn` barely move. That tells you which features need frequent re-estimation and which can be cached for days.

## A3. Character mix churn

```
        jaccard_vs_day1  days_elapsed  jaccard_vs_prev
day                                                   
141021           1.0000             0              NaN
141022           0.7544             1           0.7544
141023           0.6736             2           0.6878
141024           0.5748             3           0.6393
141025           0.4235             4           0.5873
141026           0.3605             5           0.6327
141027           0.3201             6           0.6461
141028           0.2821             7           0.6667
141029           0.2308             8           0.6949
141030           0.1396             9           0.5810
```

The top-200 character set has a **Jaccard half-life of ~3.5 days** and only 14% of the original set survives to day 10. Consecutive days overlap ~0.7, so roughly a third of the popular roster turns over every single day. Any per-character statistic estimated on a week of data is describing a cohort that no longer exists.

## A4. Model staleness: how fast does a frozen model decay?

Train once on days 141022-141024 (encoders and trees both frozen at that point), then score every later day without touching it.

> **A bug this measurement caught.** The first version of this curve showed calibration exploding from 0.98 to 2.18 and RIG going negative on day 141027 -- a cliff, not a decay. That was not drift. `day_of_week` was being passed as a numeric ordinal, the training window covered only Wed-Fri (dow 2-4), and day 141027 is the single Monday in the log (dow 0). Every row fell off the left end of the tree's split range. Dropping the feature (it is also unidentifiable on 10 days -- each weekday occurs once or twice, perfectly confounded with that date's traffic mix) turns the cliff back into a smooth decay. `scripts/debug_staleness.py` isolates it. The general lesson: a sudden metric cliff in a temporal backtest is far more often a feature encoding bug than a real regime change, and 'looks like drift' is the most expensive way to be wrong about it.

Frozen model: 23 trees, 337,666 train rows.
```
        days_since_train    auc    rig  log_loss  calib_ratio
day                                                          
141025                 1 0.6825 0.0594    0.4608       0.9802
141026                 2 0.6763 0.0546    0.4664       0.9636
141027                 3 0.6895 0.0632    0.4628       0.9540
141028                 4 0.6970 0.0653    0.4192       1.0552
141029                 5 0.6784 0.0550    0.4348       1.0334
141030                 6 0.6768 0.0516    0.4256       1.0238
```

With the encoding bug fixed, the decay is **mild**: over six days the frozen model loses 0.0057 AUC (0.8%), 13.2% of its RIG, and calibration moves from 0.980 to 1.024. The day-to-day variation (0.6763 to 0.6970 AUC) is comparable to the trend, and day 141028 actually scores *better* than day 141025.

**The honest reading: ten days is not enough data to set a retrain cadence.** What this does establish is an upper bound -- a fully frozen model does not fall apart within a week here, so daily retraining is ample and hourly retraining of the tree ensemble would be solving a problem this data does not show. The features that genuinely move fast (per-entity click statistics, the calibration level) are handled by the cheap refresh paths below rather than by retraining.

## A5. Which part of the decay is cheap to fix?

Same frozen tree ensemble, but the target encodings are recomputed each day from data up to the previous day. If most of the decay comes back, the expensive nightly full retrain can be relaxed and a cheap feature-store refresh carries the load.
```
        days_since_train    auc    rig  log_loss  calib_ratio
day                                                          
141025                 1 0.6843 0.0580    0.4615       1.0716
141026                 2 0.6801 0.0546    0.4664       1.0782
141027                 3 0.6943 0.0629    0.4630       1.1030
141028                 4 0.6883 0.0346    0.4329       1.3079
141029                 5 0.6815 0.0443    0.4398       1.1971
141030                 6 0.6834 0.0508    0.4260       1.1536
```

Side by side:
```
        frozen_auc  refreshed_encodings_auc  auc_recovered  frozen_rig  refreshed_rig  rig_recovered
day                                                                                                 
141025      0.6825                   0.6843         0.0017      0.0594         0.0580        -0.0014
141026      0.6763                   0.6801         0.0039      0.0546         0.0546        -0.0001
141027      0.6895                   0.6943         0.0048      0.0632         0.0629        -0.0003
141028      0.6970                   0.6883        -0.0087      0.0653         0.0346        -0.0307
141029      0.6784                   0.6815         0.0031      0.0550         0.0443        -0.0107
141030      0.6768                   0.6834         0.0066      0.0516         0.0508        -0.0008
```

Refreshing the encodings nudges **AUC up** on 5 of 6 days (mean +0.0019) -- ranking gets slightly better, as expected, because the per-entity statistics are fresher.

But **calibration gets clearly worse**: the ratio moves from 1.002 on average to 1.152, and RIG drops on every single day (mean -0.0073).

**This is the important finding in the section, and it is not the one I expected.** A frozen tree ensemble's leaf values are fitted against the encoder distribution that existed at training time. Swap fresher encodings underneath it and the inputs shift relative to the split thresholds and leaf constants the trees learned; the ordering improves because the new numbers are more informative, but the absolute level the leaves emit is now wrong. You get a better ranker and a worse probability.

**Operational conclusion:** refreshing the feature store under a frozen model is *not* safe on its own. The refresh must be paired with a recalibration pass, which is cheap -- refit isotonic on the last few hours of logged traffic, as in the training script. Concretely:

  - **hourly**: refit the calibrator on recent served traffic. Cheapest and highest value; it is what tracks the day-level CTR movement in A1.
  - **hourly**: refresh target-encoding and count tables -- but only together with the calibrator refit above, never alone.
  - **daily**: retrain the tree ensemble. A4 shows a week-long upper bound on how fast this needs to happen.
  - **continuously**: the decayed-posterior layer in Part B, which adapts between all of the above.

## B. Adaptation prototype: inventory-constrained replay

Ad groups = `C17` (400 distinct creative/campaign-like values). Cohorts = character popularity deciles. For each (cohort, hour) the policy fills a slot budget from whichever ad groups have inventory that hour, and we read off the realised clicks.

The slot budget is **50% of the hour's logged volume**, deliberately less than total inventory. This is the part that has to be right: if the policy is handed as many slots as there is inventory it has no choice to make, every policy collapses onto the logged allocation, and the comparison is vacuous. Constraining the budget is what turns this into an actual decision problem.

Policies compared:
  - `logged`: the mix actually served (the status quo)
  - `static_greedy`: allocate by CTR estimated once on the train window
  - `decayed_greedy`: allocate by a 48h-decayed posterior mean
  - `decayed_thompson`: sample from the decayed posterior (explores)
  - `decayed_thompson_fatigue`: + damping of recently over-served groups

Replay window: 270,315 impressions, 54 hours, 251 ad groups, 11 cohorts.
```
                          impressions    ctr    hhi  top10_ad_share  ad_entropy  mean_cohort_ad_entropy  ctr_vs_logged  top10_vs_logged
policy                                                                                                                                 
logged                         130434 0.1675 0.0219          0.3774      4.3109                  4.0457         0.0000           0.0000
static_greedy                  135169 0.2229 0.0315          0.4632      4.0291                  3.9412         0.3302           0.2275
decayed_greedy                 135169 0.2234 0.0311          0.4598      4.0392                  3.9493         0.3333           0.2184
decayed_thompson               135169 0.2243 0.0298          0.4520      4.0657                  3.9784         0.3387           0.1976
decayed_thompson_fatigue       135169 0.1999 0.0112          0.2114      4.7804                  4.7050         0.1930          -0.4399
```

Highest-CTR policy: **decayed_thompson** (+33.9% vs the logged mix) -- but it buys that by concentrating harder, pushing the top-10 ad groups from 37.7% to 45.2% of all exposure. Higher CTR through less variety is the easy win and the one that causes fatigue.

**The fatigue-damped policy is the answer to the brief**, and it dominates the status quo on both axes at once:

  - CTR **+19.3%** vs the logged mix (0.1675 -> 0.1999)
  - top-10 ad groups' share of exposure **down 44%** (37.7% -> 21.1%)
  - HHI 0.0219 -> 0.0112, and per-cohort ad diversity 4.046 -> 4.705 nats

So it holds CTR (in fact beats it comfortably) while cutting repetition on the dominant cohorts roughly in half. Relative to the unconstrained bandit it gives back about 14 points of CTR lift, which is the explicit price of the diversity -- a choice to make, not a loss to hide.

### Sensitivity to the fatigue strength knob

```
             impressions    ctr    hhi  top10_ad_share  ad_entropy  mean_cohort_ad_entropy  ctr_vs_logged
policy                                                                                                   
fatigue=0.0       135169 0.2243 0.0298          0.4520      4.0657                  3.9784         0.3387
fatigue=0.2       135169 0.2010 0.0114          0.2163      4.7700                  4.6404         0.1999
fatigue=0.6       135169 0.1999 0.0112          0.2114      4.7804                  4.7050         0.1930
fatigue=1.5       135169 0.1993 0.0111          0.2103      4.7843                  4.7300         0.1898
fatigue=4.0       135169 0.1989 0.0111          0.2101      4.7858                  4.7388         0.1871
```

The knob is monotone but **not smooth**: almost the entire effect lands between 0.0 and 0.2, and everything above that barely moves either metric. That is worth knowing before shipping it -- it is effectively a switch with a short ramp, not a dial, so the useful tuning range is 0 to ~0.3 and turning it to 4.0 buys nothing over 0.2. The saturation happens because the damping is `1/(1 + k*exposure)`: once `k*exposure` is comfortably above 1 for the heavy arms, further increases in `k` rescale all of them together and stop changing the ordering.

A production version would replace the global `k` with a per-cohort target on repetition rate (e.g. 'no user sees the same creative more than 3 times a day') and solve for the `k` that hits it, so the knob is expressed in a unit the product team can reason about.