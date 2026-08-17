# Candidate ranking

## 1. Ranking real slates from held-out traffic

A 'slate' here is a set of held-out impressions that share the same request context but were served different ads. Grouping on a context key and ranking within the group gives a ranking metric on real labels, with no simulation and no assumption about what the retrieval layer would have proposed.
```
                            n_groups  recall@1    mrr   ndcg  group_auc  median_slate_size  mean_slate_size
context_key                                                                                                
site+app+character+hour         1851    0.5143 0.7401 0.8065     0.5518             1.0000           1.0692
site+app+device_model+hour      6445    0.4461 0.6570 0.7376     0.5698             1.0000           1.8888
site+character+hour             4616    0.5737 0.7644 0.8230     0.6525             1.0000           1.2394
app+character+hour              7897    0.6062 0.7788 0.8316     0.6916             1.0000           1.4131
```

`group_auc` is the honest headline: within a fixed context, how often does the model put a clicked ad above a non-clicked one. It sits well above 0.5, so the model is genuinely discriminating between ads for the *same* opportunity, not just between easy and hard contexts.

**Caveat, stated up front:** natural slates in this log are tiny (median size 1, mean 1.1-1.9), because the logged system served one ad per opportunity and contexts rarely repeat exactly. Only groups with both a click and a non-click are scoreable, so these metrics rest on a few thousand small groups and are correspondingly noisy. Looser context keys give bigger groups and better-looking numbers, but they also let genuine context differences leak back in -- which is why the tightest key (`site+app+character+hour`, group_auc 0.543) is the conservative read and the loosest (`app+character+hour`, 0.688) is the optimistic one. The truth is in between. A production system would get this metric properly from an interleaving or explore-bucket experiment, not from logged single-slot data.

Note `group_auc` is lower than the pooled AUC. That is expected and healthy: pooled AUC gets free credit for separating a high-CTR surface from a low-CTR one, which the ranker never has to do -- the surface is fixed at request time. Reporting only the pooled number would overstate how much value the ranker adds.

## 2. Is the ranker context-sensitive?

Take one fixed set of 40 candidate ads. Score it against 300 different randomly drawn contexts. If the model had no context x ad interaction, every context would produce an identical ordering and the pairwise Spearman correlation between orderings would be 1.0.

- pairwise Spearman between context orderings: mean 0.394, p10 -0.046, p90 0.828
- distinct ads winning the top slot across 300 contexts: **23 of 40**
- most frequent winner takes the top slot 27.0% of the time
- pCTR spread for the same ad across contexts: mean within-ad sd 0.0748 vs across-ad sd 0.0231

That last line is worth sitting with: the same ad's pCTR moves 3.2x more as the *context* changes than the average pCTR moves as the *ad* changes. Context dominates. Practically, this means most of the model's value is in deciding how much an impression is worth (pricing, pacing, whether to bid at all), and a smaller but real slice is in choosing between ads for it. Worth being honest about, because it also tells you where the next modelling effort pays off: richer ad-side features, not richer context features.

Orderings genuinely move with context, so per-impression ranking is doing real work rather than reproducing a global ad ranking.

## 3. Worked examples

Two contexts drawn to be as different as the data allows, run through the full ranker with safety gating, budget pacing, frequency capping and advertiser diversity switched on.

### Context A: sfw character, turn <=2 (fresh session)

- character: `fantasy_37ecea` -- A sarcastic champion sworn to the high banner, rebellious and unyielding in the field.
- tier `sfw`, creator `community`, 1,210 lifetime interactions
- turn 1 of 2, site `f9c69707`, app `ecad2386`, hour 14102900
```
     ad_id advertiser_id  p_ctr  p_ctr_ucb  bid_cpc     ev  pacing_mult  fatigue_mult  diversity_mult  final_score                                                                                    reason
rank                                                                                                                                                                                                        
1     AD09          ADV1 0.3731     0.3818   2.1000 0.8019       2.0000        1.0000          1.0000       1.6037                                                                        pacing boost x2.00
2     AD00          ADV0 0.3731     0.3775   2.5000 0.9437       1.0417        1.0000          1.0000       0.9830                                                                  scored on model estimate
3     AD06          ADV2 0.4863     0.4896   3.2000 1.5668       0.5128        1.0000          1.0000       0.8035                                                                     pacing throttle x0.51
4     AD03          ADV3 0.2652     0.2678   4.0000 1.0710       0.5952        1.0000          1.0000       0.6375                                                                     pacing throttle x0.60
5     AD05          ADV1 0.4162     0.4656   0.9500 0.4424       2.0000        1.0000          0.7000       0.6193     cold ad (n=60), +0.0495 explore bonus; pacing boost x2.00; advertiser-diversity x0.70
6     AD01          ADV1 0.3613     0.3667   1.1000 0.4034       2.0000        1.0000          0.4900       0.3953                                            pacing boost x2.00; advertiser-diversity x0.49
7     AD04          ADV0 0.1380     0.1431   1.7500 0.2504       1.0417        1.0000          0.7000       0.1826                                      scored on model estimate; advertiser-diversity x0.70
8     AD02          ADV2 0.3613     0.4238   0.8000 0.3391       0.5128        1.0000          0.7000       0.1217  cold ad (n=15), +0.0625 explore bonus; pacing throttle x0.51; advertiser-diversity x0.70
9     AD07          ADV3 0.1380     0.1493   1.4000 0.2090       0.5952        1.0000          0.7000       0.0871                                         pacing throttle x0.60; advertiser-diversity x0.70
10    AD08          ADV0 0.1833     0.1852   0.6000 0.1111       1.0417        1.0000          0.4900       0.0567                                      scored on model estimate; advertiser-diversity x0.49
```

### Context B: mature character, turn >=10 (deep roleplay)

- character: `mystery_39684c` -- A mysterious private investigator with a cold case file and a colder coffee, playful under the desk lamp.
- tier `mature`, creator `community`, 496 lifetime interactions
- turn 18 of 22, site `1fbe01fe`, app `ecad2386`, hour 14102900
```
     ad_id advertiser_id  p_ctr  p_ctr_ucb  bid_cpc     ev  pacing_mult  fatigue_mult  diversity_mult  final_score                                                                                 reason
rank                                                                                                                                                                                                     
1     AD09          ADV1 0.2072     0.2146   2.1000 0.4506       2.0000        1.0000          1.0000       0.9011                                                                     pacing boost x2.00
2     AD04          ADV0 0.2139     0.2200   1.7500 0.3849       1.0417        1.0000          1.0000       0.4010                                                               scored on model estimate
3     AD05          ADV1 0.2139     0.2558   0.9500 0.2430       2.0000        1.0000          0.7000       0.3402  cold ad (n=60), +0.0419 explore bonus; pacing boost x2.00; advertiser-diversity x0.70
4     AD01          ADV1 0.2072     0.2117   1.1000 0.2329       2.0000        1.0000          0.4900       0.2282                                         pacing boost x2.00; advertiser-diversity x0.49
5     AD07          ADV3 0.2072     0.2205   1.4000 0.3086       0.5952        1.0000          1.0000       0.1837                                                                  pacing throttle x0.60
```
Filtered before scoring: `AD00` (blocked: character tier 'mature' exceeds advertiser cap); `AD02` (blocked: character tier 'mature' exceeds advertiser cap); `AD03` (blocked: character tier 'mature' exceeds advertiser cap); `AD06` (blocked: character tier 'mature' exceeds advertiser cap); `AD08` (blocked: character tier 'mature' exceeds advertiser cap)

Reading example B: the mature character removes every advertiser whose `max_safety_tier` is `sfw` before any scoring happens -- a high bid cannot buy past a brand-safety rule. `ADV2` is throttled because it has spent 97% of its daily budget by midday. The two cold ads carry a visible UCB bonus, which is the ranker deliberately paying a little expected revenue to learn their true rate.

## 4. Ordering under uncertainty

When the model is unsure, the tie-break is deliberate rather than arbitrary. Sweeping the exploration weight on the same slate:
Averaged over 200 real contexts, so the numbers are not an artifact of one slate. `true_ev` uses the model's *unboosted* pCTR, i.e. the revenue we actually expect to collect; the gap between z=0 and higher z is the price paid for information.
```
           cold_ad_wins_top_slot  mean_rank_of_cold_ads  mean_true_ev_top1  ev_cost_vs_z0
explore_z                                                                                
0.0                       0.0150                 6.6300             0.3529         0.0000
0.5                       0.0400                 6.2850             0.3512        -0.0046
1.0                       0.0750                 5.9425             0.3458        -0.0198
2.0                       0.2100                 5.3025             0.3331        -0.0559
4.0                       0.3350                 4.6700             0.3190        -0.0958
```

The knob is smooth and the price is legible. At `z=1`, cold ads take the top slot 5x as often as at `z=0` (7.5% vs 1.5%) for a 2.0% give-back in expected revenue. At `z=4` a third of top slots go to cold ads, for a 9.6% give-back.

There is no free lunch and no cliff either -- exploration bought is roughly linear in revenue given up across this range, so the right `z` is a business decision about how fast the catalogue needs to be learned, not a hyperparameter to tune offline. Given that ~37% of creative ids in any held-out window are unseen (see the cold-start report), something in the `z=1` region is the defensible default: it keeps a meaningful learning rate on new creatives for a couple of percent of revenue.

Note the bonus is a function of the ad's own impression count, so it decays automatically as an ad accumulates data -- there is no separate 'new ad' code path to maintain and no threshold to tune.