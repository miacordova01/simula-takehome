# Cold start

## Q1. How much traffic is cold?

Measured on the held-out window against everything seen before it.
```
         entity  cold_rows  cold_share  cold_ctr  warm_ctr
2     device_ip     165548      0.6124    0.1680    0.1685
6           C14     103535      0.3830    0.1370    0.1876
7           C17      98470      0.3643    0.1370    0.1861
1     device_id      46640      0.1725    0.1472    0.1726
0  character_id       2490      0.0092    0.1647    0.1682
4        app_id       1486      0.0055    0.0700    0.1688
3       site_id        863      0.0032    0.1101    0.1684
5  device_model        417      0.0015    0.1463    0.1682
```

Cold *characters* are rare (0.9% of held-out rows -- the catalogue is small and mostly already served), but cold **device_ips are 61.2% of traffic**. 'New user' is the dominant cold-start case by two orders of magnitude, not 'new character'. Effort should be sized accordingly: the character cold-start question in the brief is real but small, and the user cold-start question is the one that moves revenue.

The creative ids are also substantially cold (`C14`/`C17` ~37-38% unseen) and notably lower-CTR when cold, which is the ad-rotation effect: new creatives enter constantly and start below the average.

Characters in the catalogue never served at all: 3 (these are the true zero-impression cold start).

## Q2. Which features carry signal before any click exists?

Restrict to test rows whose entity is unseen in training, then ask which columns still separate clicks. A feature is cold-start-usable only if it is an *attribute* of the entity rather than a learned statistic about it.

Cold-user slice: 165,548 rows, CTR 0.1680
```
               column  cold_user_auc
0             site_id         0.6524
1              app_id         0.6112
15                C21         0.6032
12                C18         0.5956
8                 C14         0.5852
4        device_model         0.5813
11                C17         0.5783
3        app_category         0.5723
13                C19         0.5566
2       site_category         0.5532
14                C20         0.5426
20       character_id         0.5401
10                C16         0.5341
9                 C15         0.5331
21   num_interactions         0.5325
6    device_conn_type         0.5290
16         banner_pos         0.5220
7                  C1         0.5163
5         device_type         0.5158
17        hour_of_day         0.5157
18        safety_tier         0.5139
22      char_age_days         0.5022
19       creator_type         0.5003
23  conversation_turn         0.4997
24  session_msg_count         0.4993
```

For a brand-new user we still have the full publisher surface (site_id AUC ~0.65), the creative attributes, and the device model. That is why the cold-device slice barely loses accuracy in the main model: user identity was never carrying much of the signal.

## Q3. Bootstrapping a brand-new character

Two candidate bootstraps, compared against each other on out-of-time data:

  A. **Metadata backoff** -- predict the character's safety_tier CTR.
  B. **Content model** -- TF-IDF over `character_description`, ridge regression onto per-character CTR, trained on characters seen in the training window and applied to held-out characters.

Simulated by holding out a random 20% of characters entirely: they are removed from the encoder's history, so their test rows look brand new.

Held-out characters: 999; their test rows: 58,436
Characters with >=30 training impressions used to fit the content model: 2,614
Held-out characters with >=30 test impressions for scoring: 449
```
                     method  corr_with_actual    mse
0              global prior               NaN 0.0028
1       safety_tier backoff            0.2609 0.0026
2  description TF-IDF ridge            0.4773 0.0023
```

The description text beats metadata backoff -- worth shipping a content tower for character cold start.

## Q4. When has an entity graduated off the cold-start path?

Graduation is not a vibe -- it is the point where the entity's own click history predicts its future better than the prior does. Procedure: bucket characters by how many impressions they had in the training window, then compare two predictors of their *test-window* CTR: their own smoothed training CTR vs the safety-tier prior. The crossover is the threshold.

With prior_weight=0 (0 = raw CTR, no shrinkage):
```
                   n_characters  mse_own_history  mse_tier_prior  own_better_by
train_impressions                                                              
(25, 50]                     31         0.007952        0.006464      -0.001488
(50, 100]                   399         0.006721        0.004699      -0.002021
(100, 200]                  553         0.004075        0.003897      -0.000178
(200, 400]                  681         0.002674        0.002476      -0.000198
(400, 800]                  512         0.001540        0.001771       0.000230
(800, 100000]                84         0.001153        0.002091       0.000938
```

With prior_weight=25 (0 = raw CTR, no shrinkage):
```
                   n_characters  mse_own_history  mse_tier_prior  own_better_by
train_impressions                                                              
(25, 50]                     31         0.006391        0.006464       0.000073
(50, 100]                   399         0.005685        0.004699      -0.000986
(100, 200]                  553         0.003867        0.003897       0.000030
(200, 400]                  681         0.002562        0.002476      -0.000086
(400, 800]                  512         0.001516        0.001771       0.000255
(800, 100000]                84         0.001154        0.002091       0.000937
```

- prior_weight=0: own history first beats the tier prior at **(400, 800]** impressions, and stays ahead above it.
- prior_weight=25: own history first beats the tier prior at **(25, 50]** impressions, and stays ahead above it.

Two honest readings. With **no** shrinkage a character needs roughly **400 impressions** before its own click rate is a better predictor than its tier prior -- below that the raw rate is mostly noise and actively harmful. With shrinkage the penalty for using own-history early largely disappears (the worst bucket goes from -0.0020 to -0.0010 MSE), because the estimator *is* the prior when data is thin and slides continuously toward the entity's own rate as evidence arrives. Shrinkage does not move the crossover much; what it does is make being on the wrong side of it nearly costless.

**Operational conclusion: do not build a discrete cold-start path with a graduation event.** A shrunk estimator plus a UCB exploration bonus gives a continuous handoff, removes the threshold as a tuning parameter, and eliminates the discontinuity in serving behaviour that a hard cutoff would cause. The `seen_*` flags in the model let the trees learn a separate response surface for the genuinely-unseen case without us hand-coding one.