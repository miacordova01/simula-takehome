# CTR model: training and evaluation

Loaded 1,000,000 rows in 1.0s

## Building time-aware features

Each day is encoded using only days strictly before it, mirroring a feature store rebuilt by a nightly batch job. Day 141021 is dropped because it has no history.
```
```
Encoded 888,935 rows x 65 features in 31.7s
- train 618,620 rows (CTR 0.1860)
- valid 142,909 rows (CTR 0.1654)
- test  127,406 rows (CTR 0.1714)

## Baseline 1: predict the base rate

```
                   value
n            127406.0000
actual_ctr        0.1714
pred_ctr          0.1860
auc               0.5000
pr_auc            0.1714
log_loss          0.4588
norm_entropy      1.0016
rig              -0.0016
calib_ratio       1.0851
ece               0.0146
```

## Baseline 2: hashed logistic regression on raw categoricals

trained in 128.7s on a 1,048,576-dim hashed space
```
                   value
n            127406.0000
actual_ctr        0.1714
pred_ctr          0.1743
auc               0.6998
pr_auc            0.3277
log_loss          0.4252
norm_entropy      0.9281
rig               0.0719
calib_ratio       1.0170
ece               0.0250
```

## Model: LightGBM on time-aware target encodings

Recency weighting: exponential, 72h half-life (oldest train row weighted 0.252 vs newest 1.000)
```
```
trained 94 trees in 20.2s
Two recalibration strategies:
- fixed isotonic fitted on the validation day: ECE 0.01443 -> 0.01839, calib ratio 1.0052 -> 0.8946
- rolling isotonic (12h window, 2h attribution lag): ECE 0.01443 -> 0.01262, calib ratio 1.0052 -> 0.9534

The fixed calibrator **hurts**: the validation day ran at CTR 0.1654 against the test window's 0.1714, so it bakes in yesterday's level and systematically underpredicts. This is a concrete instance of the drift problem, and the reason the production answer is a short rolling recalibration rather than a frozen one.

Using **rolling isotonic** downstream.

### Head-to-head on the held-out test days
```
                                    n  actual_ctr  pred_ctr    auc  pr_auc  log_loss  norm_entropy     rig  calib_ratio    ece
base_rate                 127406.0000      0.1714    0.1860 0.5000  0.1714    0.4588        1.0016 -0.0016       1.0851 0.0146
hashed_logreg             127406.0000      0.1714    0.1743 0.6998  0.3277    0.4252        0.9281  0.0719       1.0170 0.0250
lightgbm_raw              127406.0000      0.1714    0.1723 0.7104  0.3412    0.4165        0.9092  0.0908       1.0052 0.0144
lightgbm_fixed_isotonic   127406.0000      0.1714    0.1534 0.7099  0.3328    0.4194        0.9155  0.0845       0.8946 0.0184
lightgbm_rolling_isotonic 127406.0000      0.1714    0.1634 0.7081  0.3366    0.4181        0.9126  0.0874       0.9534 0.0126
```

LightGBM improves RIG over the hashed baseline by **21.7%** relative, and AUC by +0.0083 absolute.

### Calibration on test

Bid = pCTR x value, so calibration error is spend error. Decile table:
```
            n  pred_ctr  actual_ctr  ratio
decile                                    
0       12773    0.0274      0.0312 0.8796
1       12972    0.0572      0.0631 0.9067
2       12494    0.0728      0.0813 0.8951
3       12747    0.0925      0.1085 0.8526
4       12841    0.1244      0.1410 0.8823
5       12694    0.1696      0.2049 0.8275
6       12689    0.1906      0.1767 1.0786
7       12791    0.2267      0.2392 0.9478
8       12680    0.2740      0.2743 0.9989
9       12725    0.4007      0.3955 1.0132
```
Overall predicted/actual = **0.9534**, ECE = 0.01262

### Metrics sliced by cold-start status

Aggregate numbers hide the cold path. `seen_*` flags come from the encoder: 1 if the entity appeared in history, 0 if it is brand new.
```
                            n  actual_ctr  pred_ctr    auc  pr_auc  log_loss  norm_entropy     rig  calib_ratio    ece
overall           127406.0000      0.1714    0.1634 0.7081  0.3366    0.4181        0.9126  0.0874       0.9534 0.0126
cold_character       415.0000      0.1855    0.1783 0.7337  0.3912    0.4265        0.8890  0.1110       0.9612 0.0479
warm_character    126991.0000      0.1714    0.1634 0.7080  0.3365    0.4180        0.9126  0.0874       0.9534 0.0126
cold_device_ip     75575.0000      0.1710    0.1597 0.7085  0.3410    0.4183        0.9145  0.0855       0.9339 0.0151
warm_device_ip     51831.0000      0.1721    0.1689 0.7074  0.3312    0.4177        0.9098  0.0902       0.9816 0.0127
cold_site            167.0000      0.0958    0.1792 0.5058  0.0960    0.3470        1.0990 -0.0990       1.8701 0.1098
unknown_device_id 101408.0000      0.1761    0.1704 0.7091  0.3415    0.4245        0.9120  0.0880       0.9677 0.0133
known_device_id    25998.0000      0.1530    0.1361 0.7004  0.3117    0.3929        0.9184  0.0816       0.8895 0.0204
```

## Feature importance (gain)

```
                            feature        gain  split  gain_share
55                te_x_site_id__C17 107488.3568    367      0.3197
47         te_x_site_id__banner_pos  50883.3382    400      0.1513
17                       te_site_id  29022.5889    323      0.0863
23                        te_app_id  28483.5058    559      0.0847
19                   te_site_domain  12493.4773    309      0.0372
53    te_x_character_id__banner_pos   8895.4954    566      0.0265
49                 te_x_app_id__C18   7662.8001    300      0.0228
29                  te_device_model   6115.6319    455      0.0182
45                  te_character_id   5829.9612    515      0.0173
35                           te_C14   5320.2509    373      0.0158
54      te_x_device_model__C15__C16   5122.8301    422      0.0152
58                       cnt_app_id   4662.5874    354      0.0139
7                       hour_of_day   4463.3702    342      0.0133
57                      cnt_site_id   4015.3676    402      0.0119
61                 cnt_device_model   3737.7007    455      0.0111
50           te_x_app_category__C21   3663.3476    267      0.0109
62                 cnt_character_id   3419.1053    446      0.0102
12                 log_interactions   2613.6727    360      0.0078
13                    char_age_days   2540.5570    385      0.0076
64                          cnt_C17   2403.5609    258      0.0071
37                           te_C17   2385.3810    250      0.0071
48     te_x_site_category__C15__C16   2262.6121    173      0.0067
41                           te_C20   2103.7543    253      0.0063
63                          cnt_C14   2023.8838    258      0.0060
56  te_x_hour_bucket__site_category   2016.1059    215      0.0060
15                      safety_tier   2014.2902    141      0.0060
27                  te_app_category   1975.0309    185      0.0059
52            te_x_safety_tier__C21   1921.9530    189      0.0057
43                           te_C21   1920.8026    162      0.0057
31                     te_device_ip   1880.0801    260      0.0056
```

- character-derived features carry 8.5% of total gain
- conversation-state features carry 0.8% of total gain

## Ablations: what is each feature family actually worth?

Each row retrains the full model with one family removed. The honest test of whether the character layer earns its serving cost.
  no character features: 55 features, 81 trees
  no conversation state: 61 features, 156 trees
  no cross features: 55 features, 92 trees
  no target encodings: 40 features, 143 trees
  no count features: 57 features, 157 trees
  no device features: 52 features, 89 trees
```
                         auc  norm_entropy    rig  log_loss  auc_delta  rig_delta_rel
full model            0.7104        0.9092 0.0908    0.4165     0.0000         0.0000
no character features 0.7033        0.9153 0.0847    0.4193    -0.0070        -0.0662
no conversation state 0.7113        0.9093 0.0907    0.4165     0.0010        -0.0002
no cross features     0.7078        0.9103 0.0897    0.4170    -0.0026        -0.0113
no target encodings   0.6411        0.9603 0.0397    0.4399    -0.0693        -0.5622
no count features     0.7104        0.9101 0.0899    0.4169    -0.0000        -0.0095
no device features    0.7053        0.9137 0.0863    0.4186    -0.0051        -0.0493
```

## Persisting artifacts
- wrote `/Users/miacordova/Documents/Claude/Projects/simula-takehome/artifacts/model.txt` (94 trees)
- wrote `/Users/miacordova/Documents/Claude/Projects/simula-takehome/artifacts/encoders.pkl`