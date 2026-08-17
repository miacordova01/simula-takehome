# Signal audit

Train window: days 141021-141027. Test window: days 141028-141030.
train rows 729,685 (CTR 0.1850), test rows 270,315 (CTR 0.1682)

## T1/T2: out-of-time transfer AUC vs a shuffled control

`auc` = past-estimated per-value CTR scoring the future.
`auc_shuffled` = same after permuting the column (pure-noise floor, ~0.500).
`edge` = auc - auc_shuffled. Anything under ~0.005 is not usable signal.

```
               column  n_unique    auc  auc_shuffled    edge
0             site_id      2663 0.6594        0.4965  0.1629
1         site_domain      2905 0.6481        0.5020  0.1461
2                 C21        57 0.6153        0.5015  0.1138
3                 C14      2133 0.6078        0.5004  0.1074
4                 C18         4 0.6063        0.4997  0.1066
5                 C17       400 0.6030        0.4989  0.1041
6              app_id      3139 0.6030        0.4990  0.1040
7        device_model      5174 0.5938        0.5007  0.0932
8                 C19        65 0.5825        0.5004  0.0822
9        app_category        27 0.5719        0.5003  0.0716
10         app_domain       208 0.5724        0.5013  0.0711
11      site_category        21 0.5662        0.4998  0.0664
12                C20       162 0.5577        0.4973  0.0604
13       character_id      4997 0.5404        0.4979  0.0424
14          device_ip    547394 0.5337        0.4985  0.0352
15   device_conn_type         4 0.5351        0.5001  0.0350
16                C16         9 0.5341        0.5003  0.0338
17                C15         8 0.5330        0.5001  0.0328
18   num_interactions      1355 0.5322        0.5015  0.0307
19         banner_pos         7 0.5231        0.5002  0.0228
20          device_id    152548 0.5195        0.5000  0.0195
21        hour_of_day        24 0.5141        0.4999  0.0143
22        safety_tier         3 0.5134        0.5015  0.0119
23        device_type         4 0.5120        0.5003  0.0117
24                 C1         7 0.5125        0.5011  0.0114
25      is_first_turn         2 0.5013        0.5011  0.0002
26      char_age_days       303 0.5015        0.5016 -0.0001
27  session_msg_count       139 0.4988        0.4991 -0.0002
28   session_progress      2500 0.5000        0.5004 -0.0004
29       creator_type         2 0.4998        0.5007 -0.0010
30  conversation_turn       113 0.5003        0.5024 -0.0021
31        day_of_week         7 0.4938        0.4995 -0.0056
```

**Carries signal (edge >= 0.010):** site_id, site_domain, C21, C14, C18, C17, app_id, device_model, C19, app_category, app_domain, site_category, C20, character_id, device_ip, device_conn_type, C16, C15, num_interactions, banner_pos, device_id, hour_of_day, safety_tier, device_type, C1

**No usable standalone signal (edge < 0.005):** is_first_turn, char_age_days, session_msg_count, session_progress, creator_type, conversation_turn, day_of_week

## Is `character_id` more than binomial noise?

- characters with >=100 train impressions: 1,879
- observed variance of per-character CTR: 0.001223
- variance expected from binomial noise alone: 0.000634
- excess (true between-character variance): 0.000589
- implied true between-character CTR sd: 0.0243 (vs base CTR 0.1850)
- share of observed spread that is real (intra-class correlation): **48.1%**

## Is the `safety_tier` lift real or confounded?

Raw CTR by tier:
```
               ctr
safety_tier       
mature      0.2019
sfw         0.1761
suggestive  0.1777
```
CTR standardised over `C15` strata:
```
             ctr_adj
safety_tier         
mature        0.2018
sfw           0.1762
suggestive    0.1775
```
CTR standardised over `site_id` strata:
```
             ctr_adj
safety_tier         
mature        0.2032
sfw           0.1771
suggestive    0.1788
```
CTR standardised over `app_id` strata:
```
             ctr_adj
safety_tier         
mature        0.2042
sfw           0.1780
suggestive    0.1795
```
CTR standardised over `C18` strata:
```
             ctr_adj
safety_tier         
mature        0.2020
sfw           0.1762
suggestive    0.1776
```

If characters were assigned to traffic at random, tier should be independent of the publisher/creative columns. Cramer's V:
```
               cramers_v
app_category      0.0103
site_category     0.0094
C15               0.0071
banner_pos        0.0043
C1                0.0040
C18               0.0035
```