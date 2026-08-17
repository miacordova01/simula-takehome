# Serving latency

Model + encoder load: 1.71s (one-time, at process start)
- trees: 94, features: 65
- encoder tables held in memory: 25 target-encoding maps, 770,475 total keys

## Scoring path, by slate size

Each row is 400 requests against real held-out contexts. Times are per request (the whole slate), single-threaded process, warm cache.
```
            p50_ms  p95_ms  p99_ms  max_ms  feature_us_mean  predict_us_mean
slate_size                                                                  
1            0.193   0.270   0.319   0.343           57.969          140.491
5            0.248   0.319   0.384   0.895           97.375          155.713
10           0.332   0.431   0.492   0.601          155.208          184.720
20           0.481   0.589   0.698   0.775          265.788          222.982
50           0.925   1.079   1.201   1.385          597.664          338.184
100          1.652   1.882   1.992   2.070         1134.768          522.995
```

## Full ranker (scoring + safety gate + UCB + pacing + fatigue + sort)

```
            p50_ms  p95_ms  p99_ms  max_ms
slate_size                                
10           0.420   0.519   0.599   0.764
20           0.633   0.721   0.801   0.843
50           1.273   1.442   1.539   1.663
100          2.325   2.564   2.645   2.750
```

**p99 at a 50-candidate slate: 1.54ms** against a 50ms budget, leaving 48ms for network, retrieval, auction and logging.

## Where the time goes

```
                                   microseconds  share
stage                                                 
context vector (once per request)        19.445  0.015
per-candidate feature fill              578.219  0.454
LightGBM batched predict                338.184  0.266
ranker business logic                   337.152  0.265
```

## Value of the context/ad split

The context vector is computed once and broadcast across the slate. If it were recomputed per candidate instead:
```
                                     us_per_request  ms_per_request
approach                                                           
context computed once (implemented)         935.848           0.936
context recomputed per candidate           1888.633           1.889
```

The split is worth **2.0x** at a 50-candidate slate, and the gap widens linearly with slate size.

## Throughput

- single core, 50-candidate slates: **~786 requests/sec**
- a 16-core box at 60% utilisation: ~7,541 req/s
- LightGBM releases the GIL during predict, so a thread pool scales close to linearly until memory bandwidth binds.