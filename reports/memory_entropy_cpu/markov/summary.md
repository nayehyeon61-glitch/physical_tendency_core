# Controlled memory / noise CPU experiment

scalar hidden-memory/OU synthetic mechanism study; not an ERA5 weather-skill experiment

| arm | parameters | RMSE mean ± seed std | CRPS mean ± seed std | 90% coverage |
| --- | ---: | ---: | ---: | ---: |
| markov | 337 | 0.1365 ± 0.0004 | 0.1105 ± 0.0004 | 0.0000 ± 0.0000 |
| memory | 1553 | 0.1374 ± 0.0005 | 0.1113 ± 0.0004 | 0.0000 ± 0.0000 |
| white | 354 | 0.1379 ± 0.0021 | 0.0778 ± 0.0013 | 0.8837 ± 0.0053 |
| memory_white | 1570 | 0.1372 ± 0.0012 | 0.0774 ± 0.0007 | 0.8863 ± 0.0031 |
| memory_colored | 1571 | 0.1371 ± 0.0012 | 0.0774 ± 0.0007 | 0.8841 ± 0.0055 |
| memory_colored_kl | 1571 | 0.1372 ± 0.0012 | 0.0774 ± 0.0007 | 0.8841 ± 0.0055 |

Same width and optimizer updates; memory adds GRU parameters. Deterministic forecasts have zero ensemble spread.
Seed std is training-seed variability on shared fixed independent trajectory splits, not a confidence interval.
Entropy is diagnostic; the optional KL arm penalizes a variance ratio to a train-only reference.
Execution: local CPU committed controls; cloud not used; CPU threads=2; elapsed=19.4 s.
