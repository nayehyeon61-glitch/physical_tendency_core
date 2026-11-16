# Controlled memory / noise CPU experiment

scalar hidden-memory/OU synthetic mechanism study; not an ERA5 weather-skill experiment

| arm | parameters | RMSE mean ± seed std | CRPS mean ± seed std | 90% coverage |
| --- | ---: | ---: | ---: | ---: |
| markov | 337 | 0.0213 ± 0.0010 | 0.0175 ± 0.0010 | 0.0000 ± 0.0000 |
| memory | 1553 | 0.0065 ± 0.0009 | 0.0056 ± 0.0007 | 0.0000 ± 0.0000 |
| white | 354 | 0.0239 ± 0.0014 | 0.0168 ± 0.0009 | 0.5069 ± 0.0301 |
| memory_white | 1570 | 0.0080 ± 0.0021 | 0.0057 ± 0.0014 | 0.3267 ± 0.0484 |
| memory_colored | 1571 | 0.0087 ± 0.0009 | 0.0057 ± 0.0015 | 0.3792 ± 0.2204 |
| memory_colored_kl | 1571 | 0.0053 ± 0.0019 | 0.0032 ± 0.0013 | 0.6546 ± 0.1553 |

Same width and optimizer updates; memory adds GRU parameters. Deterministic forecasts have zero ensemble spread.
Seed std is training-seed variability on shared fixed independent trajectory splits, not a confidence interval.
Entropy is diagnostic; the optional KL arm penalizes a variance ratio to a train-only reference.
Execution: local CPU committed controls; cloud not used; CPU threads=2; elapsed=20.7 s.
