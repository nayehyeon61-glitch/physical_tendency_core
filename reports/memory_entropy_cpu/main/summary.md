# Controlled memory / noise CPU experiment

scalar hidden-memory/OU synthetic mechanism study; not an ERA5 weather-skill experiment

| arm | parameters | RMSE mean ± seed std | CRPS mean ± seed std | 90% coverage |
| --- | ---: | ---: | ---: | ---: |
| markov | 337 | 0.3286 ± 0.0005 | 0.2653 ± 0.0003 | 0.0000 ± 0.0000 |
| memory | 1553 | 0.2571 ± 0.0021 | 0.2081 ± 0.0023 | 0.0000 ± 0.0000 |
| white | 354 | 0.3261 ± 0.0030 | 0.1956 ± 0.0018 | 0.6445 ± 0.0081 |
| memory_white | 1570 | 0.2584 ± 0.0040 | 0.1460 ± 0.0023 | 0.8728 ± 0.0169 |
| memory_colored | 1571 | 0.2589 ± 0.0044 | 0.1460 ± 0.0027 | 0.8908 ± 0.0045 |
| memory_colored_kl | 1571 | 0.2589 ± 0.0044 | 0.1460 ± 0.0027 | 0.8919 ± 0.0049 |

Same width and optimizer updates; memory adds GRU parameters. Deterministic forecasts have zero ensemble spread.
Seed std is training-seed variability on shared fixed independent trajectory splits, not a confidence interval.
Entropy is diagnostic; the optional KL arm penalizes a variance ratio to a train-only reference.
Execution: local CPU committed pilot; cloud not used; CPU threads=2; elapsed=17.1 s.
