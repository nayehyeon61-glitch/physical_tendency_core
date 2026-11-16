# Controlled memory / noise CPU experiment

scalar hidden-memory/OU synthetic mechanism study; not an ERA5 weather-skill experiment

| arm | parameters | RMSE mean ± seed std | CRPS mean ± seed std | 90% coverage |
| --- | ---: | ---: | ---: | ---: |
| markov | 337 | 0.3662 ± 0.0052 | 0.2972 ± 0.0046 | 0.0000 ± 0.0000 |
| memory | 1553 | 0.2998 ± 0.0047 | 0.2445 ± 0.0046 | 0.0000 ± 0.0000 |
| white | 354 | 0.3631 ± 0.0061 | 0.2216 ± 0.0039 | 0.6169 ± 0.0106 |
| memory_white | 1570 | 0.3069 ± 0.0080 | 0.1735 ± 0.0047 | 0.8922 ± 0.0136 |
| memory_colored | 1571 | 0.2961 ± 0.0014 | 0.1672 ± 0.0008 | 0.8775 ± 0.0081 |
| memory_colored_kl | 1571 | 0.2961 ± 0.0014 | 0.1672 ± 0.0008 | 0.8785 ± 0.0088 |

Same width and optimizer updates; memory adds GRU parameters. Deterministic forecasts have zero ensemble spread.
Seed std is training-seed variability on shared fixed independent trajectory splits, not a confidence interval.
Entropy is diagnostic; the optional KL arm penalizes a variance ratio to a train-only reference.
Execution: local CPU committed controls; cloud not used; CPU threads=2; elapsed=17.9 s.
