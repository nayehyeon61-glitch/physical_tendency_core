# Controlled memory / noise CPU experiment

scalar hidden-memory/OU synthetic mechanism study; not an ERA5 weather-skill experiment

| arm | parameters | RMSE mean ± seed std | CRPS mean ± seed std | 90% coverage |
| --- | ---: | ---: | ---: | ---: |
| memory_entropy_off | 1586 | 0.2584 ± 0.0040 | 0.1460 ± 0.0023 | 0.8728 ± 0.0169 |
| memory_entropy_on | 1586 | 0.2593 ± 0.0070 | 0.1466 ± 0.0042 | 0.8725 ± 0.0180 |
| memory_entropy_mean | 1586 | 0.2565 ± 0.0031 | 0.2077 ± 0.0033 | 0.0000 ± 0.0000 |

Same width and optimizer updates; memory adds GRU parameters. Deterministic forecasts have zero ensemble spread.
Seed std is training-seed variability on shared fixed independent trajectory splits, not a confidence interval.
Centered negative Gaussian entropy is a direct causal memory input in the matched off/on arms; mean uses the same learned law with sampling disabled.
Execution: local CPU direct entropy input; cloud not used; CPU threads=2; elapsed=52.1 s.

Direct entropy-input off/on/mean arms have identical parameter counts and initialized weights.
Their centered Gaussian negative entropy is exactly -log(sigma/reference_sigma), fed causally into memory.
The mean arm trains the on law and disables sampling at forecast; its mean path is not an exact nonlinear ensemble mean.
