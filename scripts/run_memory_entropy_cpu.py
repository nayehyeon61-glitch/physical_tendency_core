#!/usr/bin/env python3
"""Run a controlled scalar closure study on CPU; no ERA5 download is needed.

Quick: python scripts/run_memory_entropy_cpu.py --smoke --output runs/memory_entropy_smoke
Study: python scripts/run_memory_entropy_cpu.py --seeds 0 1 2 --include-kl --output runs/memory_entropy_cpu
"""

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from pgcfm.memory_entropy_toy import (ARMS, PhysicalClosure, ToyConfig, evaluate_model,
                                     split_data, train_model, train_reference_scale)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=ROOT / "runs/memory_entropy_cpu")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--arms", choices=list(ARMS), nargs="+", default=list(ARMS)[:-1])
    p.add_argument("--include-kl", action="store_true")
    p.add_argument("--iterations", type=int, default=300)
    p.add_argument("--width", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--sequence", type=int, default=32)
    p.add_argument("--warmup", type=int, default=8)
    p.add_argument("--history", type=int, default=16)
    p.add_argument("--horizon", type=int, default=40)
    p.add_argument("--members", type=int, default=64)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--data-seed", type=int, default=1729)
    p.add_argument("--kl-weight", type=float, default=.01)
    p.add_argument("--dt", type=float, default=.1)
    p.add_argument("--memory-strength", type=float, default=.8,
                   help="0 is the no-physical-memory generator control")
    p.add_argument("--memory-time", type=float, default=1.0,
                   help="hidden physical-memory relaxation time, dimensionless")
    p.add_argument("--noise-std", type=float, default=.6,
                   help="0 is the noise-free generator control (use --burn-in 0 for transients)")
    p.add_argument("--noise-time", type=float, default=.4,
                   help="hidden OU forcing correlation time, dimensionless")
    p.add_argument("--white-forcing", action="store_true",
                   help="independent discrete force per step (not a white-noise SDE limit); with "
                        "--memory-strength 0 the observations are truly Markov")
    p.add_argument("--burn-in", type=int, default=100)
    p.add_argument("--smoke", action="store_true", help="one seed, 10 iterations, small independent splits")
    p.add_argument("--execution-label", default="unspecified CPU",
                   help="honest provenance, e.g. local CPU smoke or Codex Cloud CPU")
    a = p.parse_args()
    if a.smoke:
        a.seeds, a.iterations, a.members = [a.seeds[0]], min(a.iterations, 10), 16
        a.horizon = min(a.horizon, 20)
    if a.include_kl and "memory_colored_kl" not in a.arms:
        a.arms.append("memory_colored_kl")
    if min(a.iterations, a.width, a.batch_size, a.threads, a.members, a.horizon) < 1:
        p.error("positive budgets are required")
    if len(set(a.seeds)) != len(a.seeds):
        p.error("seeds must be unique")
    return a


def save_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def aggregate(records):
    fields = ("mean_rollout_rmse", "mean_rollout_crps", "coverage90", "rms_spread",
              "one_step_mse", "one_step_nll", "variance_ratio", "mean_conditional_entropy", "rho")
    rows = []
    for arm in dict.fromkeys(r["arm"] for r in records):
        rr = [r for r in records if r["arm"] == arm]
        row = {"arm": arm, "n_seeds": len(rr), "parameters": rr[0]["parameters"]}
        for metric in fields:
            values = [r["metrics"].get(metric) for r in rr]
            if all(v is not None and np.isfinite(v) for v in values):
                row[metric + "_mean"] = float(np.mean(values))
                row[metric + "_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
            else:
                row[metric + "_mean"], row[metric + "_std"] = None, None
        rows.append(row)
    return rows


def main():
    a = parse_args()
    torch.set_num_threads(a.threads)
    torch.set_num_interop_threads(1)
    a.output.mkdir(parents=True, exist_ok=True)
    config = ToyConfig(dt=a.dt, memory_time=a.memory_time, memory_strength=a.memory_strength,
                       noise_time=a.noise_time, noise_std=a.noise_std, white_forcing=a.white_forcing,
                       burn_in=a.burn_in)
    length = max(96, a.history + a.horizon + 2, a.sequence + 2)
    data = split_data(config, n_train=32 if a.smoke else 128,
                      n_validation=8 if a.smoke else 32, n_test=16 if a.smoke else 64,
                      length=length, data_seed=a.data_seed)
    ref = train_reference_scale(data["train"], config.dt)
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    except subprocess.CalledProcessError:
        revision, dirty = None, None
    provenance = {"execution_label": a.execution_label, "hostname": platform.node(), "platform": platform.platform(),
                  "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
                  "device": "cpu", "threads": a.threads, "git_revision": revision, "git_dirty": dirty,
                  "config": asdict(config), "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
                  "split_sizes": {k: list(v.shape) for k, v in data.items()},
                  "split_generator_seeds": {"train": a.data_seed, "validation": a.data_seed + 1, "test": a.data_seed + 2},
                  "reference_scale_train_only": ref,
                  "scope": "scalar hidden-memory/OU synthetic mechanism study; not an ERA5 weather-skill experiment",
                  "fairness": "same width and optimizer-update budget; GRU arms have extra parameters, reported per arm",
                  "seed_uncertainty": "sample std across training seeds on shared independent trajectory splits, not a confidence interval",
                  "crps_estimator": "fair ensemble CRPS (unbiased pairwise spread term for iid ensemble members)",
                  "entropy_units": "differential entropy of scalar transition displacement in fixed dimensionless x units",
                  "forecast_inputs": "observed history only; hidden generator m/eta and future truth never supplied",
                  "potential": "known G = -dU/dx with U=0.4*x^2+0.025*x^4; no second potential added",
                  "entropy_regularizer": "optional same-mean Gaussian variance KL to train-only reference, beta preset; not raw entropy maximization"}
    provenance["source_sha256"] = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in ("pgcfm/memory_entropy_toy.py", "pgcfm/stochastic_memory.py",
                     "scripts/run_memory_entropy_cpu.py")
    }
    records, lead_rows, examples = [], [], {}
    start = time.perf_counter()
    for seed in a.seeds:
        for arm in a.arms:
            torch.manual_seed(seed)
            model = PhysicalClosure(arm, config.dt, ref, a.width)
            t0 = time.perf_counter()
            fit = train_model(model, data["train"], data["validation"], seed=seed,
                              iterations=a.iterations, batch_size=a.batch_size, sequence=a.sequence,
                              warmup=a.warmup, kl_weight=a.kl_weight)
            metrics, leads, example = evaluate_model(model, data["test"], history=a.history,
                                                     horizon=a.horizon, members=a.members, seed=100000 + seed)
            record = {"arm": arm, "seed": seed, "parameters": sum(p.numel() for p in model.parameters()),
                      "seconds": time.perf_counter() - t0, "fit": fit, "metrics": metrics}
            records.append(record)
            lead_rows.extend({"arm": arm, "seed": seed, **r} for r in leads)
            if seed == a.seeds[0] and example is not None:
                examples[arm] = example
            print(json.dumps({k: record[k] for k in ("arm", "seed", "parameters", "seconds", "metrics")}), flush=True)
            # Recoverable partial result after every finished fit; a failed run
            # remains identifiable, rather than silently looking complete.
            (a.output / "results.partial.json").write_text(json.dumps(
                {"complete": False, "provenance": provenance, "runs": records}, indent=2, allow_nan=False))
    rows = aggregate(records)
    result = {"complete": True, "provenance": provenance, "elapsed_seconds": time.perf_counter() - start,
              "runs": records, "aggregate": rows}
    (a.output / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    save_csv(a.output / "summary.csv", rows)
    save_csv(a.output / "leads.csv", lead_rows)
    lines = ["# Controlled memory / noise CPU experiment", "", provenance["scope"], "",
             "| arm | parameters | RMSE mean ± seed std | CRPS mean ± seed std | 90% coverage |",
             "| --- | ---: | ---: | ---: | ---: |"]
    def fmt(row, field):
        m, s = row[field + "_mean"], row[field + "_std"]
        return "n/a" if m is None else f"{m:.4f}" + ("" if s is None else f" ± {s:.4f}")
    for row in rows:
        lines.append(f"| {row['arm']} | {row['parameters']} | {fmt(row, 'mean_rollout_rmse')} | "
                     f"{fmt(row, 'mean_rollout_crps')} | {fmt(row, 'coverage90')} |")
    lines += ["", "Same width and optimizer updates; memory adds GRU parameters. Deterministic forecasts have zero ensemble spread.",
              "Seed std is training-seed variability on shared fixed independent trajectory splits, not a confidence interval.",
              "Entropy is diagnostic; the optional KL arm penalizes a variance ratio to a train-only reference.",
              f"Execution: {a.execution_label}; CPU threads={a.threads}; elapsed={result['elapsed_seconds']:.1f} s.", ""]
    (a.output / "summary.md").write_text("\n".join(lines))
    try:
        os.environ.setdefault("MPLCONFIGDIR", str(a.output / ".mpl-cache"))
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
        for arm in a.arms:
            rr = [r for r in lead_rows if r["arm"] == arm]
            if len(rr) != len(a.seeds) * a.horizon:
                continue  # divergent arms remain explicit in JSON; never invent a plotted curve
            for ax, metric in zip(axes, ("rmse", "crps", "coverage90")):
                values = np.array([[r[metric] for r in rr if r["seed"] == seed] for seed in a.seeds])
                xx = np.arange(1, a.horizon + 1) * config.dt
                ax.plot(xx, values.mean(0), label=arm)
                if len(a.seeds) > 1:
                    sd = values.std(0, ddof=1)
                    ax.fill_between(xx, values.mean(0) - sd, values.mean(0) + sd, alpha=.12)
                ax.set(xlabel="lead time (dimensionless)", ylabel="fair CRPS" if metric == "crps" else metric)
        axes[2].axhline(.9, color="black", ls=":", lw=1)
        axes[0].legend(fontsize=7)
        fig.suptitle("Synthetic hidden-memory + OU forcing; CPU neural closure")
        fig.tight_layout()
        fig.savefig(a.output / "comparison.png", dpi=160)
        plt.close(fig)
    except ImportError:
        print("matplotlib unavailable; JSON/CSV/Markdown were saved", flush=True)
    (a.output / "results.partial.json").unlink(missing_ok=True)
    print(f"Saved {a.output / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
