#!/usr/bin/env bash
# Run on an already allocated CPU host. This script does not provision cloud resources.
set -euo pipefail
cd "$(dirname "$0")/.."

mode=${1:-study}
python_bin=${PYTHON_BIN:-python3}
output=${OUTDIR:-runs/memory_entropy_cpu}
label=${EXECUTION_LABEL:-CPU host; cloud status unverified}
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MPLCONFIGDIR="$output/.mpl-cache"
mkdir -p "$MPLCONFIGDIR"

"$python_bin" -m pytest -q tests/test_stochastic_memory.py tests/test_memory_entropy_toy.py
common=(--seeds 0 1 2 --iterations 300 --width 16 --threads 2 --members 64 --include-kl)
case "$mode" in
  smoke)
    "$python_bin" scripts/run_memory_entropy_cpu.py "${common[@]}" --smoke \
      --output "$output" --execution-label "$label"
    ;;
  study)
    "$python_bin" scripts/run_memory_entropy_cpu.py "${common[@]}" \
      --output "$output" --execution-label "$label"
    ;;
  controls)
    "$python_bin" scripts/run_memory_entropy_cpu.py "${common[@]}" --memory-strength 0 \
      --output "$output/no_physical_memory" --execution-label "$label"
    "$python_bin" scripts/run_memory_entropy_cpu.py "${common[@]}" --noise-std 0 --burn-in 0 \
      --output "$output/no_noise" --execution-label "$label"
    "$python_bin" scripts/run_memory_entropy_cpu.py "${common[@]}" --memory-strength 0 --white-forcing \
      --output "$output/markov" --execution-label "$label"
    ;;
  *)
    printf 'Usage: %s {smoke|study|controls}\n' "$0" >&2
    exit 2
    ;;
esac
