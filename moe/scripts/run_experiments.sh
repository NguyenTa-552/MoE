#!/usr/bin/env bash
# Source-only matrix. Run within an existing compute allocation.
set -euo pipefail
if [[ $# -lt 2 ]]; then
  echo "Usage: bash moe/scripts/run_experiments.sh CONFIG OUTPUT_ROOT [SEED ...]" >&2
  exit 2
fi
config=$1
output_root=$2
shift 2
if [[ $# -eq 0 ]]; then set -- 0; fi
python_bin=${MOE_PYTHON:-python}
device=${MOE_DEVICE:-cuda}
for seed in "$@"; do
  if [[ -e "$output_root/seed-$seed" ]]; then
    echo "Refusing to reuse existing study directory: $output_root/seed-$seed" >&2
    exit 2
  fi
done
"$python_bin" -m unittest discover -s moe/tests -v
studies=()
failed=0
for seed in "$@"; do
  study="$output_root/seed-$seed"
  studies+=("$study")
  if ! "$python_bin" -m moe.cli run --config "$config" --output "$study" --seed "$seed" --device "$device"; then
    echo "Seed $seed failed; preserving its status and continuing the requested seeds." >&2
    failed=1
  fi
done
existing=()
for study in "${studies[@]}"; do
  if [[ -f "$study/status.json" ]]; then existing+=("$study"); fi
done
if [[ ${#existing[@]} -gt 0 ]]; then
  "$python_bin" -m moe.cli report --studies "${existing[@]}" --output "$output_root/report"
fi
exit "$failed"
