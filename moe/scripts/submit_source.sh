#!/usr/bin/env bash
# Explicit resource allocation; no cluster-specific partition/QoS defaults.
set -euo pipefail
if [[ $# -lt 5 ]]; then
  echo "Usage: bash moe/scripts/submit_source.sh PARTITION TIME MEMORY CONFIG OUTPUT_ROOT [SEED ...]" >&2
  echo "Example: bash moe/scripts/submit_source.sh allowed_partition 04:00:00 32G configs/local.json runs/study 0" >&2
  exit 2
fi
partition=$1
wall_time=$2
memory=$3
config=$4
output=$5
shift 5
if [[ $# -eq 0 ]]; then set -- 0; fi
mkdir -p logs
sbatch --partition="$partition" --time="$wall_time" --mem="$memory" \
  --gres=gpu:1 --cpus-per-task=4 --job-name=dapo-moe \
  --output='logs/dapo-moe-%j.out' --error='logs/dapo-moe-%j.err' \
  --chdir="$PWD" --export=ALL \
  moe/scripts/run_experiments.sh "$config" "$output" "$@"
