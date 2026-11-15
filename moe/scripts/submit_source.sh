#!/usr/bin/env bash
# Compatibility entry point; the cluster launcher initializes the job environment.
set -euo pipefail
preview=()
if [[ ${1:-} == --dry-run ]]; then preview=(--dry-run); shift; fi
if [[ $# -lt 5 ]]; then
  echo "Usage: bash moe/scripts/submit_source.sh [--dry-run] PARTITION TIME MEMORY CONFIG OUTPUT_ROOT [SEED ...]" >&2
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
export CLUSTER_PARTITION="$partition" CLUSTER_TIME="$wall_time" CLUSTER_MEM="$memory"
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec bash "$script_dir/run_cluster.sh" "${preview[@]}" source "$config" "$output" "$@"
