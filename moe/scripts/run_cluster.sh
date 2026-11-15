#!/usr/bin/env bash
# Submit on the login node; initialize modules and execute on the compute node.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash moe/scripts/run_cluster.sh [--dry-run] MODE [arguments...]

Modes:
  test                         Numerical tests inside an allocation
  smoke [--output DIR]         Synthetic CPU workflow inside an allocation
  preflight --config FILE ...  Path/split checks and CUDA diagnostics; no training
  source CONFIG [OUTPUT_ROOT [SEED ...]]
                               Staged source study; default seed 0
  evaluate --study DIR --variant ID --dataset NAME --data-root DIR ...
                               Locked target evaluation; supply --baseline-result for non-B0

Output defaults to runs/dapo-moe-MODE-JOB_ID. Source OUTPUT_ROOT may be supplied
explicitly. All relative paths resolve from this checkout's root.

Environment overrides (use a partition your account can access):
  CLUSTER_PARTITION    student9696 (default); student4848/student4896 supported
  CLUSTER_GPU_GRES     Exact GRES; default is one 96 GB MIG on student9696
  CLUSTER_QOS          qos96 on student9696 by default; empty omits --qos
  CLUSTER_CONDA_ENV    Existing environment name/path (default: DAPOenv)
  CLUSTER_CUDA_MODULE  CUDA module (default: cuda12.8, as in the archived script)
  CLUSTER_CPUS         CPUs per task (default: 4)
  CLUSTER_MEM          Host memory (default: 16G)
  CLUSTER_TIME         Wall time (default: 01:00:00)
  MOE_PYTHON           Interpreter within the activated environment (default: python)
  MOE_DEVICE           Experiment device (default: cuda)
  MOE_VARIANTS         Optional space-separated retained variants, e.g. "B3 P1"

--dry-run prints the sbatch command without submitting or running Python.
EOF
}

batch=false
dry_run=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --batch) batch=true; shift ;;
    --dry-run) dry_run=true; shift ;;
    --help|-h) usage; exit 0 ;;
    *) break ;;
  esac
done
mode=${1:-}
case "$mode" in
  test|smoke|preflight|source|evaluate) shift ;;
  *) usage >&2; exit 2 ;;
esac
if [[ $mode == source && $# -lt 1 ]]; then
  echo 'source requires a configuration file.' >&2
  exit 2
fi
if "$batch" && "$dry_run"; then
  echo '--batch and --dry-run cannot be combined.' >&2
  exit 2
fi

if ! "$batch"; then
  project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
  export MOE_CLUSTER_PROJECT_ROOT="$project_root"
  partition=${CLUSTER_PARTITION:-student9696}
  default_qos=
  case "$partition" in
    student4848|student4896)
      default_gres=gpu:nvidia_rtx_pro_6000_blackwell_server_edition_2g.48gb:1 ;;
    student9696)
      default_gres=gpu:nvidia_rtx_pro_6000_blackwell_server_edition_4g.96gb:1
      default_qos=qos96 ;;
    *)
      default_gres=
      if [[ -z ${CLUSTER_GPU_GRES:-} ]]; then
        echo 'Set CLUSTER_GPU_GRES explicitly for this partition.' >&2
        exit 2
      fi ;;
  esac
  command_args=(sbatch --export=ALL --chdir="$project_root"
    --job-name="dapo-moe-$mode" --partition="$partition"
    --nodes=1 --ntasks=1 --cpus-per-task="${CLUSTER_CPUS:-4}"
    --mem="${CLUSTER_MEM:-16G}" --time="${CLUSTER_TIME:-01:00:00}"
    --gres="${CLUSTER_GPU_GRES:-$default_gres}"
    --output="$project_root/logs/%x-%j.out"
    --error="$project_root/logs/%x-%j.err")
  qos=${CLUSTER_QOS-$default_qos}
  if [[ -n $qos ]]; then command_args+=(--qos="$qos"); fi
  command_args+=("$project_root/moe/scripts/run_cluster.sh" --batch "$mode" "$@")
  if "$dry_run"; then
    printf '%q ' "${command_args[@]}"
    printf '\n'
    exit 0
  fi
  command -v sbatch >/dev/null || { echo 'Submit from the cluster login node: sbatch is unavailable.' >&2; exit 1; }
  # SLURM opens logs before the batch script starts.
  mkdir -p -- "$project_root/logs"
  exec "${command_args[@]}"
fi

: "${SLURM_JOB_ID:?Batch execution requires a SLURM allocation}"
: "${MOE_CLUSTER_PROJECT_ROOT:?Submit using run_cluster.sh to set the checkout root}"
cd -- "$MOE_CLUSTER_PROJECT_ROOT"
if ! type module >/dev/null 2>&1 && [[ -r /etc/profile.d/lmod.sh ]]; then
  set +u
  source /etc/profile.d/lmod.sh
  set -u
fi
type module >/dev/null 2>&1 || { echo 'Lmod module command is unavailable in this batch shell.' >&2; exit 1; }
# The guide requires source activate in noninteractive jobs. Activation scripts
# and Lmod can reference unset variables; keep errexit but relax nounset here.
set +u
module load miniconda3
module load "${CLUSTER_CUDA_MODULE:-cuda12.8}"
source activate "${CLUSTER_CONDA_ENV:-DAPOenv}"
set -u

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
python_bin=${MOE_PYTHON:-python}
export MOE_PYTHON="$python_bin"
export MOE_DEVICE="${MOE_DEVICE:-cuda}"
# Never replace CUDA_VISIBLE_DEVICES; SLURM assigns the permitted MIG instance.
printf 'Job %s; node %s; partition %s; environment %s\n' \
  "$SLURM_JOB_ID" "${SLURM_NODELIST:-unknown}" "${SLURM_JOB_PARTITION:-unknown}" "${CLUSTER_CONDA_ENV:-DAPOenv}"
printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:-unset}"
"$python_bin" -c 'import sys, torch; print("Python:", sys.version); print("PyTorch:", torch.__version__, "CUDA build:", torch.version.cuda); print("Visible CUDA devices:", torch.cuda.device_count()); print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unavailable")'
if [[ $MOE_DEVICE == cuda* ]]; then
  "$python_bin" -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable: verify the allocated MIG and your reproduced PyTorch environment"'
fi

output="runs/dapo-moe-$mode-$SLURM_JOB_ID"
case "$mode" in
  source)
    config=$1
    shift
    source_output=${1:-$output}
    if [[ $# -gt 0 ]]; then shift; fi
    command_args=(bash moe/scripts/run_experiments.sh "$config" "$source_output" "$@") ;;
  test) command_args=("$python_bin" -m unittest discover -s moe/tests -v "$@") ;;
  smoke) command_args=("$python_bin" -m moe.cli smoke --output "$output" "$@") ;;
  preflight) command_args=("$python_bin" -m moe.cli run --dry-run --output "$output" "$@") ;;
  evaluate) command_args=("$python_bin" -m moe.cli evaluate-target --output "$output" --device "$MOE_DEVICE" "$@") ;;
esac
printf 'Running: '
printf '%q ' "${command_args[@]}"
printf '\n'
# sbatch has already allocated this single-task compute-node job. Direct
# execution matches the guide and does not require a nested srun command.
exec "${command_args[@]}"
