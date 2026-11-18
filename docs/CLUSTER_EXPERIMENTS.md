# Running MoE experiments on the cluster

Start with [the fresh-server guide](SERVER_FROM_SCRATCH.md) for cloning, source
split creation, concrete configuration fields, and the ordered validation sequence.

These instructions adapt the supplied **Cluster User Guide (2026-09-23)** and
`script.sh` in `docs/reference/anomaly_detection_clip_reference.zip`. The guide PDF
remains a local ignored file. No server jobs were submitted during development.

## Prepare on the login node

Connect through your assigned NetBird VPN and SSH account. Use the login node for
environment setup, data organization, and submission; run training, inference,
numerical tests, and smoke workflows inside SLURM allocations (guide pages 2–5).

Transfer this updated checkout, including `moe/scripts/run_cluster.sh`. Initialize
the pinned `DAPO/` submodule using `git submodule update --init --recursive`.
The verified upstream commit is `290411d7374cc2182ad3491b826131af1ff73224`.
Reuse the environment, backbone weights,
and baseline checkpoint from your successful reproduction.

```bash
cd /path/to/MoE
module load miniconda3
module load cuda12.8
source activate DAPOenv
export CLUSTER_CONDA_ENV=DAPOenv
```

The job script repeats module loading and activation on the compute node; activating
only on the login node is insufficient. The guide recommends Conda, then conda-forge,
then pip as a fallback. Do not replace a working reproduction environment with the
local CPU test environment. Ensure its PyTorch build supports the allocated Blackwell
GPU; loading a CUDA module alone does not change the PyTorch wheel's CUDA support.

`cuda12.8` follows the archived working script and the guide's named module. Override
`CLUSTER_CUDA_MODULE` with an available compatible module if required. The guide
warns that CUDA 12.6 does not work on these GPUs even though an older example later
uses it; the launcher follows the explicit warning. It preserves SLURM's MIG visibility
instead of overwriting `CUDA_VISIBLE_DEVICES`.

## Choose permitted resources

The configured default is `student9696`, QoS `qos9696`, one 96 GB MIG, four CPUs,
16 GB host RAM, and one hour, using Conda environment `DAPOenv` as requested.
Your account must have access to the selected partition and QoS.

| Partition | Requested GRES |
| --- | --- |
| `student4848`, `student4896` | `gpu:nvidia_rtx_pro_6000_blackwell_server_edition_2g.48gb:1` |
| `student9696` | `gpu:nvidia_rtx_pro_6000_blackwell_server_edition_4g.96gb:1` |

The guide lists a 24-hour limit for student partitions. For other partitions,
including `researcher`, set `CLUSTER_GPU_GRES` explicitly. The launcher does not
use `qos9696` on other partitions. Override `CLUSTER_QOS` with a permitted value, or
set it to an empty string to omit the QoS option.
Check actual permissions and availability with `sinfo` and `scontrol show partition`.

```bash
export CLUSTER_PARTITION=student9696
export CLUSTER_QOS=qos9696
export CLUSTER_TIME=01:00:00
export CLUSTER_MEM=16G
# Optional account-specific overrides:
# export CLUSTER_QOS=YOUR_ALLOWED_QOS
# export CLUSTER_CUDA_MODULE=YOUR_AVAILABLE_MODULE

bash moe/scripts/run_cluster.sh --dry-run test
bash moe/scripts/run_cluster.sh test
bash moe/scripts/run_cluster.sh smoke
```

`--dry-run` prints the submission command, performs no Python workload, and does not
contact SLURM. Actual submission creates `logs/` before calling `sbatch`. Each job
logs its node, environment, Python/PyTorch/CUDA versions, and visible GPU. Synthetic
smoke writes `runs/dapo-moe-smoke-JOB_ID`; it is software validation, not an experiment.

## Prepare and check a real experiment

Follow `moe/README.md` to audit spatial mask semantics and create immutable
expert-training, router-training, and source-validation manifests directly from
`meta.json`. The default stratifies images by existing product/species labels and
records an image-independence assumption; a verified group map is optional for
known repeated objects/acquisitions. Copy `moe/configs/experiment.example.json` to
`moe/configs/local.json` and supply server paths and the exact reproduced settings.
The archived script's dataset path was
`$HOME/triet-work/datasets/cde-mtad/canonical/mvtec`; verify your current location
instead of assuming that historical path is still correct. Config paths expand
`$HOME`, `${HOME}`, and leading `~`; relative paths resolve from the checkout root.

Keep read-only datasets under `/home` if convenient. The guide says `/compute_home`
is faster for writes but shared and limited to 2 TB. For dense evaluation, create a
job-appropriate scratch directory there and set its absolute path as `scratch_dir`
in the config. Keep persistent reports/checkpoints under your own storage and clean
up scratch after jobs. The launcher never copies or deletes datasets automatically.

```bash
bash moe/scripts/run_cluster.sh preflight --config moe/configs/local.json --seed 0
```

Preflight checks source image/mask existence, label mappings, configuration settings,
split hashes, and the allocated CUDA environment. It does not
load the full model, verify checkpoint parity, or run training. Those checks occur
at the beginning of the real source study. Prepare cached backbone weights if
compute nodes cannot access the internet.

## Submit source studies

For a short real-data feasibility job, copy the config to another file and set
`expert_epochs` and `router_epochs` to 1. Keep the same source manifests, avoid
overwriting full-run configs, and use a new output directory. Synthetic smoke is
separate from this real-data feasibility check.

```bash
MOE_VARIANTS="B3 P1" bash moe/scripts/run_cluster.sh source runs/configs/feasibility.json

# After feasibility, choose a time within your actual allocation/partition limit.
CLUSTER_TIME=08:00:00 CLUSTER_MEM=32G \
  bash moe/scripts/run_cluster.sh source moe/configs/local.json runs/moe-seed0 0

# Retained comparisons use matched manifests and a fresh output root.
CLUSTER_TIME=12:00:00 CLUSTER_MEM=32G \
  bash moe/scripts/run_cluster.sh source moe/configs/local.json runs/moe-matched 0 1 2
```

The time examples are requests, not runtime estimates. Multiple seeds in one job run
sequentially; request only your available budget. With no explicit output root,
source results go to `runs/dapo-moe-source-JOB_ID/seed-0`. Numerical tests run before
source training on the compute node. Source selection, complementarity, P2 gates,
and failed-run reporting are unchanged. Target tests are never started automatically.
`MOE_VARIANTS` restricts optional comparisons for feasibility; B0/B2/O1 and the
shared expert bank are always prerequisites. Omit it for the full comparison set.

The previous explicit-resource entry point remains supported and now initializes
the same compute-node environment:

```bash
bash moe/scripts/submit_source.sh --dry-run student9696 01:00:00 16G \
  moe/configs/local.json runs/moe-explicit 0
```

## Submit locked target evaluation

Review and lock the source study first. Evaluate B0 once per target split and seed,
then provide that paired result for each retained nonbaseline variant:

```bash
bash moe/scripts/run_cluster.sh evaluate \
  --study runs/moe-seed0/seed-0 --variant B0 \
  --dataset mpdd --data-root /datasets/mpdd --output runs/target/seed-0/B0

# Submit after B0 completes successfully and result.json exists.
bash moe/scripts/run_cluster.sh evaluate \
  --study runs/moe-seed0/seed-0 --variant P1 \
  --dataset mpdd --data-root /datasets/mpdd \
  --baseline-result runs/target/seed-0/B0/result.json --output runs/target/seed-0/P1
```

Replace `/datasets/mpdd` with the actual root. Add `--manifest` for a same-dataset
held-out target split. Source-selected thresholds and checkpoints remain locked.
Evaluation verifies `source-locks.json`, the resolved source configuration, split
summary, and checkpoint hashes. Exact source-image duplicates are rejected even
under different target paths. Preserve the source artifacts after selection.

## Monitor and collect artifacts

```bash
squeue -u "$USER"
tail -f logs/dapo-moe-source-JOB_ID.out
sacct -j JOB_ID --format=JobID,JobName,Partition,Elapsed,State,ExitCode,MaxRSS
scancel JOB_ID
```

Logs use `logs/dapo-moe-MODE-JOB_ID.out` and `.err`. Results live in the specified
output root, with config, provenance, model/data hashes, split counts, status,
checkpoints, per-class metrics, routing diagnostics, prediction exports, and reports.
An interrupted job can leave `status=running`; consult SLURM accounting before
interpreting completion. Existing study directories are never resumed or overwritten
by this launcher; retain interrupted artifacts and use a fresh output root.
