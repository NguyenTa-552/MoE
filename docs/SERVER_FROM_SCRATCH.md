# Run on the server from a fresh clone

Use the existing reproduced DAPO checkpoint, prepared datasets, and `DAPOenv`.
The defaults are `student9696`, QoS `qos96`, one 96 GB MIG, and `cuda12.8`.
No environment exports are required for those defaults.

## 1. Make this version available

The audit fixes and launch instructions are local changes until published or
transferred. A clone retrieves only committed, pushed files. From the development
checkout, review `git status` and `git diff`, commit the intended implementation and
documentation files, and push `moe-experiments`. Keep datasets, checkpoints, local
configs, environments, and the private cluster PDF out of the commit. Alternatively,
transfer the updated working tree to the server.

## 2. Clone on the login node

Connect using your assigned VPN/SSH details, then:

```bash
mkdir -p "$HOME/projects"
cd "$HOME/projects"
git clone --branch moe-experiments https://github.com/NguyenTa-552/MoE.git
cd MoE
git clone https://github.com/boschresearch/defect-aware-prompt-optimization.git DAPO
git -C DAPO checkout 290411d7374cc2182ad3491b826131af1ff73224
git rev-parse HEAD
git status --short
```

`DAPO/` is ignored and must be provided separately. Keep it unchanged. If a checkout
already exists, inspect its status before updating rather than cloning over it.

## 3. Activate the reproduced environment

```bash
module load miniconda3
module load cuda12.8
source activate DAPOenv
```

Use this environment for metadata/configuration preparation. The launcher repeats
activation inside every compute job. Reuse the working GPU PyTorch installation;
the local `.venv` is a CPU test environment. Verify dependencies against your
reproduction, including NumPy, SciPy, scikit-learn, Pillow, torch, torchvision, and
the upstream OpenCLIP dependencies. Numerical tests and training run in allocations.

You need the learned DAPO checkpoint containing visual prompts, text prompts, and
adapters, as well as the CLIP backbone weights. These are separate artifacts.
For offline compute nodes, cache the backbone or set `model.pretrained` to the
compatible local backbone file used in your reproduction.

## 4. Create MVTec source splits and configure paths

Replace `/actual/datasets/mvtec` below with the root containing `meta.json` and
the image/mask paths referenced by it:

```bash
python -m moe.cli split --metadata /actual/datasets/mvtec/meta.json \
  --output runs/manifests/mvtec-source --seed 0
cp moe/configs/experiment.example.json moe/configs/local.json
```

The split command uses DAPO's metadata `test` records as supervised **source** data
and stratifies the existing product/species labels into expert training, router
training, and source validation. This is not a target test. Inspect
`runs/manifests/mvtec-source/summary.json`, especially missing role coverage. It
assumes independent images; known related views need an optional `--groups` map.
An illustrated metadata example is in [moe/examples/mvtec](../moe/examples/mvtec/README.md).

Edit `moe/configs/local.json`:

```json
{
  "dapo_root": "DAPO",
  "baseline_checkpoint": "/actual/checkpoints/reproduced-dapo.pth",
  "source_root": "/actual/datasets/mvtec",
  "splits": "runs/manifests/mvtec-source"
}
```

These are fields to update in the complete copied config, not a replacement file.
Check model settings, defect names, and `label_ids` against the reproduced run.
After confirming each anomalous image's positive binary-mask pixels all belong
to its metadata species, set `single_type_mask_verified` to JSON `true`.
An image class alone does not establish that mask property. Standard MVTec binary
masks require this audit; genuinely multi-type masks use `typed_mask` instead.
JSON paths do not expand `$HOME` or `~`. Relative paths resolve from the checkout.

Optionally create a writable scratch directory and set its absolute path as
`scratch_dir`. Dense score exports need substantial disk space. Preserve the full
config; create a separate one-epoch feasibility config:

```bash
mkdir -p runs/configs
python - <<'PY'
import json
from pathlib import Path
config = json.loads(Path('moe/configs/local.json').read_text())
config.update(expert_epochs=1, router_epochs=1)
Path('runs/configs/feasibility.json').write_text(json.dumps(config, indent=2) + '\n')
PY
```

## 5. Validate before training

Submit these in sequence, waiting for each job to finish successfully:

```bash
bash moe/scripts/run_cluster.sh --dry-run test
bash moe/scripts/run_cluster.sh test
bash moe/scripts/run_cluster.sh smoke
bash moe/scripts/run_cluster.sh preflight --config moe/configs/local.json --seed 0
```

`--dry-run` only prints an `sbatch` command. Actual submissions return job IDs;
submission success is not workload success. Check logs and `sacct` after each.
Tests exercise components, smoke uses synthetic data, and preflight verifies source
file existence, split hashes, label mappings, config settings, and allocated CUDA.
Preflight does not decode every image, load the full checkpoint, or prove parity.

## 6. Run real-data feasibility, then full studies

```bash
MOE_VARIANTS="B3 P1" CLUSTER_TIME=04:00:00 CLUSTER_MEM=32G \
  bash moe/scripts/run_cluster.sh source runs/configs/feasibility.json \
  runs/moe-feasibility 0
```

B0, the expert bank, B2, and O1 are automatic prerequisites. B3/P1 run only if
their source gates permit. Review failure reasons, B0 parity, oracle gains, losses,
source selection, and disk/RAM use. One epoch is an execution check, not a final
comparison. Once it passes, run the complete controls with the full config:

```bash
CLUSTER_TIME=12:00:00 CLUSTER_MEM=32G \
  bash moe/scripts/run_cluster.sh source moe/configs/local.json runs/moe-full 0

# After reviewing seed 0, use a fresh root for matched multi-seed comparisons.
CLUSTER_TIME=24:00:00 CLUSTER_MEM=32G \
  bash moe/scripts/run_cluster.sh source moe/configs/local.json runs/moe-matched 0 1 2
```

Times are resource requests, not measured estimates. Seeds run sequentially;
adjust to observed runtime and account limits. P2 needs separately validated
brightness views and successful source comparisons. It is gated by default.

## 7. Monitor and preserve results

```bash
squeue -u "$USER"
tail -f logs/dapo-moe-source-JOB_ID.out
tail -f logs/dapo-moe-source-JOB_ID.err
sacct -j JOB_ID --format=JobID,JobName,Elapsed,State,ExitCode,MaxRSS
```

Replace `JOB_ID`. Expect `COMPLETED` and exit code `0:0`. Results include
`runs/moe-full/seed-0/status.json`, `study.json`, `source-locks.json`, variant
records/checkpoints, prediction exports, and `runs/moe-full/report/`.
Some variants can be scientifically rejected even when the job completes; inspect
their status/reason. A terminated job can leave a stale `running` status.
Existing study directories cannot be overwritten or resumed; use a fresh root.
Keep configs, manifests, source records, and checkpoints unchanged after selection.

## 8. Evaluate a separate target only after source selection

For example, with a prepared MPDD dataset and a qualifying P1 source record:

```bash
bash moe/scripts/run_cluster.sh evaluate \
  --study runs/moe-full/seed-0 --variant B0 --dataset mpdd \
  --data-root /actual/datasets/mpdd --output runs/target-mpdd/seed-0/B0

# Wait for successful B0 completion before submitting P1.
bash moe/scripts/run_cluster.sh evaluate \
  --study runs/moe-full/seed-0 --variant P1 --dataset mpdd \
  --data-root /actual/datasets/mpdd \
  --baseline-result runs/target-mpdd/seed-0/B0/result.json \
  --output runs/target-mpdd/seed-0/P1
```

Replace the target path and use a supported target matching your protocol.
For same-dataset holdouts, supply the preregistered `--manifest target_test.json`.
Evaluation checks source locks, rejects exact source-image duplicates, and requires
the paired B0 result. Never tune on target labels. Retain access to the original
source manifests and reproduced weights when evaluating.

For resource overrides and launcher details, see [the cluster runbook](CLUSTER_EXPERIMENTS.md).
For audit evidence and remaining validation limits, see [the bug audit](BUG_AUDIT.md).
