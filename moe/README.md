# Frozen DAPO residual MoE experiments

This implements the staged study in `docs/DAPO_MoE_Coding_Agent_Instructions.docx`.
It reuses the reproduced checkpoint and reads upstream blocks from `DAPO/` without
editing them. It has no runtime dependency on the removed `Anomaly_detection_CLIP` branch.
For a fresh server checkout, follow [the setup guide](../docs/SERVER_FROM_SCRATCH.md).
See [the bug audit](../docs/BUG_AUDIT.md) for fixes and validation limits.

## Blocks and scoring

| Module | Responsibility |
| --- | --- |
| `dapo.py` | Frozen backbone, visual/text prompts, spatial adapters, preprocessing, original checkpoint loading |
| `blocks/experts.py` | Three per-layer residual bottlenecks, default rank 16, scale 0.1, zero up projections |
| `blocks/router.py` | Two-layer MLP; appearance or competing-label semantics, margin, abnormality, second-slot mask |
| `blocks/targets.py` | Area occupancy, ignored pixels, detached local CE gains, normal/defect-balanced MSE |
| `blocks/selection.py` | Expert argmax only when maximum gain is strictly greater than delta; otherwise baseline |
| `training.py` | Class-balanced expert bootstraps and disjoint router-training data |
| `evaluation.py` | Source selection, dense exports, legacy metrics, routing diagnostics, example panels |
| `experiment.py` | Source gates, locked checkpoints, separate target evaluation |

Each feature layer has its own expert adapters. One router uses the mean raw visual
feature across layers and chooses the same expert at a patch across all layers.
Gain supervision averages per-layer soft-target CE. Global Dice never enters local
gain targets. Semantic ties use sorted candidate names, independent of channel order.
Router abnormality sums all defect probabilities and feature layers, matching the
baseline localization score before upsampling; the top-two margin uses mean-layer probabilities.
The appearance control zeroes semantic inputs to match MLP parameter counts.

Scoring preserves per-layer normalized DAPO similarity and softmax. Binary pixel
scores sum abnormal channels and layers after upsampling; multi-type target scores
select product anchors **before** softmax. Ensembling and task-router mixtures use
per-layer probabilities. Expert 0 is baseline. B2 averages only the three residual
experts. The unchanged CLS branch produces identical binary image scores across
these patch-only variants. All experts are currently executed; hard selection does
not reduce their compute cost.

## Environment and local verification

Use the **existing successful DAPO environment** for real experiments. Upstream
commit inspected: `290411d7374cc2182ad3491b826131af1ff73224`. Keep its vendored
OpenCLIP import path; another installed `open_clip` must not be imported first.
Set `model.pretrained` to a cached/local backbone checkpoint on offline nodes.
`moe/requirements.txt` references upstream requirements; it is not a new validated lock.

For an isolated CPU component-test environment:

```bash
python -m venv .venv
.venv/bin/python -m pip install -r moe/requirements-test.txt
.venv/bin/python -m unittest discover -s moe/tests -v
.venv/bin/python -m moe.cli smoke --output runs/moe-smoke
```

Smoke uses a small synthetic encoder, synthetic images, a synthetic objective,
and relaxed gates. It exercises expert/router training, selection, checkpoint reload,
source and synthetic target exports, and reporting. Its values are **not research
results or full DAPO reproduction evidence**. Output directories must be new.

## Audit annotations and prepare source roles

For a complete MVTec example with generated manifests and a configuration template,
see [examples/mvtec](examples/mvtec/README.md).

Actual checkpoint, dataset roots, and compute budget are not
bundled here. Do not redownload or retrain the reproduced baseline by default.

```bash
python -m moe.cli split --metadata /datasets/mvtec/meta.json \
  --output /manifests/moe-source --seed 0
cp moe/configs/experiment.example.json moe/configs/local.json
```

No manual label or group file is needed for the default workflow. Existing
`cls_name`/`product_cls` and `specie_name`/`defect_cls` fields supply the product and
defect labels. Images are shuffled within each product/species stratum and assigned
approximately 60/20/20 percent to the three roles. Strata with at least three images
reserve one for every role; two images go to experts/validation, and a singleton to
expert training. `summary.json` records incomplete coverage. Images never appear in
multiple roles. Reuse these immutable manifests for matched experiment seeds.

The automatic mode **assumes distinct images are independent**; product/species
labels do not establish physical-object identity. This assumption is recorded in
split summaries and run provenance. If you know images share an instance or
acquisition, pass `--groups /manifests/source-groups.json`. Its format is
`{ "relative/image.png": "verified-instance-or-acquisition-id" }`; shared views
must share a group. This optional mode keeps groups intact and shuffles groups
60/20/20 rather than stratifying images. `inspect-data` can generate an optional
unfilled group template. Check `summary.json` for label coverage in either mode.

Set `baseline_checkpoint`, `source_root`, `splits`, and the exact reproduced model
settings in your local config. Check the supplied source vocabulary and label IDs
against the reproduction; they reproduce the archived training configuration,
including historical aliases, and are not automatically corrected.

Supported supervision:

- `typed_mask`: class-ID masks already aligned with the configured source channels;
  nearest-neighbor ID-preserving resize. Ignore ID is currently `-1`.
- `binary_mask_with_image_type`: spatial binary masks plus a **verified single type
  applying to every positive mask pixel**. Set `single_type_mask_verified=true` only
  after this audit. Combined masks cannot recover individual constituent types.

The example deliberately leaves this audit flag false. Image-level lists alone
are insufficient for dense routing and are rejected. An image-level BCE pilot and
binary-only gain study are not implemented in this initial dense runner. Local
gain utilities handle ignored pixels, but the unchanged upstream full-image
Focal/Dice objective rejects ignored masks; adding an ignore-aware objective needs
an explicit ablation. Tiny masks erased by preprocessing fail with an image ID.

Product-held-out and defect-held-out protocols are separate split options:

```bash
python -m moe.cli split --metadata /datasets/mvtec/meta.json \
  --output /manifests/product-study \
  --held-out-products cable
python -m moe.cli split --metadata /datasets/mvtec/meta.json \
  --output /manifests/defect-study \
  --held-out-defects scratch --aliases /manifests/alias-to-canonical.json
```

Excluded images go into `target_test.json`; with `--groups`, entire related groups
go there together. For defect
holdout, also remove held-out names and synonyms from the source config vocabulary
and remap source label IDs; the config validator checks recorded forbidden names.
Candidate text names can be introduced during target evaluation. CLIP pretraining
exposure remains unknown.

## Run the staged source experiment

```bash
python -m moe.cli run --config moe/configs/local.json \
  --output runs/moe/seed-0 --seed 0 --device cuda --dry-run

# Feasibility: one seed; execute inside the available compute allocation.
bash moe/scripts/run_experiments.sh moe/configs/local.json runs/moe-feasibility 0

# Retained comparisons: three matched seeds, using a new output root.
bash moe/scripts/run_experiments.sh moe/configs/local.json runs/moe-matched 0 1 2
```

Set `MOE_PYTHON=/path/to/environment/bin/python` and `MOE_DEVICE=cpu` if needed.
The script runs numerical tests before experiments, records failed seeds, and
aggregates all attempted seeds. On the server login node, use the cluster launcher:

```bash
export CLUSTER_CONDA_ENV=DAPOenv
bash moe/scripts/run_cluster.sh --dry-run test
bash moe/scripts/run_cluster.sh test
bash moe/scripts/run_cluster.sh smoke
bash moe/scripts/run_cluster.sh source moe/configs/local.json
```

The launcher loads Miniconda and CUDA and runs `source activate` inside each job.
It defaults to `student9696`, one 96 GB MIG, QoS `qos9696`, Conda environment
`DAPOenv`, and CUDA module `cuda12.8`; environment variables override these settings. It preserves
SLURM GPU visibility and uses job IDs in default output directories. See
[the server guide](../docs/CLUSTER_EXPERIMENTS.md) for preparation, full resource
overrides, real-data feasibility jobs, and locked target submissions. The explicit
`submit_source.sh PARTITION TIME MEMORY CONFIG OUTPUT_ROOT [SEED ...]` entry point
still works through the same environment initialization.

| ID | Implementation |
| --- | --- |
| B0 | Frozen reproduced DAPO; repeated inference and zero-residual checks |
| B1 | One rank-48 residual adapter per layer; matches rank-16 × three experts' total parameters |
| B2 | Uniform expert-probability ensemble |
| B3 / B5 | Appearance / semantic soft task-loss routing in training; hard selection at evaluation, baseline included |
| B4 / P1 | Appearance / semantic gain-MSE router |
| P2 | P1 with minimum gains over canonical plus two brightness views |
| O1 | Lowest local-CE oracle including baseline; diagnostic only |
| P1-no-fallback | Gain router with forced nonbaseline selection |
| P1-no-context | Semantic context removed; parameter-matched appearance control |
| P1-shuffle | Fixed random cycle of semantic embeddings across images; own appearance/confidence preserved |

B0/B2/O1 always establish the baseline and expert-bank gate. Use `--variants` on
`moe.cli run` for a retained subset of other controls. All non-B1 comparisons reuse
the same frozen bank per seed. Experts use class-balanced source bootstraps with
independent seeds and identical schedules. B1 is evaluated at its fixed final epoch;
router epochs are selected on source validation. Task heads have one extra baseline
output relative to gain heads; other MLP dimensions match.
P1-no-context uses the same mechanism as B4; matching results are not independent evidence.
The shell launchers accept `MOE_VARIANTS="B3 P1"` for a shorter feasibility job.

Router expansion stops unless the source oracle CE improvement exceeds
`min_oracle_gain`. P2 additionally requires P1 to beat qualifying B2/B3 source F1 and
`robust_views_source_validated=true`. Inspect both configured brightness views on
source defects before enabling this flag. No geometric robustness is claimed.

The provisional source rule maximizes defect F1 subject to worst supported class
regression of at most 0.01. A small preregistered grid chooses delta and one global
defect-probability threshold. F1 averages supported product/defect pairs, excludes
normal, and reports absent pairs as null. The global threshold transfers to unseen
target names. Legacy argmax multi-type F1 and evaluation-threshold F1-max are
reported separately. No qualifying candidate produces `no_qualifying_model`.
Source selection uses the configured training vocabulary for every product; its
legacy metrics are internal controls, not a reproduction of the product-restricted
upstream source table. Target multi-type evaluation restores the exact upstream
product vocabularies. This source-selection protocol choice must be disclosed in reports.

`matched-coverage.json` compares context ablations at P1's source fallback coverage.
For the no-fallback ablation, P1 is calibrated to zero coverage. Ties can prevent
exact matching; requested and achieved source coverage are both recorded. These
comparisons do not recalibrate on target data.

## Evaluate a locked target

Run this only after reviewing the source evidence and freezing the study:

```bash
python -m moe.cli evaluate-target --study runs/moe-matched/seed-0 \
  --variant B0 --dataset mpdd --data-root /datasets/mpdd \
  --output runs/target/seed-0/B0
python -m moe.cli evaluate-target --study runs/moe-matched/seed-0 \
  --variant P1 --dataset mpdd --data-root /datasets/mpdd \
  --baseline-result runs/target/seed-0/B0/result.json \
  --output runs/target/seed-0/P1
```

Use `--manifest /manifests/product-study/target_test.json` for same-dataset holdouts.
The runner rejects exact source-image content overlap and mismatched baseline/bank/router hashes.
`source-locks.json` also anchors the resolved config, split summary, source data/code
inventories, and selection records. Preserve these files after source selection. Every
nonbaseline result requires the paired B0 target result from the same data and seed.
It does not optimize epochs, thresholds, or delta on target labels. O1 is unavailable
as a target deployment. Current target loading expects binary masks with image
defect types as in the verified upstream datasets; finer target annotations need an
explicit loader adaptation.

## Artifacts and limits

Each study exports resolved config/provenance, shape checks, image/mask SHA256
inventory, immutable split hashes, bootstrap records, expert and router checkpoints,
epoch histories, per-class CSVs, source locks, pooled/group/class-balanced routing
diagnostics, legacy metrics, confusion matrices, and compressed prediction arrays.
Mixed patches contribute to every class they contain in class-balanced routing
diagnostics. Example panels show original image, GT, baseline, selected output, and
expert index. Ensemble exports mark expert index `-1` and have no routing rates.

Disk-backed dense scores can be large: roughly `N × H × W × (4C + 8)` bytes plus
exported compressed predictions. Set `scratch_dir` to an existing scratch directory
with enough space. Disk maps are temporary; exports remain in the study output.

```bash
python -m moe.cli report --studies runs/moe-matched/seed-0 \
  runs/moe-matched/seed-1 runs/moe-matched/seed-2 --output runs/moe-matched/report
# Target report: supply target output directories for one dataset scope.
python -m moe.cli report --studies runs/target/seed-0/B0 \
  runs/target/seed-0/P1 --output runs/target/report
```

Reports include all attempted statuses, mean/sample SD, and paired baseline deltas.
One seed has null sample SD. Failures and P2 stops remain visible. The source runner
does not start target tests. Official MoECLIP reproduction, bootstrap confidence
intervals, and FLOPs profiling remain outside this implementation; null values carry
reasons. Full checkpoint/dataset equivalence must still be checked in the reproduced
server environment. No scientific run has been executed during local development.
