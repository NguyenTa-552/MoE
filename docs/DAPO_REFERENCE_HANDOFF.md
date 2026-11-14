# DAPO Reference Handoff

Captured on 2026-11-14 before the closed `Anomaly_detection_CLIP/` branch is removed. This note and its companion artifacts live outside that folder. Paths below describe the old checkout; they are not dependencies for the future implementation.

## Status and provenance

- The user confirms DAPO has already been reproduced and datasets are prepared. Do not restart reproduction or download replacement datasets by default.
- Inspected checkout: `Anomaly_detection_CLIP`, commit `7df36a39527d5c475735339cfa0abfa5bbd10cf6`; `git status --short` was empty when captured.
- Upstream `DAPO/`: commit `290411d7374cc2182ad3491b826131af1ff73224`, from `https://github.com/boschresearch/defect-aware-prompt-optimization.git`.
- Project instructions require implementation in the workspace's `moe/` directory. Treat DAPO as an unchanged reference. Neither folder removal nor a new training run was performed during this handoff.
- No `.pt` or `.pth` checkpoint was found in the inspected checkout. Sample image paths from each metadata file did not resolve relative to the bundled metadata directory. The prepared assets may be on the server; their actual locations, baseline checkpoint hash, environment, and compute allocation remain unverified locally.

## Preserved artifacts

- `reference/anomaly_detection_clip_reference.zip`: 114 reference files, including `clip_ad/`, tests, configs, scripts, historical plans, 58 evaluation JSON reports, six dataset metadata files, selected upstream top-level source files, dependency list, and the backbone model config.
- `reference/anomaly_detection_clip_manifest.json`: source commits, captured status, archive SHA256, and every archived file's size and SHA256.
- The ZIP was checked for integrity and every member checked against its SHA256. It excludes Git internals, the full vendored OpenCLIP, images, masks, checkpoints, and the execution environment. It is an archival reference, not a standalone runnable checkout.
- The primary research contract remains `DAPO_MoE_Coding_Agent_Instructions.docx` in the workspace root.

Extract the archive into a separate reference directory if needed. Obtain the full upstream source at the recorded commit to restore vendored OpenCLIP. Future runtime code must not import from the folder scheduled for removal.

## Code map and tensor interfaces

| Concern | Historical path and symbol | Contract |
| --- | --- | --- |
| Construction | `clip_ad/model.py:build_model`, `ModelConfig` | Returns model and matching preprocessing; normal is channel 0. |
| Forward | `clip_ad/model.py:SpatialSpectralCLIP.forward` | Image encoder → spatial adapter → optional fusion → text encoder → readout. No labels enter the model. |
| Adapter | `DAPO/adapter.py:LinearAdapter` | One linear projection per selected layer; removes CLS from 3D token tensors. **Mutates the supplied list** by replacing entries. |
| Text | `DAPO/learnable_prompt.py:LearnablePrompt` | Shared learned context with candidate-name templates; `forward(img_emb)` currently does not use image conditioning because the meta-net path is commented out. |
| Readout | `clip_ad/readout.py:DenseReadout` | L2 normalization, similarity / temperature, softmax, square-grid reshape, bilinear upsampling with `align_corners=True`. |
| Data | `clip_ad/data.py:validate_dataset`, `normalize_sample`, `build_dataset` | Reads nested `test` or flat class-to-samples metadata and delegates sample reading to DAPO. |
| Training | `clip_ad/train.py`, `clip_ad/losses.py` | Baseline objectives, optimizer setup, seed handling, checkpoint creation. |
| Evaluation | `clip_ad/evaluate.py`, `evaluation_protocol.py`, `multitype.py`, `metrics.py` | Separate binary and typed scoring rules and metric aggregation. |

Configured defaults: `ViT-L-14-336-quickgelu`, OpenAI pretrained weights, forced image size 518, feature layers `[6,12,18,24]`, temperature 0.07. Vision width is 1024 and projected/text dimension is 768. The expected patch grid is 37 × 37, or 1369 patches, with CLS present before the spatial adapter. These dimensions follow code/config inspection; verify runtime tensors before implementation.

For each layer, preserve pre-adapter `h: [B,1369,1024]` separately from projected `z0: [B,1369,768]`. Copy the token list before invoking the adapter so it cannot destroy access to `h`. Text anchors are `[C,768]` including normal. Readout exposes normalized patch embeddings, text embeddings, image embeddings, per-layer maps, and binary CLS logits. Its normalized embeddings alone do not preserve unnormalized `z0` for residual insertion.

## Baseline behavior to preserve

- **Binary localization:** softmax over normal and all individual defect anchors at each patch, upsample, sum abnormal channels, then sum feature layers. `dapo_anomaly_map` implements this. The readout's two-anchor `binary_probabilities` and its layer-averaged maps are different scoring rules.
- **Binary image score:** normalized CLS against normal and the normalized mean of normalized abnormal text anchors; temperature 0.07; take abnormal softmax probability. Residual patch adapters alone do not change this CLS branch.
- **Multi-type localization:** select normal and the product-specific defect anchors **before softmax**, upsample each layer, then sum layers. Selecting columns from a dataset-wide probability map changes the denominator.
- Binary names come from `DAPO/test.py`; multi-type names and aliases come from `DAPO/test_multi_defect.py` as captured in `clip_ad/dapo_multitype.json`. They can differ, including spelling and case. Preserve exact lists rather than normalizing them ad hoc.
- Binary metrics: image/pixel AUROC, AP, F1-max; pixel AUPRO. F1-max chooses a threshold on evaluation labels and is descriptive only. Undefined binary metrics are null.
- AUPRO: 200 sampled thresholds, strict score comparison, 8-connected regions with equal region weights, FPR strictly below 0.3, normalization by the retained FPR range. This is not interpolation to exactly 0.3.
- Multi-type metrics include normal. AP averages every supplied class, with absent-class AP zero; AUROC excludes undefined classes; argmax macro F1 averages labels appearing in truth or prediction. Dataset aggregation gives products equal weight. This legacy metric is distinct from the proposed defect-only primary outcome.

## Data contract and supervision limitations

Metadata fields include `img_path`, `mask_path`, `anomaly`, product (`cls_name` or `product_cls`), and defect type (`specie_name` or `defect_cls`). Normalized records supply the canonical keys. The loader rejects anomalous samples missing a defect label.

Bundled evaluation/source-record counts: MVTec 1725, VisA 1878, MPDD 458, MAD-Real 271, MAD-Sim 5589, Real-IAD 19802. These are metadata record counts, not verified on-disk image counts or independent split sizes.

The existing reader uses a binary mask and one image-level defect name. `typed_targets` assigns that name's channel ID to all positive mask pixels; normal is 0. It cannot distinguish multiple defect types within one combined mask. Validate the actual prepared annotations before claiming dense semantic supervision. Maintain explicit combined-category limitations.

Image preprocessing uses square bicubic resize and center crop; mask preprocessing uses resize, crop, and tensor conversion. Training converts masks to integer before thresholding; evaluation uses `> 0.5`. Keep baseline behavior and isolate any improved supervision resampling. Mixed patch occupancy and tiny-defect preservation need explicit checks.

Historical source training intentionally used MVTec's metadata `test` partition, following upstream supervised source training. This does not authorize target-test training. The old trainer uses all source records without expert/router/validation separation. Create immutable grouped manifests for those roles; metadata does not establish acquisition-group independence by itself.

## Checkpoints and historical commands

Original checkpoint aliases accepted by the wrapper:

| Model block | Accepted checkpoint keys |
| --- | --- |
| Visual prompts | `visual_prompt` or `visual_prompt_state_dict` |
| Text prompts | `prompt_learner` or `prompt_learner_state_dict` |
| Spatial adapters | `linear_adapter` or `trainable_adapter_state_dict` |

Wrapper checkpoints have `format_version=1`, `model_config`, `defect_names`, and `blocks`; training adds epoch, experiment, optimizer, and Python/NumPy/Torch/CUDA RNG states. Frozen backbone weights are reconstructed from `pretrained`, not saved. Learned blocks use `visual_prompt`, `text_prompt`, `spatial_adapter`, and optional `fusion`. Template tensors are nonpersistent buffers rebuilt for candidate names.

Historical commands, run from the old checkout root, are recorded for reference only; placeholder asset paths require the real server locations:

```bash
python -m unittest discover -s tests -v
python -m clip_ad.evaluate --checkpoint /checkpoints/dapo/epoch_4.pth \
  --config configs/baseline.json --dataset visa --data-root /datasets/visa \
  --tasks both --output runs/baseline/visa.json
```

`--init-dapo` starts additional training; it is not evaluation or optimizer resume. `--resume` restores wrapper training state and requires identical model/experiment configuration. Do not launch the historical spectral commands for the MoE study.

Historical baseline config: five epochs, batch size 8, learning rate 0.001, seed 42; Adam betas `(0.5,0.999)`; objective `4 × sum_layers(Focal + Dice_normal + Dice_anomaly) + CE_image`. All three old configurations share those settings. They are reference settings, not verified settings of the user's reproduced checkpoint. The archived dependency list is likewise historical, not a validated environment lock.

`clip_ad/vendor.py` explicitly selects `DAPO/open_clip`; importing an unrelated installed `open_clip` first raises an error. Restore this import discipline in any future integration. Dense multi-type evaluation uses temporary disk maps and supports `--scratch-dir`.

## Historical evidence and cautions

The snapshot contains 20 baseline, 20 spectral, and 18 region reports across MPDD, VisA, MAD-Real, and MAD-Sim. Inspected examples contain only `dataset`, `defect_names`, `classes`, and `macro`, without checkpoint, data-root, epoch, or protocol metadata. Do not attribute their metrics to the current evaluator or infer a checkpoint hash from them. Re-evaluate the existing reproduced weights only if needed to establish comparable evidence; no retraining is implied.

Historical plans are not authoritative measurements. In particular, `Plan/DAPO_THREE_EXPERIMENT_COMPARISON.md` claims five MPDD region reports are byte-identical to baseline. A fresh comparison of the current snapshot found **zero** byte-identical region/baseline pairs for every dataset. Preserve the discrepancy; do not repeat the old claim as a verified fact or infer why the files changed.

## Next implementation decisions

1. Locate the existing reproduced checkpoint and prepared data roots, record their hashes and environment, and establish verified insertion shapes. Keep the baseline backbone, spatial adapters, visual prompts, and text branch frozen.
2. Build in `moe/` with no runtime dependency on the deleted branch. Restore/reference the exact upstream commit independently. Keep spectral fusion and region alignment disabled for this study.
3. Verify annotation scope and construct disjoint expert-training, router-training, and source-validation manifests. Gain targets must come from examples not used to train experts, or documented cross-fitting.
4. Start with three residual experts, provisional rank 16 and scale 0.1, zero-initialized up projections, gain routing, and an exact baseline fallback. Use the research document for semantic context and staged controls.
5. Preserve baseline scoring, detach gain targets, and verify zero-residual/fallback equivalence, frozen gradients, checkpoint round trips, label permutation behavior, ignored/mixed patches, ties, and threshold boundaries before a smoke run.
6. Establish expert complementarity before expanding routers. Select settings on source validation and lock them before target evaluation. Report failures and regressions alongside gains.
