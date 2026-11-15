# MoE implementation and local validation

The implementation is in `moe/` on branch `moe-experiments`. Initial local validation
preceded the first repository commit; scientific run records capture the accessible
HEAD and report missing commits as null rather than inventing provenance.
`DAPO/` remains clean at upstream commit
`290411d7374cc2182ad3491b826131af1ff73224`.

## Implemented

- Frozen-DAPO wrapper with original checkpoint aliases, separate raw/projected
  features, and the original normalization, temperature, interpolation, and CLS branch.
- Residual expert bank, semantic/appearance routers, detached ordinary/robust gain
  targets, strict baseline fallback, local loss diagnostics, and explicit annotation modes.
- Immutable source-role manifests with automatic product/species-stratified images
  or optional verified instance groups, separate product/defect holdout options,
  class-balanced expert bootstraps, shared-bank controls B0–B5/P1/P2/O1, and P1 ablations.
- Source-only checkpoint/threshold selection, complementarity and P2 gates,
  matched-source-coverage comparisons, separately invoked locked target evaluation,
  and paired target comparisons with B0.
- Dataset/checkpoint/split hashes, configs, epoch histories, prediction arrays,
  confusion matrices, per-class CSVs, routing diagnostics, representative panels,
  and seed aggregation with sample standard deviations and attempted statuses.

See `moe/README.md` for exact commands and configuration requirements.
Server submission follows `docs/CLUSTER_EXPERIMENTS.md`. The cluster launcher loads
modules and activates the reproduced environment inside the SLURM allocation, requests
the partition's exact MIG GRES, and preserves scheduler GPU visibility. Nine mocked
launcher tests cover submission and environment initialization without server access.

## Verification

The latest audit passed all 37 CPU tests. Findings and remaining server validation
are recorded in [BUG_AUDIT.md](BUG_AUDIT.md); fresh-server commands are in
[SERVER_FROM_SCRATCH.md](SERVER_FROM_SCRATCH.md).

The CPU suite covers baseline/fallback equivalence, real upstream prompt/adapter
interfaces and Focal/Dice losses with a tiny encoder, gradient isolation, checkpoint
reload, gain signs, robustness views, selection boundaries, deterministic ties,
label permutations, ignored/mixed/tiny patches, empty-defect cases, grouped split
hashes, held-out synonyms, metrics recomputed from exported arrays, and router
tampering rejection. The end-to-end synthetic workflow includes source training,
selection, checkpoint reload, synthetic target evaluation, and report generation.

```bash
.venv/bin/python -m unittest discover -s moe/tests -v
bash -n moe/scripts/run_experiments.sh moe/scripts/submit_source.sh moe/scripts/run_cluster.sh
python -m compileall -q moe
```

No pretrained checkpoint, real dataset images, or GPU allocation was supplied in
this workspace. No scientific training or target experiment was launched. Synthetic
outputs are software validation only, with explicit synthetic encoders/objectives
and relaxed gates; they provide no evidence of research performance.

## Scope and departures requiring disclosure

The initial runner supports dense typed masks or audited single-type binary masks
with image labels. Image-list-only BCE pilots and binary-only gain studies remain
unimplemented. Ignored masks work for local gains but are rejected by the unchanged
full-image upstream training loss. Source selection uses the configured source
training vocabulary; target evaluation uses upstream product-specific vocabularies.

One shared router chooses a patch's expert across feature layers; gain targets
average per-layer local CE. The frozen CLS term is constant and omitted from expert
optimization. B1 uses a fixed final epoch. P1-no-context duplicates B4's mechanism.
All experts are executed before hard selection. FLOPs, confidence intervals, and
official MoECLIP reproduction remain uncomputed and are reported accordingly.

Automatic metadata splitting assumes independent images, without inferring physical
instance IDs from product/species labels. This departure from verified acquisition/
instance grouping is recorded in summaries and run provenance. Known related views
can still be kept together with an explicit group map.

Before real execution, locate the existing reproduced baseline and dataset roots,
audit mask semantics, create source manifests from existing labels, inspect split
coverage, configure the existing server environment, and verify full checkpoint
parity against the reproduction artifacts. Enable P2 only after source validation
of its two brightness views and successful source control comparisons.
