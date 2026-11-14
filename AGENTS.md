# Repository Guidelines

## Project Structure & Module Organization

This workspace currently contains `DAPO_MoE_Coding_Agent_Instructions.docx`, the implementation specification and reporting contract for a DAPO-based mixture-of-experts research prototype. No source tree, tests, datasets, checkpoints, or dependency manifests are present. Git history is not accessible from this directory.

Before implementation, verify the intended DAPO checkout and branch. Locate the baseline model, dataset loader, text embeddings, losses, evaluator, and checkpoint loader; record their actual paths and tensor contracts. Keep experts, routing, gain targets, selection, configuration, and diagnostics in separate modules following the verified checkout’s organization.

## Build, Test, and Development Commands

No build, installation, training, or test commands are established in this workspace. Do not assume commands such as `pytest` or invent entry points.

After obtaining the source checkout:

- Run `git status --short` and `git rev-parse HEAD` to record local changes and provenance.
- Follow its dependency manifest and documented setup commands.
- Run the existing baseline evaluator before modifying model behavior.
- Record exact smoke-test, training, and evaluation commands with resolved configurations.

## Coding Style & Naming Conventions

Follow the verified source project’s indentation, naming, formatter, and lint configuration; none is currently supplied here. Use descriptive names for expert indices, gains, thresholds, and supervision modes. Make research settings configurable and document tensor dimensions. Preserve baseline preprocessing, label mapping, normalization, pooling, and scoring unless explicitly testing an isolated ablation.

## Testing Guidelines

No testing framework or coverage threshold is configured. Add correctness checks for baseline equivalence with zero residuals and forced fallback, frozen gradients, detached gain targets, checkpoint round trips, threshold boundaries, ties, ignored labels, and candidate-label permutations. Verify grouped split disjointness. Complete a tiny CPU or device smoke run before long experiments.

## Commit & Pull Request Guidelines
Use folder DAPO as git clone from https://github.com/boschresearch/defect-aware-prompt-optimization/tree/main
Do not change or affect folder DAPO, only use it as reference

Use folder moe as the main folder for implementation
Existing commit conventions cannot be verified. Use concise, imperative commit subjects and keep changes focused. Pull requests should describe the mechanism, specification deviations, validation commands and outcomes, and remaining blockers. Link relevant issues and reproducible artifacts when available; report unrun stages explicitly.

## Research & Artifact Discipline

Use source data for training, selection, and calibration; reserve target tests for locked evaluation. Preserve baseline checkpoints and user edits. Record seeds, configurations, split and checkpoint hashes, failed runs, and per-class regressions. Represent unavailable metrics as null with a reason, and never present smoke runs as experimental results.
