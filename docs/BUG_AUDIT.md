# Project bug audit

Audit date: 2026-11-15. Scope: `moe/`, source manifests, metrics, study locking,
reporting, and cluster submission. The upstream `DAPO/` reference was not modified.

## Findings and fixes

| Impact | Finding | Implemented correction |
| --- | --- | --- |
| High | Target evaluation could use a changed source configuration or selection record. | `experiment.py` writes `source-locks.json` and verifies configuration, selection records, split summary, data inventory, and checkpoint hashes before evaluation. Local backbone files are hashed too. |
| High | Different image paths could hide identical source/target images or duplicate content across training/validation roles. | `io.py` and `experiment.py` reject cross-role and source/target image SHA256 overlap; target manifests also reject duplicate IDs. This does not detect visually similar acquisitions. |
| High | Predictions at ignored pixels increased primary F1 false positives. | `evaluation.py` excludes ignored pixels from both truth and predictions. `metrics.py` returns null metrics for entirely ignored arrays. |
| Medium | Dense metrics copied the full image-by-class score array into RAM despite memory-mapped exports. | Metrics retain the score memmap and process individual channels; valid-label argmax no longer copies the entire class matrix. Large datasets still require adequate RAM and scratch space. |
| Medium | Invalid source assets and parameters failed after expensive initialization. | `config.py` checks image/mask existence, positive defect mappings, model checkout, numeric settings, and explicit mask-audit confirmation during preflight. |
| Medium | Configuration failures were absent from attempted-run reports; a later-stage failure could mislabel an earlier variant. | Failed preflight attempts now persist status/study records; stage tracking covers matched-coverage evaluation. |
| Medium | Aggregation could combine incompatible baselines/configurations; repeated seed arguments could create misleading attempts. | Reports verify protocol and provenance compatibility; the source launcher rejects duplicate seed arguments. |
| Operational | Long training appeared silent and expert epoch history was only saved at the end. | Expert/router epoch progress is logged; expert history is persisted after each epoch. |

## Validation

All **37 CPU tests passed**, including nine mocked cluster-launcher tests, actual
upstream prompt/adapter/loss interface checks with a tiny encoder, and the synthetic
source-to-target workflow. Regression cases cover ignored labels, missing assets,
duplicate content, altered locks/configurations/router weights, incompatible reports,
failed attempts, and retained-variant forwarding. Shell syntax, Python compilation,
and whitespace checks passed.

## Server validation still required

No real dataset images, reproduced checkpoint, or GPU allocation were available.
The tests establish software behavior, not experimental accuracy or full-checkpoint
parity. Run the reproduced DAPO evaluator with its original command and compare B0
on the same target protocol before interpreting MoE gains. Source selection uses
the configured training vocabulary; target evaluation restores product-specific
vocabularies, so source scores are not the upstream source-table reproduction.

All experts execute before selection; routing currently saves no expert compute.
Automatic splitting assumes independent images; use verified groups for related
views. Old studies without the new locks need revalidation through a fresh run.
See [the fresh-server guide](SERVER_FROM_SCRATCH.md) for the execution sequence.
