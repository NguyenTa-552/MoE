"""Aggregate every attempted seed with sample standard deviations."""

from collections import defaultdict
from pathlib import Path
import statistics

from .io import read_json, write_json


def aggregate(studies, output):
    grouped = defaultdict(list)
    failures = []
    synthetic = False
    scope = None
    seen = set()
    for directory in map(Path, studies):
        status = read_json(directory / 'status.json')
        if status['status'] != 'completed':
            failures.append(dict(path=str(directory), **status))
        if (directory / 'result.json').is_file():
            record = read_json(directory / 'result.json')
            records = {record['variant']: record}
        elif (directory / 'study.json').is_file():
            records = read_json(directory / 'study.json')['results']
        else:
            records = {}
        for variant, record in records.items():
            run_scope = (record['split'], record.get('dataset'))
            if scope is not None and run_scope != scope:
                raise ValueError('Aggregate one source/target dataset scope at a time')
            scope = run_scope
            key = (variant, record['seed'])
            if key in seen:
                raise ValueError(f'Duplicate variant/seed in report: {key}')
            seen.add(key)
            grouped[variant].append(record)
            synthetic |= record.get('synthetic_smoke', False)
    table = {}
    for variant, runs in grouped.items():
        values = [r['metrics']['macro_defect_f1'] for r in runs
                  if r.get('metrics') is not None and r['metrics']['macro_defect_f1'] is not None]
        paired = [r['comparison']['macro_delta'] for r in runs
                  if r.get('comparison') is not None and r['comparison']['macro_delta'] is not None]
        table[variant] = dict(attempted=len(runs), measured=len(values),
            seeds=[r['seed'] for r in runs], statuses=[r['status'] for r in runs],
            mean=statistics.mean(values) if values else None,
            sample_std=statistics.stdev(values) if len(values) > 1 else None,
            std_reason=None if len(values) > 1 else 'Fewer than two measured seeds',
            mean_paired_delta_to_B0=statistics.mean(paired) if paired else None,
            sample_std_paired_delta=statistics.stdev(paired) if len(paired) > 1 else None,
            runs=runs)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'summary.json', dict(variants=table, failed_studies=failures,
               synthetic_smoke=synthetic, scope=scope))
    lines = ['# Experiment report', '', '## 1. Decision and evidence', '',
             'Synthetic smoke validation only; these are not research results.' if synthetic else
             'Review the complete comparisons and worst-class regressions before continuing. Novelty is unconfirmed.', '',
             f'Evaluation scope: {scope}. Failed studies: {len(failures)}.', '',
             '## 2. Implementation and supervision', '',
             'Frozen DAPO with residual visual experts; separate source expert/router training. '
             'Read each run’s supervision_mode, inspection.json, config.json, and provenance.json.', '',
             '## 3. Splits and selection', '',
             'Immutable grouped source manifests separate expert training, router training, and validation. '
             'Epoch, gain threshold, and global defect-probability threshold are selected on source validation '
             'under the configured worst-class regression constraint. Target evaluation uses these locked values.', '',
             '## 4. Complete comparison', '',
             '| Variant | Attempted | Measured | Macro defect F1 mean | Sample SD | Paired delta to B0 |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    def number(value):
        return 'null' if value is None else f'{value:.6f}'
    for variant, row in table.items():
        lines.append(f'| {variant} | {row["attempted"]} | {row["measured"]} | {number(row["mean"])} | {number(row["sample_std"])} | {number(row["mean_paired_delta_to_B0"])} |')
    lines += ['', 'Attempted run statuses:', '']
    for variant, row in table.items():
        for run in row['runs']:
            lines.append(f'- {variant}, seed {run["seed"]}: {run["status"]}. {run.get("reason") or ""}')
    for failure in failures:
        lines.append(f'- Failed study {failure["path"]}: {failure.get("error", failure["status"])}')
    lines += ['', '## 5. Ablations and oracle gap', '',
              'Source study artifacts include complementarity.json and matched-coverage.json. '
              'The label-informed O1 is a local-loss diagnostic and is not deployable or an AUROC bound.', '',
              '## 6. Per-class regressions and examples', '',
              'Read per-class CSVs, per_class_deltas/worst_class_delta in run records, and prediction exports. '
              'Representative panels show the image, ground truth, baseline, selected prediction, and expert index; '
              'sampling ranks selected-minus-baseline local CE and is recorded in failure_examples.json.', '',
              '## 7. Compute and reproducibility', '',
              'Run records contain parameter counts and artifact hashes; study.json records elapsed time and peak device memory. '
              'All experts are computed before selection, so conditional adapter counts do not imply lower executed cost. '
              'Full invocations, dependency versions, baseline/data inventories, and split hashes are exported.', '',
              '## 8. Limitations and next action', '',
              'A stopped or no_qualifying_model status requires diagnosis before extending the study. '
              'No qualifying model is retained when the source constraint fails. '
              'Official MoECLIP reproduction, confidence intervals, and FLOPs are not supplied by this runner. '
              'Binary image scoring is frozen; patch-only adapters cannot improve its CLS score. '
              'A positive source result justifies a locked target test, not a novelty claim.']
    (output / 'report.md').write_text('\n'.join(lines) + '\n')
    return table
