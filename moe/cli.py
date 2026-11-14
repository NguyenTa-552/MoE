"""Command-line entry points; split preparation requires only Python."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    split = commands.add_parser('split', help='Create immutable grouped source-role manifests')
    split.add_argument('--metadata', required=True)
    split.add_argument('--groups', required=True, help='JSON mapping img_path to verified instance/acquisition ID')
    split.add_argument('--output', required=True)
    split.add_argument('--seed', type=int, default=0)
    split.add_argument('--held-out-products', nargs='+', default=[])
    split.add_argument('--held-out-defects', nargs='+', default=[])
    split.add_argument('--aliases', help='JSON alias-to-canonical names, required for defect-held-out protocol')
    inspect = commands.add_parser('inspect-data', help='List metadata fields/counts and create an unfilled group template')
    inspect.add_argument('--metadata', required=True)
    inspect.add_argument('--output', required=True)
    run = commands.add_parser('run', help='Run source-only staged experiments')
    run.add_argument('--config', required=True)
    run.add_argument('--output', required=True)
    run.add_argument('--seed', type=int, default=0)
    run.add_argument('--device', default='cuda')
    run.add_argument('--variants', nargs='+', help='Controls to run; B0/B2/O1 always establish baseline and bank gates')
    run.add_argument('--dry-run', action='store_true', help='Validate paths/splits and print a plan without loading weights')
    target = commands.add_parser('evaluate-target', help='Evaluate a qualifying source-locked model without tuning')
    target.add_argument('--study', required=True)
    target.add_argument('--variant', required=True)
    target.add_argument('--dataset', required=True)
    target.add_argument('--data-root', required=True)
    target.add_argument('--output', required=True)
    target.add_argument('--device', default='cuda')
    target.add_argument('--manifest', help='Explicit target_test.json for same-dataset held-out protocols')
    target.add_argument('--baseline-result', help='B0 target result.json from the same split and seed; required for nonbaseline variants')
    smoke = commands.add_parser('smoke', help='Synthetic CPU training/selection/checkpoint/report exercise')
    smoke.add_argument('--output', required=True)
    report = commands.add_parser('report', help='Aggregate attempted seeds within one source/target dataset scope')
    report.add_argument('--studies', nargs='+', required=True)
    report.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.command == 'split':
        from .splits import create_splits
        from .io import read_json
        result = create_splits(args.metadata, args.groups, args.output, args.seed,
                               held_out_products=args.held_out_products,
                               held_out_defects=args.held_out_defects,
                               aliases=None if args.aliases is None else read_json(args.aliases))
    elif args.command == 'inspect-data':
        from collections import Counter
        from .splits import records_from_metadata
        from .io import write_json, sha256
        rows = records_from_metadata(args.metadata)
        result = dict(records=len(rows), products=dict(Counter(r['product'] for r in rows)),
                      defects=dict(Counter(r['defect'] for r in rows)), metadata_sha256=sha256(args.metadata),
                      instruction='Fill each null with a verified instance/acquisition group; image IDs alone do not establish independence')
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / 'inspection.json', result, exclusive=True)
        write_json(output / 'groups.template.json', {r['image_id']: None for r in rows}, exclusive=True)
    elif args.command == 'run' and args.dry_run:
        from .config import load_config
        from .splits import validate_splits
        from .io import sha256
        config = load_config(args.config)
        manifests = validate_splits(config['splits'])
        result = dict(status='dry_run_only', baseline_sha256=sha256(config['baseline_checkpoint']),
                      source_counts={k: len(v['rows']) for k, v in manifests.items()},
                      output=args.output, seed=args.seed, device=args.device,
                      target_evaluation='not scheduled', weights_loaded=False)
    elif args.command == 'run':
        from .experiment import run_source
        records = run_source(args.config, args.output, args.seed, args.device, args.variants)
        result = {variant: record['status'] for variant, record in records.items()}
    elif args.command == 'evaluate-target':
        from .experiment import evaluate_target
        result = evaluate_target(args.study, args.variant, args.dataset, args.data_root, args.output,
                                 args.device, args.manifest, args.baseline_result)
    elif args.command == 'smoke':
        from .smoke import run_smoke
        result = run_smoke(args.output)
    else:
        from .report import aggregate
        table = aggregate(args.studies, args.output)
        result = {variant: {key: value for key, value in row.items() if key != 'runs'}
                  for variant, row in table.items()}
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
