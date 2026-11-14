"""Stage-gated source study and separately locked target evaluation."""

import importlib.metadata
from pathlib import Path
import platform
import random
import sys
import time

import numpy as np
import torch

from .blocks.experts import ExpertBank
from .blocks.router import GainRouter
from .config import load_config
from .dapo import FrozenDAPO
from .evaluation import evaluate, compare_to_baseline, export_per_class
from .io import git_commit, read_json, sha256, write_json, dataset_inventory
from .model import VARIANTS, TASK_VARIANTS, SEMANTIC_VARIANTS, expert_logits
from .objectives import DAPOObjective
from .splits import validate_splits, records_from_metadata
from .training import train_experts, train_router_epoch, shuffled_donors


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_bank(baseline, config, seed, single=False):
    # Bias-free bottlenecks permit exact total-parameter matching for B1.
    experts = 1 if single else config['experts']
    rank = config['rank'] * config['experts'] if single else config['rank']
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        bank = ExpertBank(baseline.input_dim, baseline.output_dim, len(baseline.layers),
                          experts, rank, config['alpha'])
        bank.initialization_seeds = []
        for index, expert in enumerate(bank.experts):
            initialization_seed = seed * 1000 + index + 1
            bank.initialization_seeds.append(initialization_seed)
            torch.manual_seed(initialization_seed)
            for adapter in expert:
                adapter.down.reset_parameters()
                torch.nn.init.zeros_(adapter.up.weight)
    return bank.to(next(baseline.parameters()).device)


def make_router(baseline, config, variant):
    return GainRouter(baseline.input_dim, baseline.output_dim, config['experts'],
                      config['hidden_dim'], semantic=variant in SEMANTIC_VARIANTS,
                      task_loss=variant in TASK_VARIANTS).to(next(baseline.parameters()).device)


def provenance(config, config_path, seed):
    packages = {}
    for package in ('torch', 'torchvision', 'numpy', 'scipy', 'scikit-learn', 'timm'):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    return dict(seed=seed, git_commit=git_commit(),
                git_commit_reason=None if git_commit() else 'Repository has no accessible HEAD commit',
                baseline_sha256=sha256(config['baseline_checkpoint']), config_sha256=sha256(config_path),
                split_manifest_sha256=sha256(Path(config['splits']) / 'summary.json'),
                supervision_mode=config['supervision_mode'], vocabulary=config['defect_names'],
                python=sys.version, platform=platform.platform(), dependencies=packages,
                invocation=sys.argv, regression_tolerance=config['regression_tolerance'],
                baseline_training_seed=config.get('baseline_training_seed'),
                baseline_training_seed_reason=None if config.get('baseline_training_seed') is not None else 'Not supplied; MoE seeds do not identify baseline training seed',
                pretrained_exposure='unknown', synthetic_smoke=config.get('synthetic_smoke', False),
                score_space='DAPO per-layer softmax probabilities; expert ensembles average probabilities',
                limitations=['Image-type assignment cannot recover distinct types within combined masks']
                if config['supervision_mode'] == 'binary_mask_with_image_type' else [])


def choose(evaluation, baseline_primary, tolerance):
    selected = None
    for primary in evaluation['primary']:
        if primary['macro_defect_f1'] is None:
            continue
        comparison = None if baseline_primary is None else compare_to_baseline(primary, baseline_primary)
        worst = None if comparison is None else comparison['worst_class_delta']
        if comparison is not None and (worst is None or worst < -tolerance):
            continue
        if selected is None or primary['macro_defect_f1'] > selected['primary']['macro_defect_f1']:
            selected = dict(primary=primary, comparison=comparison)
    return selected


def save_result(directory, variant, base, selection, evaluation, status='completed',
                epoch=None, delta=None, reason=None, bank_hash=None, router_path=None,
                resources=None):
    record = dict(base, run_id=f'{variant}-seed-{base["seed"]}', variant=variant,
                  split='source_val', status=status, reason=reason,
                  selection_metric=f'macro_defect_f1; worst-class delta >= -{base["regression_tolerance"]} (provisional)',
                  selected_epoch=epoch, delta=delta, expert_bank_sha256=bank_hash,
                  metrics=None if selection is None else selection['primary'],
                  per_class=None if selection is None else selection['primary']['per_class'],
                  comparison=None if selection is None else selection['comparison'],
                  routing=None if evaluation is None else evaluation['routing'],
                  resources=resources, artifact_paths=dict(router=router_path,
                      metrics=str(directory / f'{variant}-evaluation.json') if evaluation else None,
                      per_class=str(directory / f'{variant}-per-class.csv') if selection else None,
                      predictions=str(directory / 'predictions' / variant) if (directory / 'predictions' / variant).is_dir() else None),
                  evaluation_runtime_seconds=None if evaluation is None else evaluation['evaluation_runtime_seconds'],
                  resources_reason=None if resources else 'Per-variant resources not profiled; see study-level runtime and memory',
                  router_sha256=sha256(router_path) if router_path else None,
                  supervision_mode=base['supervision_mode'])
    write_json(directory / f'{variant}.json', record)
    if selection is not None:
        export_per_class(directory / f'{variant}-per-class.csv', selection['primary'], selection['comparison'])
    if evaluation is not None:
        write_json(directory / f'{variant}-evaluation.json', evaluation)
    return record


def run_source(config_path, output, seed=0, device='cuda', variants=None):
    from .data import ManifestDataset
    config = load_config(config_path)
    manifests = validate_splits(config['splits'])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    base = provenance(config, config_path, seed)
    write_json(output / 'config.json', config, exclusive=True)
    write_json(output / 'provenance.json', base, exclusive=True)
    requested = list(VARIANTS) if variants is None else variants
    if set(requested) - set(VARIANTS):
        raise ValueError('Unknown experiment variant')
    requested = [variant for variant in VARIANTS if variant in requested]
    write_json(output / 'status.json', dict(status='running', requested=requested))
    start = time.monotonic()
    results = {}
    stage = 'B0'
    try:
        set_seed(seed)
        if device.startswith('cuda'):
            torch.cuda.reset_peak_memory_stats(torch.device(device))
        baseline = FrozenDAPO(config, config['baseline_checkpoint'], config['defect_names'], device)
        base['baseline_checkpoint_epoch'] = getattr(baseline, 'checkpoint_epoch', None)
        base['hardware'] = torch.cuda.get_device_name(torch.device(device)) if device.startswith('cuda') else platform.processor() or 'CPU'
        datasets = {role: ManifestDataset(manifest['rows'], config['source_root'], baseline,
                                         config['label_ids'], config['supervision_mode'])
                    for role, manifest in manifests.items()}
        inventory = dataset_inventory([row for m in manifests.values() for row in m['rows']], config['source_root'])
        write_json(output / 'source-data-inventory.json', inventory)
        base['dataset_sha256'] = inventory['sha256']
        write_json(output / 'provenance.json', base)
        validation = datasets['source_val']
        # Inspect and export exact runtime shapes before training.
        first = baseline(validation[0]['image'][None].to(device))
        again = baseline(validation[0]['image'][None].to(device))
        for key in ('z0', 'baseline_logits', 'image_logits'):
            torch.testing.assert_close(first[key], again[key], rtol=0, atol=0)
        bank = make_bank(baseline, config, seed)
        torch.testing.assert_close(expert_logits(baseline, bank, first)[..., 1:, :],
                                   first['baseline_logits'].unsqueeze(-2).expand(-1, -1, -1, config['experts'], -1),
                                   atol=1e-6, rtol=1e-6)
        write_json(output / 'inspection.json', dict(
            h=list(first['h'].shape), z0=list(first['z0'].shape), text=list(first['text'].shape),
            baseline_logits=list(first['baseline_logits'].shape),
            baseline_checkpoint=config['baseline_checkpoint'], baseline_sha256=base['baseline_sha256'],
            deterministic_repeat=True, zero_residual_parity=True,
            objective='Upstream typed Focal + normal/anomaly Dice, weight 4; frozen CLS CE is constant',
            model_location='moe/dapo.py:FrozenDAPO; upstream DAPO/adapter.py and DAPO/open_clip',
            scope='Runtime wrapper checks; full upstream evaluator parity requires the existing reproduction artifacts'))
        initial = evaluate(baseline, None, None, validation, 'B0', device, config['threshold_grid'],
                           scratch=config.get('scratch_dir'), export=output / 'predictions/B0')
        baseline_selection = choose(initial, None, config['regression_tolerance'])
        if baseline_selection is None:
            raise ValueError('Source validation has no supported defect class; revise grouped splits before training')
        results['B0'] = save_result(output, 'B0', base, baseline_selection, initial,
                                    epoch=getattr(baseline, 'checkpoint_epoch', None), delta=0.,
                                    resources=dict(frozen_baseline_parameters=sum(p.numel() for p in baseline.parameters()),
                                                   total_adapter_parameters=0, active_adapter_parameters=0,
                                                   executed_adapter_parameters=0, router_parameters=0,
                                                   flops=None, flops_reason='Not profiled'))
        write_json(output / 'baseline-lock.json', baseline_selection)
        objective = DAPOObjective()
        stage = 'expert_bank'
        train_experts(baseline, bank, datasets['expert_train'], config, seed, output, objective, device)
        bank_path = output / 'expert_bank.pt'
        torch.save(dict(state=bank.state_dict(), config=config, baseline_sha256=base['baseline_sha256'],
                        initialization_seeds=bank.initialization_seeds,
                        feature_dimensions=[baseline.input_dim, baseline.output_dim],
                        supervision_mode=config['supervision_mode'], vocabulary=baseline.names), bank_path)
        bank_hash = sha256(bank_path)
        baseline_primary = baseline_selection['primary']
        bests = {'B0': baseline_selection}
        for variant in ('B2', 'O1'):
            stage = variant
            evaluation = evaluate(baseline, bank, None, validation, variant, device, config['threshold_grid'],
                                  scratch=config.get('scratch_dir'), export=output / f'predictions/{variant}')
            picked = choose(evaluation, baseline_primary, config['regression_tolerance'])
            bests[variant] = picked
            results[variant] = save_result(output, variant, base, picked, evaluation,
                status='diagnostic' if variant == 'O1' else ('completed' if picked else 'no_qualifying_model'), bank_hash=bank_hash,
                resources=dict(frozen_baseline_parameters=sum(p.numel() for p in baseline.parameters()),
                               total_adapter_parameters=sum(p.numel() for p in bank.parameters()),
                               active_adapter_parameters=sum(p.numel() for p in bank.parameters()) // (config['experts'] if variant == 'O1' else 1),
                               executed_adapter_parameters=sum(p.numel() for p in bank.parameters()),
                               router_parameters=0, flops=None, flops_reason='Not profiled; all experts are computed'))
            if variant == 'O1':
                complementarity = evaluation['complementarity']
        write_json(output / 'complementarity.json', complementarity)
        gate = complementarity['oracle_gain'] > config['min_oracle_gain']
        for variant in requested:
            stage = variant
            if variant in ('B0', 'B2', 'O1'):
                continue
            if variant != 'B1' and not gate:
                results[variant] = save_result(output, variant, base, None, None, status='stopped',
                    reason='Expert oracle CE gain does not exceed preregistered complementarity gate', bank_hash=bank_hash)
                continue
            if variant == 'P2':
                controls = [bests.get(key) for key in ('P1', 'B2', 'B3')]
                reason = None
                if not config['robust_views_source_validated']:
                    reason = 'Photometric views have not been validated on source defects'
                elif any(value is None for value in controls) or not (
                    controls[0]['primary']['macro_defect_f1'] > max(v['primary']['macro_defect_f1'] for v in controls[1:])):
                    reason = 'P1 has not beaten qualified B2 and B3 on source validation'
                if reason:
                    results[variant] = save_result(output, variant, base, None, None, status='stopped', reason=reason, bank_hash=bank_hash)
                    continue
            active_bank, router = bank, None
            if variant == 'B1':
                active_bank = make_bank(baseline, config, seed + 10000, single=True)
                single_output = output / 'single_adapter'
                single_output.mkdir()
                train_experts(baseline, active_bank, datasets['expert_train'], config, seed,
                              single_output, objective, device)
                single_path = output / 'single_adapter.pt'
                torch.save(dict(state=active_bank.state_dict(), baseline_sha256=base['baseline_sha256']), single_path)
                active_hash = sha256(single_path)
                epochs = 1
            else:
                # Same initialization seed for all router controls; heads differ
                # only by baseline output in task-loss controls.
                set_seed(seed + 20000)
                router = make_router(baseline, config, variant)
                optimizer = torch.optim.Adam(router.parameters(), lr=config['router_lr'])
                generator = torch.Generator().manual_seed(seed + 30000)
                active_hash, epochs = bank_hash, config['router_epochs']
            donors_train = shuffled_donors(len(datasets['router_train']), seed) if variant == 'P1-shuffle' else None
            donors_val = shuffled_donors(len(validation), seed) if variant == 'P1-shuffle' else None
            if donors_train is not None:
                write_json(output / 'semantic_shuffle.json', dict(seed=seed, training=donors_train, validation=donors_val))
            best, best_evaluation, best_epoch, best_delta = None, None, None, None
            history = []
            checkpoint_path = output / f'{variant}.pt'
            for epoch in range(epochs):
                train_loss = None if router is None else train_router_epoch(
                    baseline, active_bank, router, datasets['router_train'], config, variant,
                    optimizer, objective, device, generator, donors_train)
                deltas = [0.] if variant in TASK_VARIANTS or variant in ('B1', 'P1-no-fallback') else config['delta_grid']
                for delta in deltas:
                    evaluation = evaluate(baseline, active_bank, router, validation, variant, device,
                                          config['threshold_grid'], delta, config.get('scratch_dir'), donors=donors_val)
                    picked = choose(evaluation, baseline_primary, config['regression_tolerance'])
                    history.append(dict(epoch=epoch, delta=delta, training_loss=train_loss,
                                        candidates=evaluation['primary'], qualified=picked is not None))
                    if picked is not None and (best is None or picked['primary']['macro_defect_f1'] > best['primary']['macro_defect_f1']):
                        reported_epoch = config['expert_epochs'] - 1 if variant == 'B1' else epoch
                        best, best_evaluation, best_epoch, best_delta = picked, evaluation, reported_epoch, delta
                        torch.save(dict(router=None if router is None else router.state_dict(), variant=variant,
                                        epoch=reported_epoch, delta=delta, threshold=best['primary']['threshold'],
                                        baseline_sha256=base['baseline_sha256'], expert_bank_sha256=active_hash,
                                        config=config, seed=seed, supervision_mode=config['supervision_mode'],
                                        vocabulary=baseline.names, feature_dimensions=[baseline.input_dim, baseline.output_dim]), checkpoint_path)
            write_json(output / f'{variant}-epochs.json', history)
            bests[variant] = best
            if best is not None:
                checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
                if router is not None:
                    router.load_state_dict(checkpoint['router'])
                best_evaluation = evaluate(baseline, active_bank, router, validation, variant, device,
                    [best['primary']['threshold']], best_delta, config.get('scratch_dir'),
                    output / f'predictions/{variant}', donors_val)
            params = sum(p.numel() for p in active_bank.parameters())
            resources = dict(total_adapter_parameters=params,
                             frozen_baseline_parameters=sum(p.numel() for p in baseline.parameters()),
                             active_adapter_parameters=params if variant == 'B1' else params // config['experts'],
                             executed_adapter_parameters=params,
                             router_parameters=0 if router is None else sum(p.numel() for p in router.parameters()),
                             flops=None, flops_reason='Not profiled; all experts are currently computed before selection')
            results[variant] = save_result(output, variant, base, best, best_evaluation,
                status='completed' if best else 'no_qualifying_model', epoch=best_epoch, delta=best_delta,
                reason=None if best else 'No source-selected epoch/threshold satisfies the worst-class constraint',
                bank_hash=active_hash, router_path=str(checkpoint_path) if best else None, resources=resources)
        matched = {}
        if bests.get('P1') is not None:
            p1_record = results['P1']
            reference_coverage = p1_record['routing']['fallback_coverage']
            for ablation in ('P1-no-context', 'P1-shuffle', 'P1-no-fallback'):
                if bests.get(ablation) is None:
                    continue
                # No-fallback's attainable coverage is zero: compare it with
                # P1 calibrated to zero, not a fictitious thresholded ablation.
                candidate_variant = 'P1' if ablation == 'P1-no-fallback' else ablation
                desired = 0. if ablation == 'P1-no-fallback' else reference_coverage
                checkpoint = torch.load(output / f'{candidate_variant}.pt', map_location=device, weights_only=False)
                router = make_router(baseline, config, candidate_variant)
                router.load_state_dict(checkpoint['router'])
                donors = shuffled_donors(len(validation), seed) if candidate_variant == 'P1-shuffle' else None
                calibration = evaluate(baseline, bank, router, validation, candidate_variant, device,
                    [checkpoint['threshold']], checkpoint['delta'], config.get('scratch_dir'), donors=donors,
                    coverage_target=desired)['coverage_calibration']
                evaluation = evaluate(baseline, bank, router, validation, candidate_variant, device,
                    [checkpoint['threshold']], calibration['delta'], config.get('scratch_dir'), donors=donors)
                matched[ablation] = dict(evaluated_variant=candidate_variant, calibration=calibration,
                    metrics=evaluation['primary'][0], routing=evaluation['routing'],
                    reference_variant=ablation if ablation == 'P1-no-fallback' else 'P1',
                    reference_metrics=results[ablation if ablation == 'P1-no-fallback' else 'P1']['metrics'])
        write_json(output / 'matched-coverage.json', matched)
        write_json(output / 'study.json', dict(results=results, complementarity=complementarity,
                   runtime_seconds=time.monotonic() - start,
                   peak_device_memory_bytes=torch.cuda.max_memory_allocated(torch.device(device)) if device.startswith('cuda') else None,
                   target_evaluation='not run', decision='Continue source investigation' if gate else 'NO GO: insufficient expert complementarity'))
        write_json(output / 'status.json', dict(status='completed', target_evaluation='not run'))
        return results
    except Exception as error:
        results[stage] = save_result(output, stage, base, None, None, status='failed',
                                    reason=f'{type(error).__name__}: {error}')
        write_json(output / 'study.json', dict(results=results, status='failed',
                                              target_evaluation='not run'))
        write_json(output / 'status.json', dict(status='failed', error=f'{type(error).__name__}: {error}',
                                               runtime_seconds=time.monotonic() - start))
        raise


def evaluate_target(study, variant, dataset_name, data_root, output, device='cuda', manifest_path=None,
                    baseline_result=None):
    """Use source-locked weights/thresholds; never optimize on target metrics."""
    from .data import ManifestDataset
    from .protocol import DAPO_DEFECT_NAMES
    if variant in ('O1',):
        raise ValueError('The label-informed oracle is a source diagnostic, not a target deployment')
    if variant != 'B0' and baseline_result is None:
        raise ValueError('Evaluate B0 once, then supply its target result via --baseline-result for paired comparisons')
    study, output = Path(study), Path(output)
    config = read_json(study / 'config.json')
    lock = read_json(study / f'{variant}.json')
    if lock['metrics'] is None:
        raise ValueError('No qualifying source-selected model exists for this variant')
    if sha256(config['baseline_checkpoint']) != lock['baseline_sha256']:
        raise ValueError('The frozen baseline checkpoint differs from source training')
    protocol = read_json(Path(__file__).parent / 'configs/dapo_multitype.json').get(dataset_name)
    if protocol is None:
        raise ValueError('Target evaluation requires an upstream-supported multi-type dataset')
    # Names are selected before softmax independently for each task/product.
    binary_names = list(DAPO_DEFECT_NAMES[dataset_name])
    names = list(dict.fromkeys(binary_names + protocol['defect_names']))
    if manifest_path is None:
        rows = records_from_metadata(Path(data_root) / 'meta.json')
    else:
        manifest = read_json(manifest_path)
        if manifest['role'] != 'target_test' or manifest['domain'] != 'target':
            raise ValueError('Target evaluation requires a target_test manifest')
        rows = manifest['rows']
    source_manifests = validate_splits(config['splits'])
    if Path(data_root).resolve() == Path(config['source_root']).resolve():
        source_ids = {r['image_id'] for m in source_manifests.values() for r in m['rows']}
        if source_ids & {r['image_id'] for r in rows}:
            raise ValueError('Target evaluation overlaps source training/selection images')
        source_groups = {r['group_id'] for m in source_manifests.values() for r in m['rows']}
        if source_groups & {r.get('group_id') for r in rows}:
            raise ValueError('Target evaluation overlaps source instance/acquisition groups')
    label_ids = {alias: names.index(canonical) + 1 for alias, canonical in protocol['label_aliases'].items()
                 if canonical in names}
    for canonical in names:
        label_ids[canonical] = names.index(canonical) + 1
    for row in rows:
        if row['product'] not in protocol['products']:
            raise ValueError(f'Unknown target product: {row["product"]}')
        if row['anomaly'] and row['defect'] not in label_ids:
            matches = {value for key, value in label_ids.items() if key.casefold() == row['defect'].casefold()}
            if len(matches) != 1:
                raise ValueError(f'Unmapped target type: {row["defect"]}')
            label_ids[row['defect']] = matches.pop()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'status.json', dict(status='running', stage='locked_target_evaluation'))
    start = time.monotonic()
    try:
        set_seed(lock['seed'])
        baseline = FrozenDAPO(config, config['baseline_checkpoint'], names, device)
        dataset = ManifestDataset(rows, data_root, baseline, label_ids, 'binary_mask_with_image_type')
        bank, router = None, None
        if variant != 'B0':
            bank = make_bank(baseline, config, lock['seed'] + 10000 if variant == 'B1' else lock['seed'], single=variant == 'B1')
            bank_path = study / ('single_adapter.pt' if variant == 'B1' else 'expert_bank.pt')
            if sha256(bank_path) != lock['expert_bank_sha256']:
                raise ValueError('Expert bank differs from the source-selected artifact')
            bank.load_state_dict(torch.load(bank_path, map_location=device, weights_only=False)['state'])
            bank.requires_grad_(False).eval()
        if variant not in ('B0', 'B1', 'B2'):
            if sha256(study / f'{variant}.pt') != lock['router_sha256']:
                raise ValueError('Router checkpoint differs from the source-selected artifact')
            checkpoint = torch.load(study / f'{variant}.pt', map_location=device, weights_only=False)
            for key in ('baseline_sha256', 'expert_bank_sha256'):
                if checkpoint[key] != lock[key]:
                    raise ValueError(f'Checkpoint provenance mismatch: {key}')
            router = make_router(baseline, config, variant)
            router.load_state_dict(checkpoint['router'])
        donors = shuffled_donors(len(dataset), lock['seed']) if variant == 'P1-shuffle' else None
        evaluation = evaluate(baseline, bank, router, dataset, variant, device,
            [lock['metrics']['threshold']], lock['delta'] or 0., config.get('scratch_dir'),
            output / 'predictions', donors, protocol, binary_names)
        inventory = dataset_inventory(rows, data_root)
        write_json(output / 'target-data-inventory.json', inventory)
        record = dict(lock, split='target_test', dataset=dataset_name, status='completed',
                      vocabulary=names, source_vocabulary=lock['vocabulary'],
                      training_supervision_mode=lock['supervision_mode'], supervision_mode='binary_mask_with_image_type',
                      metrics=evaluation['primary'][0], per_class=evaluation['primary'][0]['per_class'],
                      comparison=None, routing=evaluation['routing'],
                      target_metadata_sha256=sha256(Path(data_root) / 'meta.json'),
                      target_manifest_sha256=None if manifest_path is None else sha256(manifest_path),
                      dataset_sha256=inventory['sha256'],
                      source_lock_sha256=sha256(study / f'{variant}.json'),
                      runtime_seconds=time.monotonic() - start,
                      evaluation_runtime_seconds=evaluation['evaluation_runtime_seconds'],
                      target_supervision='binary mask with image-level type; combined-mask limitation',
                      selection='Locked source epoch, delta, and global defect-probability threshold; no target tuning')
        record['artifact_paths'] = dict(source_study=str(study), router=lock['artifact_paths']['router'],
                                        predictions=str(output / 'predictions'), metrics=str(output / 'evaluation.json'),
                                        per_class=str(output / 'per-class.csv'))
        if baseline_result is not None:
            paired = read_json(baseline_result)
            if paired['variant'] != 'B0':
                raise ValueError('The paired target reference must be B0')
            for key in ('seed', 'baseline_sha256', 'dataset', 'dataset_sha256', 'target_manifest_sha256'):
                if paired[key] != record[key]:
                    raise ValueError(f'Paired B0 target reference differs in {key}')
            record['comparison'] = compare_to_baseline(record['metrics'], paired['metrics'])
        write_json(output / 'result.json', record)
        write_json(output / 'evaluation.json', evaluation)
        export_per_class(output / 'per-class.csv', evaluation['primary'][0], record['comparison'])
        write_json(output / 'status.json', dict(status='completed'))
        return record
    except Exception as error:
        write_json(output / 'status.json', dict(status='failed', error=f'{type(error).__name__}: {error}'))
        raise
