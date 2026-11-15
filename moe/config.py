"""Strict configuration validation for the initial dense study."""

from pathlib import Path
import math
from .io import read_json
from .splits import validate_splits


def validate_source_assets(config):
    """Check actual source files and labels without importing CLIP/PyTorch."""
    manifests = validate_splits(config['splits'])
    root = Path(config['source_root'])
    for role, manifest in manifests.items():
        for row in manifest['rows']:
            paths = [row['img_path']]
            if row['anomaly']:
                paths.append(row['mask_path'])
                if config['supervision_mode'] == 'binary_mask_with_image_type':
                    channel = config['label_ids'].get(row['defect'])
                    if channel is None or channel <= 0:
                        raise ValueError(f'{role}: missing positive label ID for {row["defect"]!r}')
            for relative in paths:
                if not (root / relative).is_file():
                    raise FileNotFoundError(f'{role}: missing source asset {root / relative}')
    return manifests


def load_config(path):
    config = read_json(path)
    for field in ('dapo_root', 'baseline_checkpoint', 'source_root', 'splits'):
        if not config.get(field):
            raise ValueError(f'Set {field} in a copied experiment configuration')
        if not Path(config[field]).exists():
            raise FileNotFoundError(config[field])
    if not Path(config['baseline_checkpoint']).is_file():
        raise ValueError('baseline_checkpoint must be a file')
    if not (Path(config['dapo_root']) / 'open_clip/__init__.py').is_file():
        raise ValueError('dapo_root must contain the upstream vendored OpenCLIP')
    if config['supervision_mode'] not in ('typed_mask', 'binary_mask_with_image_type'):
        raise ValueError('Dense study requires verified typed masks or explicitly limited binary masks with image types')
    if config['supervision_mode'] == 'binary_mask_with_image_type' and config.get('single_type_mask_verified') is not True:
        raise ValueError('Audit that every positive binary-mask pixel belongs to its image type, then explicitly set single_type_mask_verified=true; image labels alone are insufficient')
    names = config['defect_names']
    if not names or any(not isinstance(name, str) or not name.strip() for name in names) or len(set(names)) != len(names):
        raise ValueError('Source vocabulary must contain unique defect names')
    summary = read_json(Path(config['splits']) / 'summary.json')
    forbidden = {name.casefold() for name in summary.get('forbidden_source_names', [])}
    if forbidden & {name.casefold() for name in names}:
        raise ValueError('Remove held-out defect names/synonyms from the source vocabulary and remap label IDs')
    if any(not isinstance(v, int) or v < 0 or v > len(names) for v in config['label_ids'].values()):
        raise ValueError('Label IDs must index normal + source defect names')
    if config['experts'] != 3:
        raise ValueError('The initial controlled study uses exactly three experts')
    if config['model']['temperature'] != 0.07:
        raise ValueError('Changing DAPO temperature requires an isolated ablation')
    for field in ('expert_epochs', 'router_epochs', 'batch_size', 'rank', 'hidden_dim'):
        if type(config[field]) is not int or config[field] < 1:
            raise ValueError(f'{field} must be a positive integer')
    for field in ('expert_lr', 'router_lr'):
        if not math.isfinite(config[field]) or config[field] <= 0:
            raise ValueError(f'{field} must be positive and finite')
    if not math.isfinite(config['alpha']) or config['alpha'] < 0:
        raise ValueError('alpha must be finite and nonnegative')
    if not math.isfinite(config['regression_tolerance']) or not 0 <= config['regression_tolerance'] <= 1:
        raise ValueError('regression_tolerance must be in [0, 1]')
    if not math.isfinite(config['min_oracle_gain']) or (config['min_oracle_gain'] < 0 and not config.get('synthetic_smoke')):
        raise ValueError('min_oracle_gain must be finite and nonnegative for real studies')
    if config.get('scratch_dir') is not None and not Path(config['scratch_dir']).is_dir():
        raise ValueError('Create scratch_dir before submitting this job')
    if not config['delta_grid'] or not config['threshold_grid']:
        raise ValueError('Preregister finite source-selection grids')
    if any(not math.isfinite(x) for x in config['delta_grid'] + config['threshold_grid']):
        raise ValueError('Selection grids must be finite')
    if any(not 0 < x < 1 for x in config['threshold_grid']):
        raise ValueError('Probability thresholds must be strictly between zero and one')
    factors = config['robust_brightness_factors']
    if len(factors) != 2 or any(not 0.9 <= factor <= 1.1 or factor == 1 for factor in factors):
        raise ValueError('P2 requires exactly two mild nonidentity brightness factors in [0.9, 1.1]')
    validate_source_assets(config)
    return config
