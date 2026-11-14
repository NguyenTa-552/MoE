"""Strict configuration validation for the initial dense study."""

from pathlib import Path
import math
from .io import read_json


def load_config(path):
    config = read_json(path)
    for field in ('dapo_root', 'baseline_checkpoint', 'source_root', 'splits'):
        if not config.get(field):
            raise ValueError(f'Set {field} in a copied experiment configuration')
        if not Path(config[field]).exists():
            raise FileNotFoundError(config[field])
    if config['supervision_mode'] not in ('typed_mask', 'binary_mask_with_image_type'):
        raise ValueError('Dense study requires verified typed masks or explicitly limited binary masks with image types')
    if config['supervision_mode'] == 'binary_mask_with_image_type' and not config.get('single_type_mask_verified'):
        raise ValueError('Audit that every positive binary-mask pixel belongs to its image type, then explicitly set single_type_mask_verified=true; image labels alone are insufficient')
    names = config['defect_names']
    if not names or len(set(names)) != len(names):
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
        if config[field] < 1:
            raise ValueError(f'{field} must be positive')
    if not config['delta_grid'] or not config['threshold_grid']:
        raise ValueError('Preregister finite source-selection grids')
    if any(not math.isfinite(x) for x in config['delta_grid'] + config['threshold_grid']):
        raise ValueError('Selection grids must be finite')
    if any(not 0 < x < 1 for x in config['threshold_grid']):
        raise ValueError('Probability thresholds must be strictly between zero and one')
    factors = config['robust_brightness_factors']
    if len(factors) != 2 or any(not 0.9 <= factor <= 1.1 or factor == 1 for factor in factors):
        raise ValueError('P2 requires exactly two mild nonidentity brightness factors in [0.9, 1.1]')
    return config
