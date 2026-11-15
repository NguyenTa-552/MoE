"""Explicitly synthetic stand-ins exercise orchestration without CLIP weights."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

from .blocks.scoring import BaselineScorer, dense_maps
from .io import read_json, write_json
from .splits import create_splits


class SyntheticDAPO(nn.Module):
    def __init__(self, config, checkpoint, names, device='cpu'):
        super().__init__()
        from torchvision import transforms
        self.input_dim, self.output_dim, self.layers = 6, 4, [1, 2]
        self.names = list(names)
        self.projection = nn.Linear(6, 4, bias=False)
        self.register_buffer('text', F.normalize(torch.arange(1, (len(names) + 1) * 4 + 1).float().reshape(-1, 4), dim=-1))
        self.scorer = BaselineScorer()
        self.preprocess = transforms.ToTensor()
        self.mask_transform = transforms.ToTensor()
        self.to(device).requires_grad_(False).eval()

    @torch.no_grad()
    def forward(self, images):
        patches = F.adaptive_avg_pool2d(images, 2).flatten(2).transpose(1, 2)
        h = torch.cat((patches, patches.square()), -1)
        h = torch.stack((h, h * 0.9))
        z0 = self.projection(h)
        image = z0.mean((0, 2))
        binary = torch.cat((self.text[:1], F.normalize(self.text[1:].mean(0, keepdim=True), dim=-1)))
        return dict(h=h, z0=z0, text=self.text, image=image,
                    image_logits=self.scorer(image, binary), baseline_logits=self.scorer(z0, self.text))


class SyntheticObjective:
    def __call__(self, probabilities, mask):
        maps = dense_maps(probabilities, mask.shape[-2:])
        target = F.one_hot(mask, probabilities.shape[-1]).permute(0, 3, 1, 2)
        return -(maps.clamp_min(1e-8).log() * target[None]).sum(2).mean()


def run_smoke(output):
    from .experiment import run_source, evaluate_target
    from .report import aggregate
    torch.set_num_threads(1)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    root = output / 'images'
    root.mkdir()
    rows, groups = [], {}
    for group in range(6):
        for label in range(3):
            name = f'g{group}-c{label}'
            rgb = np.full((8, 8, 3), 30 + group * 5, dtype=np.uint8)
            mask = np.zeros((8, 8), dtype=np.uint8)
            if label:
                rgb[:4, :4, label - 1] = 230
                mask[:4, :4] = 255
            Image.fromarray(rgb).save(root / f'{name}.png')
            Image.fromarray(mask).save(root / f'{name}-mask.png')
            rows.append(dict(img_path=f'{name}.png', mask_path=f'{name}-mask.png',
                             anomaly=int(label > 0), cls_name='synthetic',
                             specie_name=['good', 'scratch', 'rust'][label]))
            groups[f'{name}.png'] = f'instance-{group}'
    write_json(root / 'meta.json', {'test': {'synthetic': rows}})
    write_json(output / 'groups.json', groups)
    create_splits(root / 'meta.json', output / 'groups.json', output / 'splits')
    checkpoint = output / 'synthetic-baseline.pt'
    torch.save({'synthetic_only': True}, checkpoint)
    config = read_json(Path(__file__).parent / 'configs/experiment.example.json')
    config.update(dapo_root=str(Path('DAPO').resolve()), baseline_checkpoint=str(checkpoint),
                  source_root=str(root), splits=str(output / 'splits'),
                  defect_names=['scratch', 'rust'], label_ids={'good': 0, 'scratch': 1, 'rust': 2},
                  expert_epochs=1, router_epochs=1, batch_size=3, rank=2, hidden_dim=8,
                  threshold_grid=[0.3, 0.6], delta_grid=[0.], synthetic_smoke=True,
                  single_type_mask_verified=True,
                  min_oracle_gain=-1., regression_tolerance=1., robust_views_source_validated=False)
    # The relaxed gate/tolerance are confined to synthetic orchestration checks.
    write_json(output / 'smoke-config.json', config)
    with patch('moe.experiment.FrozenDAPO', SyntheticDAPO), patch('moe.experiment.DAPOObjective', SyntheticObjective):
        records = run_source(output / 'smoke-config.json', output / 'study', seed=0, device='cpu')
        target = output / 'target-images'
        target.mkdir()
        target_rows = []
        for i, label in enumerate(('good', 'scratches', 'major_rust')):
            original = root / f'g0-c{i}.png'
            target_pixels = np.array(Image.open(original)).copy()
            target_pixels[-1, -1] = 99 + i
            Image.fromarray(target_pixels).save(target / f'{i}.png')
            Image.open(root / f'g0-c{i}-mask.png').save(target / f'{i}-mask.png')
            target_rows.append(dict(img_path=f'{i}.png', mask_path=f'{i}-mask.png',
                                    anomaly=int(i > 0), cls_name='metal_plate', specie_name=label))
        write_json(target / 'meta.json', {'test': {'metal_plate': target_rows}})
        evaluate_target(output / 'study', 'B0', 'mpdd', target, output / 'target-B0', 'cpu')
        evaluate_target(output / 'study', 'P1', 'mpdd', target, output / 'target-P1', 'cpu',
                        baseline_result=output / 'target-B0/result.json')
    aggregate([output / 'study'], output / 'report')
    if not any((output / 'study').glob('P1*.pt')):
        raise AssertionError('Synthetic orchestration did not save a qualifying router')
    result = dict(status='synthetic_smoke_passed', is_research_result=False,
                  stages='Grouped splits, expert training, router training, source selection, reload, locked synthetic target comparison, prediction export, report',
                  variants={key: value['status'] for key, value in records.items()},
                  full_DAPO_checkpoint_validation='not run; synthetic encoder and objective used')
    write_json(output / 'smoke-result.json', result)
    return result
