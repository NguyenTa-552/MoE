from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import numpy as np

from moe.blocks.experts import ExpertBank
from moe.blocks.router import GainRouter
from moe.blocks.targets import gain_targets
from moe.io import read_json
from moe.smoke import run_smoke, SyntheticDAPO, SyntheticObjective
from moe.training import train_router_epoch


class WorkflowTests(unittest.TestCase):
    def test_source_and_locked_target_smoke_and_tampered_router_rejection(self):
        from moe.experiment import evaluate_target
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'smoke'
            result = run_smoke(output)
            self.assertFalse(result['is_research_result'])
            self.assertEqual(result['variants']['P2'], 'stopped')
            paired = read_json(output / 'target-P1/result.json')
            self.assertIsNotNone(paired['comparison']['worst_class_delta'])
            self.assertTrue((output / 'study/matched-coverage.json').is_file())
            self.assertTrue(any((output / 'study/predictions/P1').glob('*-example.png')))
            # Recompute the primary metric independently from persisted arrays.
            from moe.evaluation import primary_metrics
            truth, scores = [], []
            for prediction in sorted((output / 'study/predictions/P1').glob('*.npz')):
                with np.load(prediction) as saved:
                    truth.append(saved['truth'])
                    scores.append(saved['scores'])
            source = read_json(output / 'study/P1.json')['metrics']
            recomputed = primary_metrics({'synthetic': dict(truth=np.stack(truth), scores=np.stack(scores),
                labels=['normal', 'scratch', 'rust'])}, source['threshold'])
            self.assertEqual(source, recomputed)
            path = output / 'study/P1.pt'
            with path.open('ab') as handle:
                handle.write(b'tampered')
            with patch('moe.experiment.FrozenDAPO', SyntheticDAPO):
                with self.assertRaisesRegex(ValueError, 'Router checkpoint differs'):
                    evaluate_target(output / 'study', 'P1', 'mpdd', output / 'target-images',
                        output / 'tampered-target', 'cpu', baseline_result=output / 'target-B0/result.json')
            self.assertEqual(read_json(output / 'tampered-target/status.json')['status'], 'failed')

    def test_robust_router_uses_two_aligned_views_and_frozen_bank(self):
        class Dataset(torch.utils.data.Dataset):
            def __len__(self):
                return 2

            def __getitem__(self, index):
                image = torch.full((3, 8, 8), 0.2 + index * 0.1)
                mask = torch.zeros(8, 8, dtype=torch.long)
                mask[:2, :2] = index + 1
                return dict(image=image, mask=mask, training_mask=mask, index=index)
        baseline = SyntheticDAPO({}, None, ['scratch', 'rust'])
        bank = ExpertBank(6, 4, 2).requires_grad_(False)
        router = GainRouter(6, 4)
        optimizer = torch.optim.Adam(router.parameters(), lr=.001)
        config = dict(batch_size=2, robust_brightness_factors=[.95, 1.05])
        with patch('moe.training.gain_targets', wraps=gain_targets) as observed:
            loss = train_router_epoch(baseline, bank, router, Dataset(), config, 'P2', optimizer,
                                      SyntheticObjective(), 'cpu', torch.Generator().manual_seed(0))
        self.assertTrue(torch.isfinite(torch.tensor(loss)))
        self.assertEqual(len(observed.call_args.args[1]), 2)
        self.assertTrue(all(p.grad is None for p in bank.parameters()))
        self.assertTrue(all(p.grad is None for p in baseline.parameters()))
        self.assertTrue(any(p.grad is not None for p in router.parameters()))


if __name__ == '__main__':
    unittest.main()
