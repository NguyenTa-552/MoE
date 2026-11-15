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
from moe.metrics import multitype_metrics


class WorkflowTests(unittest.TestCase):
    def test_source_and_locked_target_smoke_and_tampered_router_rejection(self):
        from moe.experiment import evaluate_target
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'smoke'
            with patch('moe.evaluation.multitype_metrics', wraps=multitype_metrics) as typed_metrics:
                result = run_smoke(output)
            self.assertTrue(all(isinstance(call.args[1], np.memmap) for call in typed_metrics.call_args_list))
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
            # Metadata/config edits must not silently change a source-locked model.
            config_path = output / 'study/config.json'
            original_config = config_path.read_bytes()
            config_path.write_bytes(original_config + b'\n')
            with self.assertRaisesRegex(ValueError, 'Source configuration differs'):
                evaluate_target(output / 'study', 'B0', 'mpdd', output / 'target-images',
                                output / 'tampered-config', 'cpu')
            config_path.write_bytes(original_config)
            lock_path = output / 'study/P1.json'
            original_lock = lock_path.read_bytes()
            lock_path.write_bytes(original_lock + b'\n')
            with self.assertRaisesRegex(ValueError, 'Source selection lock differs'):
                evaluate_target(output / 'study', 'P1', 'mpdd', output / 'target-images',
                    output / 'tampered-lock', 'cpu', baseline_result=output / 'target-B0/result.json')
            lock_path.write_bytes(original_lock)
            from moe.splits import records_from_metadata
            from moe.io import write_json
            target_rows = records_from_metadata(output / 'target-images/meta.json')
            write_json(output / 'duplicate-target.json', dict(role='target_test', domain='target',
                                                              rows=[target_rows[0], target_rows[0]]))
            with self.assertRaisesRegex(ValueError, 'unique, nonempty image IDs'):
                evaluate_target(output / 'study', 'B0', 'mpdd', output / 'target-images',
                                output / 'duplicate-target', 'cpu', manifest_path=output / 'duplicate-target.json')
            duplicate_image = output / 'target-images/0.png'
            original_target = duplicate_image.read_bytes()
            duplicate_image.write_bytes((output / 'images/g0-c0.png').read_bytes())
            with self.assertRaisesRegex(ValueError, 'Target image content overlaps source'):
                evaluate_target(output / 'study', 'B0', 'mpdd', output / 'target-images',
                                output / 'overlapping-target', 'cpu')
            duplicate_image.write_bytes(original_target)
            path = output / 'study/P1.pt'
            with path.open('ab') as handle:
                handle.write(b'tampered')
            with patch('moe.experiment.FrozenDAPO', SyntheticDAPO):
                with self.assertRaisesRegex(ValueError, 'Router checkpoint differs'):
                    evaluate_target(output / 'study', 'P1', 'mpdd', output / 'target-images',
                        output / 'tampered-target', 'cpu', baseline_result=output / 'target-B0/result.json')
            self.assertEqual(read_json(output / 'tampered-target/status.json')['status'], 'failed')

    def test_missing_configuration_is_retained_as_failed_attempt(self):
        from moe.experiment import run_source
        from moe.report import aggregate
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(FileNotFoundError):
                run_source(root / 'missing.json', root / 'study', device='cpu')
            self.assertEqual(read_json(root / 'study/status.json')['stage'], 'preflight')
            report = aggregate([root / 'study'], root / 'report')
            self.assertEqual(report['preflight']['statuses'], ['failed'])

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
