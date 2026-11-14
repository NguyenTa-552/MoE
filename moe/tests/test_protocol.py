from pathlib import Path
import tempfile
import unittest

import numpy as np

from moe.evaluation import primary_metrics, compare_to_baseline
from moe.experiment import choose
from moe.io import write_json
from moe.splits import create_splits, validate_splits
from moe.training import bootstrap_indices, shuffled_donors


class ProtocolTests(unittest.TestCase):
    def test_group_splits_disjoint_and_hash_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [dict(img_path=f'{i}.png', anomaly=i % 2, specie_name='scratch' if i % 2 else 'good') for i in range(18)]
            write_json(root / 'meta.json', {'test': {'object': rows}})
            groups = {f'{i}.png': f'group-{i // 2}' for i in range(18)}
            write_json(root / 'groups.json', groups)
            create_splits(root / 'meta.json', root / 'groups.json', root / 'splits')
            manifests = validate_splits(root / 'splits')
            self.assertEqual(sum(len(m['rows']) for m in manifests.values()), 18)
            with self.assertRaises(FileExistsError):
                create_splits(root / 'meta.json', root / 'groups.json', root / 'splits')
            path = root / 'splits/router_train.json'
            path.write_text(path.read_text() + '\n')
            with self.assertRaisesRegex(ValueError, 'hash'):
                validate_splits(root / 'splits')

    def test_missing_acquisition_groups_are_not_replaced_with_image_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_json(root / 'meta.json', {'object': [dict(img_path='a.png', anomaly=0)]})
            write_json(root / 'groups.json', {})
            with self.assertRaisesRegex(ValueError, 'Missing real'):
                create_splits(root / 'meta.json', root / 'groups.json', root / 'splits')

    def test_bootstraps_and_semantic_shuffle_reproducible(self):
        rows = [dict(product='object', defect='good' if i < 9 else 'scratch') for i in range(10)]
        indices = bootstrap_indices(rows, 2)
        self.assertEqual(indices, bootstrap_indices(rows, 2))
        self.assertEqual(len(indices), 10)
        self.assertEqual(sum(i == 9 for i in indices), 5)
        donors = shuffled_donors(9, 3)
        self.assertEqual(donors, shuffled_donors(9, 3))
        self.assertTrue(all(k != v for k, v in donors.items()))
        self.assertEqual(set(donors.values()), set(range(9)))

    def test_primary_excludes_normal_and_absent_classes(self):
        products = {'object': dict(truth=np.array([0, 1, 1]),
            scores=np.array([[.9, .1, .0], [.1, .9, .0], [.2, .8, .0]]),
            labels=['normal', 'scratch', 'rust'])}
        result = primary_metrics(products, .5)
        self.assertEqual(result['macro_defect_f1'], 1.)
        self.assertIsNone(result['per_class']['object/rust']['f1'])
        self.assertNotIn('object/normal', result['per_class'])

    def test_worst_class_constraint_rejects_macro_improvement_with_regression(self):
        baseline = dict(macro_defect_f1=.6, per_class={'a': {'f1': .5}, 'b': {'f1': .7}})
        candidate = dict(macro_defect_f1=.7, per_class={'a': {'f1': .9}, 'b': {'f1': .5}})
        self.assertIsNone(choose({'primary': [candidate]}, baseline, .01))
        self.assertAlmostEqual(compare_to_baseline(candidate, baseline)['worst_class_delta'], -.2)

    def test_defect_holdout_removes_synonyms_and_entire_related_groups(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [dict(img_path=f'{i}.png', anomaly=int(i == 0),
                         specie_name='scratches' if i == 0 else 'good') for i in range(10)]
            write_json(root / 'meta.json', {'object': rows})
            write_json(root / 'groups.json', {f'{i}.png': str(i // 2) for i in range(10)})
            summary = create_splits(root / 'meta.json', root / 'groups.json', root / 'splits',
                                    held_out_defects=['scratch'], aliases={'scratches': 'scratch'})
            manifests = validate_splits(root / 'splits')
            source_ids = {r['image_id'] for m in manifests.values() for r in m['rows']}
            self.assertNotIn('0.png', source_ids)
            self.assertNotIn('1.png', source_ids)
            self.assertEqual(summary['target_test']['images'], 2)
            self.assertIn('scratches', summary['forbidden_source_names'])
            self.assertIn('scratch', summary['forbidden_source_names'])

    def test_report_preserves_undefined_metrics_instead_of_zero(self):
        from moe.report import aggregate
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            study = root / 'target'
            write_json(study / 'status.json', {'status': 'completed'})
            write_json(study / 'result.json', dict(variant='B0', split='target_test', dataset='mpdd',
                seed=0, status='completed', metrics={'macro_defect_f1': None}, comparison=None))
            summary = aggregate([study], root / 'report')
            self.assertIsNone(summary['B0']['mean'])
            self.assertEqual(summary['B0']['measured'], 0)


if __name__ == '__main__':
    unittest.main()
