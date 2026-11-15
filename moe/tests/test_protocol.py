from pathlib import Path
import tempfile
import unittest

import numpy as np

from moe.evaluation import primary_metrics, compare_to_baseline
from moe.experiment import choose
from moe.io import write_json, dataset_inventory, verify_source_image_disjointness
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

    def test_metadata_only_split_stratifies_labels_without_manual_groups(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [dict(img_path=f'bottle/{label}/{i}.png', anomaly=int(label != 'good'),
                         specie_name=label) for label in ('good', 'broken_large') for i in range(10)]
            write_json(root / 'meta.json', {'test': {'bottle': rows}})
            summary = create_splits(root / 'meta.json', output=root / 'splits', seed=7)
            manifests = validate_splits(root / 'splits')
            self.assertEqual(summary['grouping'], 'image')
            self.assertTrue(summary['image_independence_assumed'])
            self.assertIsNone(summary['groups_sha256'])
            self.assertEqual(summary['incomplete_strata'], {})
            for role, count in zip(('expert_train', 'router_train', 'source_val'), (12, 4, 4)):
                self.assertEqual(len(manifests[role]['rows']), count)
                self.assertEqual(summary['roles'][role]['labels']['good'], count // 2)
                self.assertEqual(summary['roles'][role]['labels']['broken_large'], count // 2)
            create_splits(root / 'meta.json', output=root / 'again', seed=7)
            for role in manifests:
                self.assertEqual((root / 'splits' / f'{role}.json').read_bytes(),
                                 (root / 'again' / f'{role}.json').read_bytes())

    def test_rare_species_do_not_get_duplicated_to_fill_missing_roles(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [dict(img_path=f'{label}/{i}.png', anomaly=int(label != 'good'), specie_name=label)
                    for label, count in [('good', 10), ('rare', 2), ('singleton', 1)] for i in range(count)]
            write_json(root / 'meta.json', {'bottle': rows})
            summary = create_splits(root / 'meta.json', output=root / 'splits')
            validate_splits(root / 'splits')
            self.assertEqual(summary['incomplete_strata']['bottle/rare'],
                             {'expert_train': 1, 'router_train': 0, 'source_val': 1})
            self.assertEqual(summary['incomplete_strata']['bottle/singleton'],
                             {'expert_train': 1, 'router_train': 0, 'source_val': 0})
            self.assertEqual(sum(r['images'] for r in summary['roles'].values()), len(rows))

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

    def test_ignored_pixels_do_not_count_as_false_positive_defects(self):
        from moe.metrics import multitype_metrics
        scores = np.array([[.1, .9], [.1, .9]])
        truth = np.array([1, -1])
        primary = primary_metrics({'bottle': dict(truth=truth, scores=scores, labels=['normal', 'broken'])}, .5)
        self.assertEqual(primary['macro_defect_f1'], 1.)
        legacy, classes = multitype_metrics(truth, scores, ['normal', 'broken'])
        self.assertEqual(legacy['f1'], 1.)
        self.assertEqual(classes['broken']['support'], 1)
        ignored, _ = multitype_metrics(np.array([-1, -1]), scores, ['normal', 'broken'])
        self.assertIsNone(ignored['f1'])

    def test_identical_content_cannot_cross_source_roles(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'one.png').write_bytes(b'identical image bytes')
            (root / 'copy.png').write_bytes(b'identical image bytes')
            first = dict(img_path='one.png', anomaly=0)
            second = dict(img_path='copy.png', anomaly=0)
            inventory = dataset_inventory([first, second], root)
            with self.assertRaisesRegex(ValueError, 'Identical image content crosses source roles'):
                verify_source_image_disjointness({'expert_train': {'rows': [first]},
                    'router_train': {'rows': [second]}}, inventory)

    def test_preflight_checks_source_images_masks_and_label_mapping(self):
        from moe.config import validate_source_assets
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = root / 'images'
            images.mkdir()
            rows = [dict(img_path=f'{i}.png', mask_path=f'{i}-mask.png',
                         anomaly=1, specie_name='broken') for i in range(6)]
            for row in rows:
                (images / row['img_path']).write_bytes(b'image')
                (images / row['mask_path']).write_bytes(b'mask')
            write_json(root / 'meta.json', {'bottle': rows})
            create_splits(root / 'meta.json', output=root / 'splits')
            config = dict(splits=str(root / 'splits'), source_root=str(images),
                          supervision_mode='binary_mask_with_image_type', label_ids={'broken': 1})
            self.assertEqual(sum(len(m['rows']) for m in validate_source_assets(config).values()), 6)
            config['label_ids'] = {}
            with self.assertRaisesRegex(ValueError, 'missing positive label ID'):
                validate_source_assets(config)
            config['label_ids'] = {'broken': 1}
            (images / rows[0]['mask_path']).unlink()
            with self.assertRaisesRegex(FileNotFoundError, 'missing source asset'):
                validate_source_assets(config)

    def test_report_rejects_different_baselines_for_matched_seeds(self):
        from moe.report import aggregate
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for seed in (0, 1):
                directory = root / str(seed)
                write_json(directory / 'status.json', {'status': 'completed'})
                write_json(directory / 'result.json', dict(variant='B0', split='target_test', dataset='mpdd',
                    seed=seed, status='completed', metrics={'macro_defect_f1': .5}, comparison=None,
                    baseline_sha256=f'different-baseline-{seed}'))
            with self.assertRaisesRegex(ValueError, 'different configs, splits, baseline'):
                aggregate([root / '0', root / '1'], root / 'report')

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
