"""Immutable source splits, requiring real acquisition/instance group IDs."""

from collections import Counter, defaultdict
from pathlib import Path
import random

from .io import read_json, sha256, write_json


ROLES = ('expert_train', 'router_train', 'source_val')


def records_from_metadata(path):
    metadata = read_json(path)
    partition = metadata.get('test', metadata)
    if not isinstance(partition, dict) or not all(isinstance(v, list) for v in partition.values()):
        raise ValueError('Expected a test partition or flat product-to-records metadata')
    records = []
    for product, rows in partition.items():
        for source in rows:
            row = dict(source)
            row['product'] = row.get('cls_name') or row.get('product_cls') or product
            row['defect'] = row.get('specie_name') or row.get('defect_cls')
            if not row['defect'] and row.get('anomaly') == 0:
                row['defect'] = 'good'
            if not row['defect'] or 'img_path' not in row or 'anomaly' not in row:
                raise ValueError('Missing image path, anomaly flag, or defect type')
            if row['product'] != product:
                raise ValueError('Metadata product disagrees with product partition')
            row['image_id'] = row['img_path']
            records.append(row)
    if len({r['image_id'] for r in records}) != len(records):
        raise ValueError('Duplicate image IDs in metadata')
    return records


def create_splits(metadata, groups_path, output, seed=0, ratios=(0.6, 0.2, 0.2),
                  held_out_products=(), held_out_defects=(), aliases=None):
    if len(ratios) != 3 or any(x <= 0 for x in ratios) or abs(sum(ratios) - 1) > 1e-9:
        raise ValueError('Three positive split fractions must sum to one')
    records, groups = records_from_metadata(metadata), read_json(groups_path)
    if held_out_products and held_out_defects:
        raise ValueError('Product-held-out and defect-held-out are separate protocols')
    if held_out_defects and aliases is None:
        raise ValueError('Defect-held-out splits require an explicit alias-to-canonical mapping')
    aliases = {} if aliases is None else aliases
    canonical = lambda name: aliases.get(name, name).casefold()
    excluded = {canonical(name) for name in held_out_defects}
    grouped = defaultdict(list)
    for row in records:
        group = groups.get(row['image_id'])
        if not isinstance(group, str) or not group.strip():
            raise ValueError(f'Missing real instance/acquisition group for {row["image_id"]}')
        row['group_id'] = group
        grouped[group].append(row)
    target_keys = {key for key, rows in grouped.items() if any(
        row['product'] in held_out_products or canonical(row['defect']) in excluded for row in rows)}
    if (held_out_products or held_out_defects) and not target_keys:
        raise ValueError('The requested held-out products/defects matched no records')
    target_rows = [row for key in sorted(target_keys) for row in grouped.pop(key)]
    if len(grouped) < 3:
        raise ValueError('At least three independent groups are required')
    keys = sorted(grouped)
    random.Random(seed).shuffle(keys)
    first = max(1, min(len(keys) - 2, round(len(keys) * ratios[0])))
    second = max(first + 1, min(len(keys) - 1, round(len(keys) * sum(ratios[:2]))))
    partitions = (keys[:first], keys[first:second], keys[second:])
    output = Path(output)
    if output.exists():
        raise FileExistsError('Split output already exists; manifests are immutable')
    output.mkdir(parents=True)
    summary = dict(seed=seed, metadata_sha256=sha256(metadata),
                   groups_sha256=sha256(groups_path), roles={},
                   protocol='product_held_out' if held_out_products else ('defect_held_out' if held_out_defects else 'cross_dataset'),
                   held_out_products=list(held_out_products), held_out_defects=sorted(excluded),
                   forbidden_source_names=sorted(set(held_out_defects) | excluded | {key for key, value in aliases.items() if canonical(value) in excluded}))
    for role, keys in zip(ROLES, partitions):
        rows = [row for key in keys for row in grouped[key]]
        path = output / f'{role}.json'
        write_json(path, dict(role=role, domain='source', rows=rows), exclusive=True)
        summary['roles'][role] = dict(images=len(rows), groups=len(keys), sha256=sha256(path),
                                      labels=dict(Counter(r['defect'] for r in rows)),
                                      products=dict(Counter(r['product'] for r in rows)))
    if target_rows:
        path = output / 'target_test.json'
        write_json(path, dict(role='target_test', domain='target', rows=target_rows,
                             protocol=summary['protocol']), exclusive=True)
        summary['target_test'] = dict(images=len(target_rows), groups=len(target_keys), sha256=sha256(path))
    write_json(output / 'summary.json', summary, exclusive=True)
    validate_splits(output)
    return summary


def validate_splits(directory):
    summary = read_json(Path(directory) / 'summary.json')
    images, groups = set(), set()
    manifests = {}
    for role in ROLES:
        path = Path(directory) / f'{role}.json'
        if sha256(path) != summary['roles'][role]['sha256']:
            raise ValueError(f'Split hash mismatch: {role}')
        manifest = read_json(path)
        if manifest['role'] != role or manifest['domain'] != 'source':
            raise ValueError('Training and selection require source-role manifests')
        ids = {row['image_id'] for row in manifest['rows']}
        group_ids = {row['group_id'] for row in manifest['rows']}
        if not ids or images & ids or groups & group_ids or len(ids) != len(manifest['rows']):
            raise ValueError('Empty, duplicate, or overlapping image/group split')
        images |= ids
        groups |= group_ids
        manifests[role] = manifest
    return manifests
