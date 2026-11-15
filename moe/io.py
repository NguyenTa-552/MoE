"""Reproducibility helpers without ML dependencies."""

import hashlib
import json
from pathlib import Path
import subprocess


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x' if exclusive else 'w') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


def git_commit():
    result = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def dataset_inventory(rows, root):
    """Hash actual image/mask files, not just metadata paths."""
    root = Path(root)
    paths = set()
    for row in rows:
        paths.add(row['img_path'])
        if row['anomaly']:
            paths.add(row['mask_path'])
    files = {name: sha256(root / name) for name in sorted(paths)}
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return dict(root=str(root.resolve()), sha256=digest, files=files)


def implementation_inventory():
    root = Path(__file__).resolve().parent
    files = {p.relative_to(root).as_posix(): sha256(p) for p in sorted(root.rglob('*'))
             if p.is_file() and (p.suffix in ('.py', '.sh') or p.name.startswith('requirements')
                                 or p == root / 'configs/dapo_multitype.json')}
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return dict(sha256=digest, files=files)


def verify_source_image_disjointness(manifests, inventory):
    seen = {}
    for role, manifest in manifests.items():
        for row in manifest['rows']:
            digest = inventory['files'][row['img_path']]
            if digest in seen and seen[digest][0] != role:
                previous_role, previous_image = seen[digest]
                raise ValueError(f'Identical image content crosses source roles: {previous_role}/{previous_image} and {role}/{row["img_path"]}; keep duplicates in one group')
            seen[digest] = (role, row['img_path'])
