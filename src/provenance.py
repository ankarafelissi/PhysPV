"""Hashes and partition metadata linking predictions to exact inputs and code."""
import hashlib
import importlib.metadata
from pathlib import Path

from .paths import project_path


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def input_identity(config):
    return {'path': str(project_path(config['path']).resolve()),
            'sha256': file_hash(project_path(config['path']))}


def partition_metadata(splits):
    return {key: {'n_origins': len(part['origins']),
                  'first_origin': str(part['origins'][0]), 'last_origin': str(part['origins'][-1]),
                  'target_start': str(part['target_times'][0, 0]),
                  'target_end': str(part['target_times'][-1, -1]),
                  'origins_sha256': hashlib.sha256(part['origins'].asi8.tobytes()).hexdigest()}
            for key, part in splits.items()}


def environment_metadata():
    versions = {}
    for name in ('numpy', 'pandas', 'scikit-learn', 'tensorflow', 'xgboost'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {'versions': versions, 'source_sha256': {
        str(path.relative_to(project_path('.'))).replace('\\', '/'): file_hash(path)
        for path in sorted(project_path('src').rglob('*.py'))}}
