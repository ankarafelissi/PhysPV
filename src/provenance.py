"""Input-path, partition and runtime metadata for reproducible runs."""
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path

from .paths import project_path


def write_json(path, value):
    """Atomically write a JSON checkpoint."""
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def input_identity(config):
    return {'path': str(project_path(config['path']).resolve())}


def partition_metadata(splits):
    return {key: {'n_origins': len(part['origins']),
                  'first_origin': str(part['origins'][0]), 'last_origin': str(part['origins'][-1]),
                  'target_start': str(part['target_times'][0, 0]),
                  'target_end': str(part['target_times'][-1, -1])}
            for key, part in splits.items()}


def environment_metadata():
    versions = {}
    for name in ('numpy', 'pandas', 'scikit-learn', 'tensorflow', 'keras', 'xgboost',
                 'tables', 'joblib', 'PyYAML', 'matplotlib'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {'versions': versions, 'python': sys.version, 'platform': platform.platform(),
            'processor': platform.processor(), 'logical_cpus': os.cpu_count(),
            'tensorflow_deterministic_ops': os.environ.get('TF_DETERMINISTIC_OPS')}
