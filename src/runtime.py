"""Project paths, JSON records and Windows runtime setup."""
import os
import sys
import importlib.metadata
import platform
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def write_json(path, value):
    """Replace a JSON record only after its new contents are fully written."""
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def environment_metadata():
    versions = {}
    for name in ('numpy', 'pandas', 'scikit-learn', 'tensorflow', 'keras', 'xgboost',
                 'tables', 'joblib', 'PyYAML', 'matplotlib', 'optuna'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {'versions': versions, 'python': sys.version, 'platform': platform.platform(),
            'processor': platform.processor(), 'logical_cpus': os.cpu_count(),
            'tensorflow_intraop_threads': os.environ.get('TF_NUM_INTRAOP_THREADS'),
            'tensorflow_interop_threads': os.environ.get('TF_NUM_INTEROP_THREADS'),
            'tensorflow_deterministic_ops': os.environ.get('TF_DETERMINISTIC_OPS')}


def configure_runtime(headless=True):
    if headless:
        os.environ.setdefault('MPLBACKEND', 'Agg')
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    os.environ.setdefault('TF_DETERMINISTIC_OPS', '1')
    os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '4')
    os.environ.setdefault('TF_NUM_INTEROP_THREADS', '1')
    if os.name == 'nt':
        prefix = Path(sys.prefix)
        paths = [prefix / 'Library' / 'bin', prefix / 'Scripts', prefix]
        current = os.environ.get('PATH', '').split(os.pathsep)
        os.environ['PATH'] = os.pathsep.join(
            [str(p) for p in paths if p.is_dir() and str(p) not in current] + current)
