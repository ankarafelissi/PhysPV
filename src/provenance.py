"""Input-path, partition and runtime metadata for reproducible runs."""
import importlib.metadata
import json
import os
import platform
import sys
import subprocess
from pathlib import Path

from .paths import project_path


def source_record():
    """Record readable source and Git state without cryptographic identities."""
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=project_path('.'), text=True)
    files = [project_path('main.py'), project_path('requirements.txt')]
    files += sorted(project_path('src').rglob('*.py'))
    files += sorted(project_path('config').glob('*.yaml'))
    return {'commit': git('rev-parse', 'HEAD').strip(),
            'dirty': bool(git('status', '--porcelain').strip()),
            'diff': git('diff', 'HEAD', '--', 'src', 'config', 'main.py', 'requirements.txt'),
            'files': {str(p.relative_to(project_path('.'))): p.read_text(encoding='utf-8')
                      for p in files},
            'command': subprocess.list2cmdline([sys.executable, *sys.argv])}


def validate_source(saved):
    current = source_record()
    if current['commit'] != saved['commit'] or current['files'] != saved['files']:
        raise ValueError('Source or Git commit changed; create a new study instead of resuming.')


def write_json(path, value):
    """Atomically write a JSON checkpoint."""
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def input_identity(config):
    path = project_path(config['path']).resolve()
    info = path.stat()
    return {'path': str(path), 'size_bytes': info.st_size, 'modified_ns': info.st_mtime_ns}


def partition_metadata(splits):
    return {key: {'n_origins': len(part['origins']),
                  'first_origin': str(part['origins'][0]), 'last_origin': str(part['origins'][-1]),
                  'target_start': str(part['target_times'][0, 0]),
                  'target_end': str(part['target_times'][-1, -1])}
            for key, part in splits.items()}


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
            'tensorflow_deterministic_ops': os.environ.get('TF_DETERMINISTIC_OPS')}
