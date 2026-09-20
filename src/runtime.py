"""Prepare native libraries before importing scientific packages on Windows."""
import os
import sys
from pathlib import Path


def configure_runtime(headless=True):
    if headless:
        os.environ.setdefault('MPLBACKEND', 'Agg')
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    if os.name == 'nt':
        prefix = Path(sys.prefix)
        paths = [prefix / 'Library' / 'bin', prefix / 'Scripts', prefix]
        current = os.environ.get('PATH', '').split(os.pathsep)
        os.environ['PATH'] = os.pathsep.join(
            [str(p) for p in paths if p.is_dir() and str(p) not in current] + current)
