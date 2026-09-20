"""Shared smoke configuration and repository-path regression checks."""
import copy
import os
from pathlib import Path
import tempfile
import unittest

from src.runtime import configure_runtime
configure_runtime()
from src.config import load_config, smoke_config, validate_study
from src.paths import PROJECT_ROOT, project_path


class ConfigurationTests(unittest.TestCase):
    def test_smoke_overrides_preserve_source_and_physics_contract(self):
        config = load_config('config/full.yaml')
        original = copy.deepcopy(config)
        smoke = smoke_config(config)
        self.assertEqual(config, original)
        self.assertEqual(smoke['data']['horizon'], 1)
        self.assertEqual(smoke['models']['CNN_LSTM']['epochs'], 1)
        self.assertEqual(smoke['data']['plant'], original['data']['plant'])
        self.assertEqual(smoke['data']['physics'], original['data']['physics'])
        self.assertTrue(smoke['smoke_study'])

    def test_paths_and_configuration_do_not_depend_on_working_directory(self):
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                self.assertEqual(project_path('config/config.yaml'), PROJECT_ROOT / 'config/config.yaml')
                self.assertEqual(project_path(directory), Path(directory))
                self.assertEqual(load_config('config/config.yaml')['data']['p_nom_kw'], 7.44)
            finally:
                os.chdir(previous)

    def test_invalid_seed_types_raise_contract_error(self):
        study = load_config('config/full.yaml')['study']
        study['seeds'] = [1, 2, 3, 4, []]
        with self.assertRaisesRegex(ValueError, 'five'):
            validate_study(study)
        with self.assertRaisesRegex(ValueError, 'mapping'):
            validate_study(None)
