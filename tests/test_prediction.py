"""Regression checks for the independent saved-model evaluation entry point."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from src.runtime import configure_runtime
configure_runtime()
import joblib
import numpy as np
import yaml

from src.config import data_contract, load_config
from src.data import load_data, prepare_data
from src.features import build_features
from src.predict import run
from src.provenance import input_identity, partition_metadata, environment_metadata
from src.scenarios import fit_thresholds


class PredictionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = load_config('config/config.yaml')
        self.config.update(model='XGBoost', output_dir=str(self.root / 'outputs'))
        frame = build_features(load_data(self.config['data']), self.config['data'])
        self.splits, scalers = prepare_data(frame, self.config['data'])
        (self.root / 'config.yaml').write_text(yaml.safe_dump(self.config), encoding='utf-8')
        (self.root / 'metadata.json').write_text(json.dumps({
            'model': 'XGBoost', 'data_contract': data_contract(self.config['data']),
            'input': input_identity(self.config['data']), 'partitions': partition_metadata(self.splits),
            'scenario_thresholds': fit_thresholds(self.splits['TRAIN']),
            'run_id': 'fixture', 'seed': self.config['seed'],
            'environment': environment_metadata(), 'smoke': True}), encoding='utf-8')
        joblib.dump(scalers, self.root / 'scalers.joblib')

    def test_prediction_does_not_train_or_fit_scalers(self):
        model = Mock()
        model.predict.return_value = np.zeros_like(self.splits['TEST']['Y_scaled'])
        with patch('src.models.load_model', return_value=model), \
                patch('src.models.train_model', side_effect=AssertionError('Unexpected training')), \
                patch('sklearn.preprocessing.MinMaxScaler.fit', side_effect=AssertionError('Unexpected scaler fit')):
            result = run(self.root)
        self.assertEqual(result['predicted'].shape, self.splits['TEST']['Y'].shape)
        self.assertTrue((self.root / 'outputs/results' / (result['run_id'] + '.csv')).exists())
        self.assertEqual({p.name for p in (self.root / 'outputs').iterdir()}, {'models', 'figures', 'results'})

    def test_incompatible_contract_is_rejected_before_model_loading(self):
        config = copy.deepcopy(self.config)
        config['data']['features'].reverse()
        with patch('src.models.load_model') as loader:
            with self.assertRaisesRegex(ValueError, 'incompatible'):
                run(self.root, config)
            loader.assert_not_called()


if __name__ == '__main__':
    unittest.main()
