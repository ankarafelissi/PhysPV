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
from src.data import load_data, prepare_data, input_identity, partition_metadata
from src.data import build_features
from src.metrics import fit_scenario_thresholds
from src.predict import run
from src.runtime import environment_metadata


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
            'scenario_thresholds': fit_scenario_thresholds(self.splits['TRAIN']),
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
        result_dir = result['metadata_path'].parent
        self.assertEqual({p.name for p in result_dir.iterdir()},
                         {'predictions.csv', 'metrics.csv', 'raw_target_metrics.csv', 'metadata.json'})
        metadata = json.loads(result['metadata_path'].read_text(encoding='utf-8'))
        self.assertEqual(len(metadata['forecast_health']), 2 * self.config['data']['horizon'])
        self.assertEqual({p.name for p in (self.root / 'outputs').iterdir()}, {'models', 'figures', 'results'})

    def test_reusing_prediction_id_cannot_overwrite_results(self):
        model = Mock()
        model.predict.return_value = np.zeros_like(self.splits['TEST']['Y_scaled'])
        with patch('src.models.load_model', return_value=model) as loader:
            result = run(self.root, run_id='fixture')
            original = result['metadata_path'].read_bytes()
            with self.assertRaises(FileExistsError):
                run(self.root, run_id='fixture')
            self.assertEqual(loader.call_count, 1)
            self.assertEqual(result['metadata_path'].read_bytes(), original)

    def test_training_record_links_files_inside_its_study(self):
        from src.experiments import _train_record
        config = copy.deepcopy(self.config)
        config['smoke_study'] = True
        config['data']['processed_path'] = str(self.root / 'processed.h5')
        study_dir = self.root / 'outputs/results/study_fixture'
        model = Mock()
        with patch('src.train.train_model', return_value=(model, {})), \
                patch('src.train.save_model'), \
                patch('src.train.load_model', return_value=model), \
                patch('src.train.predict_scaled',
                      side_effect=lambda model, name, x: np.zeros((len(x), config['data']['horizon']))):
            record = _train_record(config, {'id': 'nonpi_reference', 'subset': []},
                                   'XGBoost', config['seed'], partition_metadata(self.splits),
                                   'train_fixture', study_dir)
        result_dir = study_dir / 'runs/train_fixture'
        self.assertEqual({p.name for p in result_dir.iterdir()},
                         {'train.log', 'history.json', 'validation_health.csv'})
        self.assertEqual(Path(record['history']).parent, result_dir)
        self.assertEqual(Path(record['validation_health']).parent, result_dir)
        self.assertTrue(np.isfinite(record['validation_mae']))

    def test_incompatible_contract_is_rejected_before_model_loading(self):
        config = copy.deepcopy(self.config)
        config['data']['features'].append('HUMIDITY[%]')
        with patch('src.models.load_model') as loader:
            with self.assertRaisesRegex(ValueError, 'incompatible'):
                run(self.root, config)
            loader.assert_not_called()


if __name__ == '__main__':
    unittest.main()
