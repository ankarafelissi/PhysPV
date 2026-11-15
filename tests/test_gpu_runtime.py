"""GPU protocol and monitoring tests with simulated drivers; no model fitting."""

import os
import io
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
from contextlib import redirect_stdout

from src.models import XGBoostForecaster, check_gpu, configure_gpu, fit_measurement
import numpy as np
from src.runtime import GPUError, ResourceMonitor, environment_metadata, synchronize_gpu


class GPURuntimeTests(unittest.TestCase):
    @unittest.skipUnless(os.name == 'nt', 'Windows runtime version regression')
    def test_windows_metadata_ignores_conflicting_platform_display_builds(self):
        with patch('src.runtime.platform.platform', return_value='Windows-10-10.0.26200-SP0'):
            parent = environment_metadata()
        with patch('src.runtime.platform.platform', return_value='Windows-10-10.0.26100-SP0'):
            child = environment_metadata()
        self.assertEqual(parent, child)
        self.assertTrue(parent['platform'].startswith('Windows-'))

    def test_horizon_progress_uses_mock_estimators_without_training(self):
        x = np.ones((4, 2, 1))
        y = np.ones((4, 2))
        backend = SimpleNamespace(XGBRegressor=Mock(side_effect=lambda **kwargs: Mock()))
        output = io.StringIO()
        with patch.dict(sys.modules, {'xgboost': backend}), redirect_stdout(output):
            model = XGBoostForecaster().fit(x, y, x, y, progress=True)
        self.assertEqual(len(model.estimators), 2)
        self.assertIn('1/2 horizons (50%)', output.getvalue())
        self.assertIn('2/2 horizons (100%)', output.getvalue())
    def test_xgboost_device_setup_does_not_import_tensorflow(self):
        with (patch.dict(sys.modules, {'tensorflow': None}),
              patch('src.models.gpu_identity', return_value={'cuda_ordinal': 0}) as identity):
            self.assertEqual(configure_gpu('XGBoost', 0), {'cuda_ordinal': 0})
        identity.assert_called_once_with(0)

    def test_driver_sync_restores_context_even_when_computation_fails(self):
        driver = Mock()
        driver.cuCtxPushCurrent_v2.return_value = 0
        driver.cuCtxSynchronize.return_value = 1
        driver.cuCtxPopCurrent_v2.return_value = 0
        with patch('src.runtime.cuda_driver', return_value=(driver, 0, None)):
            with self.assertRaisesRegex(GPUError, 'synchronization failed'):
                synchronize_gpu(0)
        driver.cuCtxPopCurrent_v2.assert_called_once()

    def nvml(self, processes):
        nvml = Mock()
        nvml.nvmlDeviceGetComputeRunningProcesses.side_effect = processes
        nvml.nvmlDeviceGetMemoryInfo.side_effect = [
            SimpleNamespace(used=value * 1024**2) for value in (300, 320, 350)
        ]
        return nvml

    def test_missing_nvml_records_null_and_reason(self):
        with patch.dict(sys.modules, {'pynvml': None}):
            with ResourceMonitor({'pci_bus_id': 'fixture'}, 100) as monitor:
                pass
        self.assertIsNone(monitor.values['peak_gpu_vram_mb'])
        self.assertIsNone(monitor.values['gpu_memory_scope'])
        self.assertTrue(monitor.values['gpu_memory_reason'])
        self.assertGreater(monitor.values['peak_cpu_ram_mb'], 0)

    def test_wddm_falls_back_to_device_and_preserves_pre_fit_baseline(self):
        sentinel = SimpleNamespace(pid=os.getpid(), usedGpuMemory=2**64 - 1)
        nvml = self.nvml([[sentinel]])
        with patch.dict(sys.modules, {'pynvml': nvml}):
            with ResourceMonitor({'pci_bus_id': 'fixture'}, 100) as monitor:
                pass
        self.assertEqual(monitor.values['gpu_memory_scope'], 'device')
        self.assertEqual(monitor.values['baseline_gpu_vram_mb'], 300)
        self.assertEqual(monitor.values['peak_gpu_vram_mb'], 350)
        self.assertTrue(monitor.values['gpu_memory_reason'])
        nvml.nvmlShutdown.assert_called_once()

    def test_process_scope_excludes_other_processes(self):
        own = SimpleNamespace(pid=os.getpid(), usedGpuMemory=100 * 1024**2)
        other = SimpleNamespace(pid=-1, usedGpuMemory=500 * 1024**2)
        nvml = self.nvml([[own, other], [own, other]])
        with patch.dict(sys.modules, {'pynvml': nvml}):
            with ResourceMonitor({'pci_bus_id': 'fixture'}, 100) as monitor:
                pass
        self.assertEqual(monitor.values['gpu_memory_scope'], 'process')
        self.assertEqual(monitor.values['peak_gpu_vram_mb'], 100)
        self.assertEqual(monitor.values['baseline_gpu_vram_mb'], 100)

    def test_failed_device_query_records_missing_instead_of_zero(self):
        nvml = self.nvml([])
        nvml.nvmlDeviceGetMemoryInfo.side_effect = RuntimeError('Driver unavailable')
        with patch.dict(sys.modules, {'pynvml': nvml}):
            with ResourceMonitor({'pci_bus_id': 'fixture'}, 100) as monitor:
                pass
        self.assertIsNone(monitor.values['peak_gpu_vram_mb'])
        self.assertIn('Driver unavailable', monitor.values['gpu_memory_reason'])
        nvml.nvmlShutdown.assert_called_once()

    def test_no_tensorflow_gpu_fails_before_probe_fit(self):
        tf = Mock()
        tf.config.list_physical_devices.return_value = []
        with patch.dict(sys.modules, {'tensorflow': tf}):
            with self.assertRaisesRegex(GPUError, 'unavailable'):
                check_gpu('CNN_LSTM', {}, 0)
        tf.keras.Sequential.assert_not_called()

    def test_fit_timer_synchronizes_both_boundaries_without_training(self):
        with (patch('src.models.synchronize_gpu') as sync,
              patch('src.models.perf_counter', side_effect=[1., 6.])):
            with fit_measurement(0, None) as timing:
                pass
        self.assertEqual(timing['fit_time_s'], 5.)
        self.assertEqual(sync.call_count, 2)


if __name__ == '__main__':
    unittest.main()
