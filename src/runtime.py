"""Project paths, JSON records and Windows runtime setup."""

import os
import sys
import importlib.metadata
import platform
import json
import ctypes
import threading
from functools import lru_cache
from pathlib import Path
from time import sleep


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(value):
    """Resolve a relative path from the repository root, preserving absolute paths."""
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def write_json(path, value):
    """Replace a JSON record only after its new contents are fully written."""
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    replace_file(temporary, path)


def replace_file(temporary, path):
    """Allow short Windows scanner locks when publishing a complete result file."""
    for attempt in range(4):
        try:
            temporary.replace(path)
            break
        except PermissionError:
            if attempt == 3:
                raise
            sleep(0.05)  # Windows scanners may briefly hold a newly written record.


def write_csv(path, frame):
    """Publish a CSV projection atomically using the same policy as manifests."""
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    frame.to_csv(temporary, index=False)
    replace_file(temporary, path)


def runtime_platform():
    """Read one native Windows version API instead of mixed cached/WMI sources."""
    if os.name != 'nt':
        return platform.platform()
    from ctypes import wintypes

    class WindowsVersion(ctypes.Structure):
        _fields_ = [
            ('size', wintypes.DWORD), ('major', wintypes.DWORD),
            ('minor', wintypes.DWORD), ('build', wintypes.DWORD),
            ('platform_id', wintypes.DWORD), ('service_pack', wintypes.WCHAR * 128),
        ]

    version = WindowsVersion()
    version.size = ctypes.sizeof(version)
    function = ctypes.WinDLL('ntdll').RtlGetVersion
    function.argtypes = [ctypes.POINTER(WindowsVersion)]
    function.restype = ctypes.c_long
    if function(ctypes.byref(version)) != 0:
        raise RuntimeError('Cannot read native Windows version for runtime validation.')
    return f'Windows-{version.major}.{version.minor}.{version.build}-{platform.machine()}'


def environment_metadata():
    """Record runtime and library versions for experiment compatibility checks."""
    versions = {}
    for name in (
        'numpy',
        'pandas',
        'scikit-learn',
        'tensorflow',
        'keras',
        'xgboost',
        'tables',
        'joblib',
        'PyYAML',
        'matplotlib',
        'optuna',
        'psutil',
        'nvidia-ml-py',
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {
        'versions': versions,
        'python': sys.version,
        'platform': runtime_platform(),
        'processor': platform.processor(),
        'logical_cpus': os.cpu_count(),
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'omp_num_threads': os.environ.get('OMP_NUM_THREADS'),
        'tensorflow_intraop_threads': os.environ.get('TF_NUM_INTRAOP_THREADS'),
        'tensorflow_interop_threads': os.environ.get('TF_NUM_INTEROP_THREADS'),
        'tensorflow_deterministic_ops': os.environ.get('TF_DETERMINISTIC_OPS'),
    }


class GPUError(RuntimeError):
    """A GPU/environment failure must stop the search, never trigger CPU fallback."""


def is_gpu_failure(error):
    """Recognize framework CUDA/OOM failures without swallowing ordinary trial errors."""
    return isinstance(error, (GPUError, MemoryError)) or any(
        marker in str(error).lower() for marker in (
            'cuda', 'cudnn', 'cublas', 'out of memory', 'resourceexhausted',
            'resource exhausted', 'device lost', 'no visible gpu',
        )
    )


@lru_cache(maxsize=8)
def cuda_driver(gpu_id):
    """Use the installed NVIDIA driver without requiring a shared cudart library."""
    try:
        library = ctypes.CDLL('nvcuda.dll' if os.name == 'nt' else 'libcuda.so.1')
    except OSError as error:
        raise GPUError('NVIDIA CUDA driver unavailable.') from error
    device = ctypes.c_int()
    context = ctypes.c_void_p()
    if library.cuInit(0) != 0 or library.cuDeviceGet(ctypes.byref(device), int(gpu_id)) != 0:
        raise GPUError(f'Cannot select CUDA-visible device {gpu_id}.')
    # CUDA Runtime frameworks use the primary context. Retain it for this process;
    # do not replace the framework's current context while inspecting the device.
    if library.cuDevicePrimaryCtxRetain(ctypes.byref(context), device) != 0:
        raise GPUError('Cannot retain the selected GPU primary context.')
    return library, device, context


def synchronize_gpu(gpu_id):
    """Synchronize the primary context, then restore the framework context stack."""
    library, _, context = cuda_driver(gpu_id)
    if library.cuCtxPushCurrent_v2(context) != 0:
        raise GPUError('Cannot activate GPU context for synchronization.')
    try:
        if library.cuCtxSynchronize() != 0:
            raise GPUError('CUDA device synchronization failed.')
    finally:
        previous = ctypes.c_void_p()
        if library.cuCtxPopCurrent_v2(ctypes.byref(previous)) != 0:
            raise GPUError('Cannot restore GPU context after synchronization.')


def gpu_identity(gpu_id):
    """Map the CUDA-visible ordinal to a physical PCI device, independent of NVML order."""
    buffer = ctypes.create_string_buffer(64)
    library, device, _ = cuda_driver(gpu_id)
    if library.cuDeviceGetPCIBusId(buffer, len(buffer), device) != 0:
        raise GPUError('Cannot resolve the selected CUDA device PCI bus ID.')
    record = {'cuda_ordinal': gpu_id, 'pci_bus_id': buffer.value.decode()}
    name = ctypes.create_string_buffer(256)
    if library.cuDeviceGetName(name, len(name), device) == 0:
        record['name'] = name.value.decode()
    try:
        import pynvml as nvml

        nvml.nvmlInit()
        try:
            handle = nvml.nvmlDeviceGetHandleByPciBusId(record['pci_bus_id'])
            for key, value in (
                ('uuid', nvml.nvmlDeviceGetUUID(handle)),
                ('name', nvml.nvmlDeviceGetName(handle)),
                ('driver', nvml.nvmlSystemGetDriverVersion()),
            ):
                record[key] = value.decode() if isinstance(value, bytes) else value
        finally:
            nvml.nvmlShutdown()
    except Exception as error:
        record['nvml_reason'] = str(error)
    return record


class ResourceMonitor:
    """Sample current-process RSS and GPU memory only during the measured fit."""

    def __init__(self, device, interval):
        self.device = device
        self.interval = interval
        self.stop = threading.Event()
        self.values = {
            'peak_cpu_ram_mb': None, 'baseline_cpu_ram_mb': None,
            'peak_gpu_vram_mb': None, 'baseline_gpu_vram_mb': None,
            'gpu_memory_scope': None, 'gpu_memory_reason': None,
            'cpu_memory_reason': None, 'sample_interval_s': interval,
            'memory_samples': 0,
        }
        self.nvml = None
        self.initialized_nvml = None
        self.process = None

    def _gpu_memory(self):
        nvml = self.nvml
        if self.values['gpu_memory_scope'] != 'device':
            try:
                processes = nvml.nvmlDeviceGetComputeRunningProcesses(self.handle)
                current = [p.usedGpuMemory for p in processes if p.pid == os.getpid()]
                # No context yet means zero; WDDM's sentinel means unavailable.
                if any(value is None or value >= 2**63 for value in current):
                    raise ValueError('Per-process GPU memory unavailable (WDDM or driver).')
                self.values['gpu_memory_scope'] = 'process'
                return sum(current) / 1024**2
            except Exception as error:
                self.values['gpu_memory_scope'] = 'device'
                self.values['gpu_memory_reason'] = str(error)
                # Never mix process and device peaks or baselines.
                self.values['peak_gpu_vram_mb'] = None
                self.values['baseline_gpu_vram_mb'] = self.device_baseline
        return nvml.nvmlDeviceGetMemoryInfo(self.handle).used / 1024**2

    def sample(self):
        if self.process is not None:
            try:
                value = self.process.memory_info().rss / 1024**2
                if self.values['baseline_cpu_ram_mb'] is None:
                    self.values['baseline_cpu_ram_mb'] = value
                self.values['peak_cpu_ram_mb'] = max(
                    self.values['peak_cpu_ram_mb'] or 0, value
                )
            except Exception as error:
                self.values['cpu_memory_reason'] = str(error)
        if self.nvml is not None:
            try:
                value = self._gpu_memory()
                if self.values['baseline_gpu_vram_mb'] is None:
                    self.values['baseline_gpu_vram_mb'] = value
                self.values['peak_gpu_vram_mb'] = max(
                    self.values['peak_gpu_vram_mb'] or 0, value
                )
            except Exception as error:
                self.values.update(peak_gpu_vram_mb=None, baseline_gpu_vram_mb=None,
                                   gpu_memory_reason=str(error))
                self.nvml = None
        self.values['memory_samples'] += 1

    def __enter__(self):
        try:
            import psutil

            self.process = psutil.Process()
        except Exception as error:
            self.values['cpu_memory_reason'] = str(error)
        try:
            import pynvml as nvml

            nvml.nvmlInit()
            self.initialized_nvml = nvml
            self.nvml = nvml
            self.handle = nvml.nvmlDeviceGetHandleByPciBusId(self.device['pci_bus_id'])
            self.device_baseline = nvml.nvmlDeviceGetMemoryInfo(self.handle).used / 1024**2
        except Exception as error:
            self.values['gpu_memory_reason'] = str(error)
            self.nvml = None
        self.sample()
        self.thread = threading.Thread(target=self._poll, daemon=True)
        self.thread.start()
        return self

    def _poll(self):
        while not self.stop.wait(self.interval):
            self.sample()

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()
        self.sample()
        if self.initialized_nvml is not None:
            try:
                self.initialized_nvml.nvmlShutdown()
            except Exception as error:
                self.values['gpu_memory_reason'] = str(error)


def configure_runtime(headless=True):
    """Set default plotting, TensorFlow and Windows library settings.

    Call before importing plotting or model libraries. Explicit environment
    settings take precedence over these defaults.
    """
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
            [str(p) for p in paths if p.is_dir() and str(p) not in current] + current
        )
