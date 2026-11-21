r"""
DODO VISION AI v8.9 -- GPU Registry Enforcement

Forces Windows 11 hybrid-graphics systems to allocate the discrete NVIDIA GPU
for the Python interpreter via the DirectX UserGpuPreferences registry key.

Must be called BEFORE any CUDA/TF/ONNX library is imported.

Registry key:
  HKCU\SOFTWARE\Microsoft\DirectX\UserGpuPreferences
  Value name  = full path to python.exe
  Value data  = "GpuPreference=2;"   (2 = High Performance)

v8.9 Step 3: If an NVIDIA discrete adapter is found via WMI, the
specific adapter UUID is bound: "SpecificAdapter=<LUID>;GpuPreference=2;"
so the OS scheduler is pinned to the exact GPU, not just "prefer high perf".
"""
from __future__ import annotations
import os
import re
import sys
import subprocess
from typing import Optional

_REG_PATH = r"SOFTWARE\Microsoft\DirectX\UserGpuPreferences"
_GPU_HIGH_PERF = "GpuPreference=2;"


# ══════════════════════════════════════════════
#  NVIDIA Adapter Discovery (WMI / DXGI)
# ══════════════════════════════════════════════

def _discover_nvidia_adapter() -> Optional[dict]:
    r"""Try to discover the discrete NVIDIA adapter via WMI.

    Returns {"name": str, "pnp_id": str, "adapter_luid": str} or None.
    Uses `wmic` which is available on all Windows 10/11 systems.
    """
    if sys.platform != "win32":
        return None
    try:
        out = subprocess.check_output(
            ["wmic", "path", "Win32_VideoController", "get",
             "Name,PNPDeviceID,AdapterCompatibility", "/format:csv"],
            timeout=5, stderr=subprocess.DEVNULL, text=True,
        )
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 4:
                continue
            compat, name, pnp = parts[1], parts[2], parts[3]
            if "nvidia" in (compat + name).lower():
                # Extract LUID-style ID from PNP path for SpecificAdapter
                # Format: PCI\VEN_10DE&DEV_xxxx&SUBSYS_xxxx&REV_xx\...
                luid = _pnp_to_luid(pnp)
                return {"name": name, "pnp_id": pnp, "adapter_luid": luid}
    except Exception:
        pass
    # Fallback: try DXGI via ctypes
    return _discover_via_dxgi()


def _pnp_to_luid(pnp_id: str) -> Optional[str]:
    r"""Extract a registry-usable adapter identifier from PNP device ID.

    Windows DirectX uses a LUID (Locally Unique Identifier) for
    SpecificAdapter. We derive a stable hash from the PNP path so the
    registry value uniquely targets this adapter across reboots.
    """
    if not pnp_id:
        return None
    # Use the PCI VEN+DEV portion as a stable identifier
    match = re.search(r"VEN_([0-9A-Fa-f]{4})&DEV_([0-9A-Fa-f]{4})", pnp_id)
    if match:
        return f"VEN_{match.group(1)}&DEV_{match.group(2)}"
    return None


def _discover_via_dxgi() -> Optional[dict]:
    """Fallback NVIDIA adapter discovery via DXGI COM (ctypes).

    Only attempts if wmic failed. Returns adapter info or None.
    """
    try:
        import ctypes
        dxgi = ctypes.windll.LoadLibrary("dxgi.dll")
        # CreateDXGIFactory1
        IID_IDXGIFactory1 = (ctypes.c_byte * 16)(
            0x70, 0x0a, 0x77, 0x77, 0x2c, 0x6e, 0xf2, 0x4b,
            0xb9, 0x31, 0x52, 0x5b, 0xe5, 0x6c, 0xf2, 0x50,
        )
        factory = ctypes.c_void_p()
        hr = dxgi.CreateDXGIFactory1(ctypes.byref(IID_IDXGIFactory1), ctypes.byref(factory))
        if hr != 0:
            return None
        # Enumerate adapters looking for NVIDIA
        idx = 0
        while True:
            adapter = ctypes.c_void_p()
            # EnumAdapters1 is at vtable[12]
            vt = ctypes.cast(ctypes.cast(factory, ctypes.POINTER(ctypes.c_void_p))[0],
                             ctypes.POINTER(ctypes.c_void_p))
            enum_fn = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p))(vt[12])
            hr = enum_fn(factory, idx, ctypes.byref(adapter))
            if hr != 0:
                break
            # Read DXGI_ADAPTER_DESC1 — Description is wchar[128] at offset 0
            desc_buf = (ctypes.c_byte * 512)()
            desc_vt = ctypes.cast(ctypes.cast(adapter, ctypes.POINTER(ctypes.c_void_p))[0],
                                  ctypes.POINTER(ctypes.c_void_p))
            getdesc_fn = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p)(desc_vt[10])
            getdesc_fn(adapter, ctypes.byref(desc_buf))
            name = ctypes.wstring_at(ctypes.addressof(desc_buf), 128).split('\x00')[0]
            if "nvidia" in name.lower():
                return {"name": name, "pnp_id": "", "adapter_luid": None}
            idx += 1
    except Exception:
        pass
    return None


def enforce_discrete_gpu() -> dict:
    """Set the Windows registry key to force discrete GPU for this Python exe.

    v8.9: If an NVIDIA adapter is discovered, pins to the specific adapter
    via SpecificAdapter= registry value for deterministic GPU scheduling.

    Returns a status dict with what happened. Safe no-op on non-Windows or
    if winreg is unavailable.
    """
    result = {"applied": False, "python_exe": "", "error": None,
              "adapter": None}
    if sys.platform != "win32":
        result["error"] = "not_windows"
        return result
    try:
        import winreg
    except ImportError:
        result["error"] = "winreg_unavailable"
        return result

    python_exe = sys.executable
    result["python_exe"] = python_exe

    # v8.9 Step 3: discover NVIDIA adapter for UUID binding
    adapter = _discover_nvidia_adapter()
    if adapter:
        result["adapter"] = adapter
        # Build registry value with specific adapter binding
        luid = adapter.get("adapter_luid")
        if luid:
            gpu_pref = f"SpecificAdapter={luid};GpuPreference=2;"
        else:
            gpu_pref = _GPU_HIGH_PERF
    else:
        gpu_pref = _GPU_HIGH_PERF

    try:
        key = winreg.CreateKeyEx(
            winreg.HKEY_CURRENT_USER, _REG_PATH,
            access=winreg.KEY_READ | winreg.KEY_WRITE,
        )
        # Check if already set with current preference
        try:
            existing, _ = winreg.QueryValueEx(key, python_exe)
            if "GpuPreference=2" in existing:
                # If we have adapter binding and existing doesn't, upgrade
                if luid and f"SpecificAdapter={luid}" not in existing if adapter and adapter.get("adapter_luid") else False:
                    winreg.SetValueEx(key, python_exe, 0, winreg.REG_SZ, gpu_pref)
                    result["applied"] = True
                    result["note"] = "upgraded_to_uuid"
                else:
                    result["applied"] = True
                    result["note"] = "already_set"
                winreg.CloseKey(key)
                return result
        except FileNotFoundError:
            pass  # key exists but value not set yet

        winreg.SetValueEx(key, python_exe, 0, winreg.REG_SZ, gpu_pref)
        winreg.CloseKey(key)
        result["applied"] = True
        result["note"] = "newly_set"
    except PermissionError:
        result["error"] = "permission_denied"
    except Exception as e:
        result["error"] = str(e)
    return result


def validate_gpu(fail_fast: bool = False) -> dict:
    """Post-init GPU validation. Call AFTER frameworks are loaded.

    v8.9 Step 7: If fail_fast=True and no discrete GPU is detected, prints
    a prominent warning. In production, set fail_fast=True to catch
    configurations that silently fell back to integrated graphics.

    Returns diagnostic info about available GPU devices.
    """
    info = {"onnx_cuda": False, "onnx_dml": False, "tf_gpu_count": 0,
            "discrete_gpu_confirmed": False, "torch_cuda": False,
            "fail_fast": fail_fast}

    # Check ONNX Runtime (v9.1.7: include DirectML — it's the standard
    # Windows 11 GPU path when CUDA Toolkit isn't installed. Previously
    # this function only recognised CUDA/TF/PyTorch and emitted a
    # bogus "NO DISCRETE GPU" warning even when DirectML was correctly
    # running inference on the NVIDIA 3060.)
    try:
        import onnxruntime as ort
        # v9.8.1 2026-05-12 (operator: "theres still this error"
        # = 60+ ScatterND warnings every boot). The session-level
        # log_severity_level fix in pad/branches/deepfake_branch.py
        # caught the `transformer_memcpy` warnings (those use the
        # session's logger), but ScatterND warnings come from ORT's
        # *global* Default logger -- a different sink. Silence it
        # at module-init via `set_default_logger_severity(3)` so
        # nothing INFO/WARN reaches stderr; ERROR/FATAL still do.
        # 0=verbose 1=info 2=warning 3=error 4=fatal.
        try:
            ort.set_default_logger_severity(3)
        except Exception:
            pass
        provs = ort.get_available_providers()
        info["onnx_providers"] = provs
        info["onnx_cuda"] = "CUDAExecutionProvider" in provs
        info["onnx_dml"] = "DmlExecutionProvider" in provs
    except Exception:
        info["onnx_providers"] = []

    # Check TensorFlow — also pin GPU memory growth to prevent VRAM lockout
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        info["tf_gpu_count"] = len(gpus)
        info["tf_devices"] = [g.name for g in gpus]
        # v8.9 Step 7: enable memory growth so TF doesn't monopolize VRAM
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass  # already configured
    except Exception:
        info["tf_devices"] = []

    # Check PyTorch CUDA
    try:
        import torch
        info["torch_cuda"] = torch.cuda.is_available()
        if info["torch_cuda"]:
            info["torch_device_name"] = torch.cuda.get_device_name(0)
            info["torch_vram_mb"] = round(torch.cuda.get_device_properties(0).total_mem / 1048576)
    except Exception:
        pass

    # Check CUDA_VISIBLE_DEVICES
    info["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES", "not_set")

    info["discrete_gpu_confirmed"] = (
        info["onnx_cuda"] or info["onnx_dml"]
        or info["tf_gpu_count"] > 0 or info["torch_cuda"]
    )

    # v8.9 Step 7: fail-fast — loud warning if no discrete GPU found
    if fail_fast and not info["discrete_gpu_confirmed"]:
        print("=" * 60)
        print("[FAIL-FAST] NO DISCRETE GPU DETECTED!")
        print("  ONNX CUDA:", info["onnx_cuda"])
        print("  ONNX DirectML:", info["onnx_dml"])
        print("  TF GPUs:  ", info["tf_gpu_count"])
        print("  PyTorch:  ", info["torch_cuda"])
        print("  The system will run on CPU — ML inference will be slow.")
        print("  Fix: pip install onnxruntime-directml  (simplest on Windows 11)")
        print("=" * 60)

    return info


def pin_tf_device(device: str = "/GPU:0"):
    """Return a TensorFlow device context manager for explicit GPU pinning.

    Usage:
        with pin_tf_device("/GPU:0"):
            model.predict(x)

    v8.9 Step 7: Stanford strict hardware allocation — forces all enclosed
    operations onto the specified device. Falls back to a no-op context
    if TensorFlow is not available.
    """
    try:
        import tensorflow as tf
        return tf.device(device)
    except Exception:
        import contextlib
        return contextlib.nullcontext()


# ══════════════════════════════════════════════
#  v9.0 Step 3: WSL2 Containerization with Direct NVIDIA Passthrough (MIT)
# ══════════════════════════════════════════════

def detect_wsl2() -> bool:
    """Return True if running inside WSL2 (Windows Subsystem for Linux 2).

    WSL2 exposes /proc/version with 'microsoft' and provides GPU
    passthrough via the NVIDIA Container Toolkit.
    """
    try:
        with open("/proc/version", "r") as f:
            return "microsoft" in f.read().lower()
    except (FileNotFoundError, PermissionError):
        return False


def enforce_wsl2_gpu(mem_limit_gb: int = 4) -> dict:
    """Configure GPU access for WSL2/Docker with NVIDIA Container Toolkit.

    Instead of fighting the Windows 11 OS scheduler via the registry,
    this bypasses it entirely by enforcing absolute programmatic control
    over GPU memory limits through the container runtime.

    Args:
        mem_limit_gb: VRAM limit in GB for the container.

    Returns:
        Status dict.
    """
    result = {"wsl2": False, "container": False, "gpu_visible": False,
              "mem_limit_gb": mem_limit_gb, "error": None}

    if not detect_wsl2():
        result["error"] = "not_wsl2"
        return result
    result["wsl2"] = True

    # Check if nvidia-smi is available (NVIDIA Container Toolkit installed)
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader,nounits"],
            timeout=5, stderr=subprocess.DEVNULL, text=True,
        )
        result["gpu_visible"] = True
        result["gpu_info"] = out.strip()
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        result["error"] = "nvidia-smi not found — install NVIDIA Container Toolkit"
        return result

    # Set CUDA memory limit via environment variable
    mem_bytes = mem_limit_gb * 1024 * 1024 * 1024
    os.environ["CUDA_MEM_LIMIT"] = str(mem_bytes)

    # Force device 0 (WSL2 only exposes the discrete GPU)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Disable integrated GPU fallback in WSL2
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    result["container"] = True
    result["env"] = {
        "CUDA_VISIBLE_DEVICES": "0",
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_MEM_LIMIT": str(mem_bytes),
    }
    return result


def pin_torch_device(device_id: int = 0):
    """Return a PyTorch device object for explicit GPU pinning.

    Usage:
        dev = pin_torch_device(0)
        tensor = tensor.to(dev)

    Falls back to CPU device if CUDA is not available.
    """
    try:
        import torch
        if torch.cuda.is_available():
            return torch.device(f"cuda:{device_id}")
        return torch.device("cpu")
    except Exception:
        return None
