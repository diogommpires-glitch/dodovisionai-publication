# Hardware setup migrated from main.py
import os
import sys

def setup_hardware():
    import faulthandler
    # v9.5.x — configure structured logging before anything else prints.
    # DODO_LOG_FORMAT=json for production aggregators, text by default.
    try:
        import log_setup as _log_setup
        _log_setup.configure(force=True)
        _bootlog = _log_setup.get_logger("boot")
        _bootlog.info("log_setup armed (format=%s, level=%s)",
                       os.environ.get("DODO_LOG_FORMAT", "text"),
                       os.environ.get("DODO_LOG_LEVEL", "INFO"))
    except Exception as _e:
        print(f"[--] log_setup failed ({_e})")
    os.makedirs("logs", exist_ok=True)
    try:
        _crash_log = open(os.path.join("logs", "crash.log"), "ab", buffering=0)
        faulthandler.enable(file=_crash_log, all_threads=True)
        # Also mirror to stderr so the terminal shows the trace live.
        faulthandler.enable(file=sys.stderr, all_threads=True)
        print("[OK] faulthandler armed (crash traces -> logs/crash.log)")
    except Exception as _fh_e:
        print(f"[--] faulthandler setup skipped ({_fh_e})")

    # v8.9: GPU registry enforcement — must run BEFORE any CUDA/TF/ONNX import
    try:
        from gpu_enforce import enforce_discrete_gpu
        _gpu_reg = enforce_discrete_gpu()
        if _gpu_reg["applied"]:
            print(f"[OK] GPU registry: discrete GPU forced for {os.path.basename(_gpu_reg['python_exe'])} ({_gpu_reg.get('note','')})")
        elif _gpu_reg.get("error"):
            print(f"[--] GPU registry: {_gpu_reg['error']}")
    except Exception as _ge:
        print(f"[--] GPU registry enforcement skipped ({_ge})")
    # v9.1.3: CUDA device selection.
    #
    # Windows Task Manager "GPU 0 / GPU 1" is NOT the same as CUDA device index.
    # On a laptop with Intel/AMD integrated graphics + NVIDIA 3060:
    #   - CUDA only enumerates NVIDIA devices (the integrated GPU is DirectX-only).
    #   - There is exactly ONE CUDA device — the 3060 — at cuda index 0.
    #   - CUDA_VISIBLE_DEVICES=1 (the old value here) HID THE 3060 and the
    #     system silently fell back to CPU, which is why inference felt slow.
    #
    # Fix: pin CUDA_DEVICE_ORDER=PCI_BUS_ID for deterministic enumeration, and
    # point CUDA_VISIBLE_DEVICES at the NVIDIA (index 0). The Windows OS-level
    # "GPU 1 = discrete" preference is already enforced by gpu_enforce above
    # via the DirectX UserGpuPreferences registry key.
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    # v9.1.4: CUDA DLL search path bootstrap.
    #
    # onnxruntime-gpu for Windows dynamically links to cudart64_12.dll,
    # cublas64_12.dll, cudnn_graph64_9.dll etc. If the CUDA Toolkit isn't on
    # PATH, the DLL load fails with "error 126" and the runtime silently
    # falls back to CPU. To avoid manual PATH config, we probe three places
    # that commonly host the CUDA DLLs on a developer machine:
    #   1. PyTorch's bundled CUDA (torch/lib) — installed via the
    #      pip `torch==2.x+cu121` wheels; includes cudart + cublas + cudnn.
    #   2. NVIDIA's CUDA Toolkit default install (C:\Program Files\NVIDIA GPU
    #      Computing Toolkit\CUDA\v12.x\bin).
    #   3. A user-local NVIDIA cuDNN drop-in (%USERPROFILE%\cuda\bin).
    # The first hit wins; os.add_dll_directory() is preferred over PATH so
    # only this process sees the addition.
    def _bootstrap_cuda_dlls():
        import glob
        candidates = []
        # 1) PyTorch wheel
        try:
            import torch  # noqa: F401
            _torch_lib = os.path.join(os.path.dirname(__import__("torch").__file__), "lib")
            if os.path.isdir(_torch_lib):
                candidates.append(_torch_lib)
        except Exception:
            pass
        # 2) NVIDIA CUDA Toolkit — pick highest version found
        for v in ("12.5", "12.4", "12.3", "12.2", "12.1", "12.0", "11.8"):
            p = fr"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v{v}\bin"
            if os.path.isdir(p):
                candidates.append(p)
                break
        # 3) Any cuDNN drop-in in user profile
        for p in glob.glob(os.path.expanduser(r"~\cuda*\bin")):
            if os.path.isdir(p):
                candidates.append(p)
        # Add all
        added = []
        for p in candidates:
            try:
                if hasattr(os, "add_dll_directory"):
                    os.add_dll_directory(p)
                # also add to PATH for libraries that use Win32 LoadLibrary
                if p not in os.environ.get("PATH", ""):
                    os.environ["PATH"] = p + os.pathsep + os.environ.get("PATH", "")
                added.append(p)
            except Exception:
                pass
        if added:
            print(f"[OK] CUDA DLL search paths added: {len(added)} dir(s)")
            for p in added:
                print(f"     - {p}")
        else:
            print("[--] No CUDA DLL directories found — ONNX will fall back to CPU.")
            print("     Install one of:")
            print("       pip install torch --index-url https://download.pytorch.org/whl/cu121")
            print("       CUDA Toolkit 12.x: https://developer.nvidia.com/cuda-downloads")
            print("       cuDNN 9.x:         https://developer.nvidia.com/cudnn-downloads")
            print("     Or use DirectML (no CUDA install needed, runs on any GPU):")
            print("       pip uninstall onnxruntime-gpu && pip install onnxruntime-directml")
    _bootstrap_cuda_dlls()

    # v9.7.2 Blackwell tune — global TF32 + cuDNN benchmark.
    # `detectors.py` already sets these locally; doing it at boot makes
    # them apply BEFORE any model loads (torch.compile / inductor / cuBLAS
    # pick up the flags during their first call).
    # Free perf on Ampere+ (sm_80+); on the 5090 Laptop (sm_120) it's
    # typically 10–30 % on convolutions and matmuls in fp32.
    try:
        import torch as _torch
        if _torch.cuda.is_available():
            _torch.backends.cuda.matmul.allow_tf32 = True
            _torch.backends.cudnn.allow_tf32 = True
            _torch.backends.cudnn.benchmark = True
            print(f"[OK] PyTorch Blackwell tune: TF32 ON, cuDNN benchmark ON")
    except Exception as _t_e:
        print(f"[--] PyTorch Blackwell tune skipped ({_t_e})")


