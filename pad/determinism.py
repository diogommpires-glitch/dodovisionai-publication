"""
Deterministic-mode helpers for the PAD eval harness  (v9.8.1 Audit D2).

Camera-ready paper reproducibility: a downstream reader must be able to
re-run `python -m pad.eval` and get bit-for-bit identical scored.jsonl
output. Without these flags, ORT's CUDA kernels and TF's autotuner use
non-deterministic algorithms (TF32 reductions, cuBLAS algorithm choice,
cuDNN convolution autotune), and PyTorch uses non-deterministic GPU ops
unless explicitly disabled.

Call `seed_everything(0)` and `lock_runtime_deterministic()` at the top
of every eval/training entry point that should be reproducible.

Reference: PyTorch reproducibility docs +
https://docs.nvidia.com/deeplearning/cudnn/developer-guide/index.html#deterministic
"""
from __future__ import annotations
import hashlib
import os
import random
import sys
from typing import Dict


def seed_everything(seed: int = 0) -> Dict[str, str]:
    """Set every RNG we care about. Returns a dict the caller can dump
    into the report's metadata block for audit-trail purposes.

    Order matters: env vars BEFORE imports of the libs that read them.
    """
    seed = int(seed)
    info = {"seed": str(seed)}

    # Python's hash randomization (set BEFORE module imports for it
    # to take effect; if Python already started, this is a no-op but
    # we record the intent for the audit log).
    os.environ.setdefault("PYTHONHASHSEED", str(seed))

    # NumPy + stdlib RNG.
    random.seed(seed)
    try:
        import numpy as _np
        _np.random.seed(seed)
        info["numpy"] = _np.__version__
    except Exception:
        pass

    # PyTorch.
    try:
        import torch as _torch
        _torch.manual_seed(seed)
        if _torch.cuda.is_available():
            _torch.cuda.manual_seed_all(seed)
        info["torch"] = _torch.__version__
    except Exception:
        pass

    # TensorFlow (rarely used at inference here, but set for
    # completeness in case a Keras model is loaded).
    try:
        import tensorflow as _tf
        _tf.random.set_seed(seed)
        info["tensorflow"] = _tf.__version__
    except Exception:
        pass

    return info


def lock_runtime_deterministic() -> Dict[str, str]:
    """Configure every backend we use to prefer deterministic algorithms
    over fast ones. Returns the applied settings for the audit log.

    Idempotent. Safe to call alongside other env-var setters.
    """
    applied: Dict[str, str] = {}

    # cuBLAS deterministic workspace
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    applied["CUBLAS_WORKSPACE_CONFIG"] = os.environ["CUBLAS_WORKSPACE_CONFIG"]

    # cuDNN: disable autotune + force deterministic algorithms.
    # PyTorch reads these flags via torch.backends.cudnn.
    try:
        import torch as _torch
        _torch.use_deterministic_algorithms(True, warn_only=True)
        applied["torch.use_deterministic_algorithms"] = "True"
        if hasattr(_torch.backends, "cudnn"):
            _torch.backends.cudnn.deterministic = True
            _torch.backends.cudnn.benchmark = False
            applied["torch.backends.cudnn.deterministic"] = "True"
            applied["torch.backends.cudnn.benchmark"] = "False"
        # TF32 (Ampere+ matmul autoreduce) — turn it off for parity
        # across compute capabilities.
        if hasattr(_torch.backends, "cuda"):
            try:
                _torch.backends.cuda.matmul.allow_tf32 = False
                _torch.backends.cudnn.allow_tf32 = False
                applied["torch.backends.cuda.matmul.allow_tf32"] = "False"
            except Exception:
                pass
    except Exception:
        pass

    # ORT-side: arena_extend_strategy + single-threaded execution makes
    # outputs deterministic across runs. The CUDAExecutionProvider has
    # its own deterministic option (added in ORT 1.18).
    os.environ.setdefault("ORT_DETERMINISTIC", "1")
    applied["ORT_DETERMINISTIC"] = os.environ["ORT_DETERMINISTIC"]

    # NumPy: no MKL random tricks. Already deterministic when seeded.

    return applied


def audit_payload(seed: int = 0) -> Dict[str, object]:
    """Return a JSON-friendly snapshot of every value the eval should
    embed in `scored.jsonl` metadata so downstream readers can verify
    the run was deterministic-mode. Includes Python+numpy+torch
    versions, ORT version, env vars, and a SHA-256 of the sys.path
    (a tiny tell that the same code was executed)."""
    out: Dict[str, object] = {
        "python_version": sys.version.split()[0],
        "platform": sys.platform,
    }
    out.update({"_seed_info": seed_everything(seed)})
    out.update({"_runtime_locks": lock_runtime_deterministic()})
    try:
        import onnxruntime as _ort
        out["onnxruntime"] = _ort.__version__
        out["onnxruntime_providers"] = _ort.get_available_providers()
    except Exception:
        pass
    # Hash sys.path so a downstream reader can spot a non-canonical
    # install (e.g. a stale .pyc shadowing a fresh source file).
    out["sys_path_sha256"] = hashlib.sha256(
        ("||".join(sys.path)).encode("utf-8")
    ).hexdigest()[:16]
    return out


def write_provenance(path: str, seed: int = 0) -> None:
    """Convenience: write `audit_payload()` to `path` as JSON. The
    eval harness should call this once per run."""
    import json
    payload = audit_payload(seed=seed)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
