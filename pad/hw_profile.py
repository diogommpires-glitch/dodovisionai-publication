"""pad/hw_profile.py -- v9.10 hardware probe + auto-tune helper.

The v9.10 max-utilization profile baked in defaults that are perfect
for the reference Legion Pro 7 (5090 Laptop 24 GB / 24-core 275HX /
64 GB RAM). On a 12 GB 4080 or a 64 GB cloud A100, those defaults are
either too aggressive (OOM) or too conservative (leaves performance on
the table). This module derives sensible (workers, batch, prefetch)
values from whatever hardware torch sees at startup, so the same
training scripts run optimally everywhere.

Two public entry points:

    profile = hardware_profile()
        Returns a dict with gpu_name, sm_count, cuda_cores, vram_gb,
        compute_capability, cpu_logical, cpu_physical, ram_gb. CUDA
        core count is derived from SM count x cores-per-SM (per the
        published NVIDIA architecture tables); accurate for Maxwell
        through Blackwell. Falls back to a CPU-only profile if no
        CUDA device is visible.

    cfg = auto_tune_for(profile)
        Returns {"workers", "batch", "prefetch"} sized for the
        profile. The reference point is the 5090 box (1536/12/4);
        smaller boxes get scaled-down values, larger boxes the same
        defaults (we don't try to push past 1536 because Phase C is
        CPU-bound on PIL augmentation past that point).

Both functions are pure (no side effects). The training scripts call
them, log the choices, then proceed.
"""
from __future__ import annotations
import os
from typing import Optional, Tuple


# Architecture -> CUDA cores per SM table. Source: NVIDIA whitepapers.
# Major version of compute capability is the architecture key:
#   12 -> Blackwell (RTX 50-series)        128 cores/SM
#    9 -> Hopper (H100 / GH200)            128 cores/SM
#    8 -> Ampere/Ada (RTX 30/40, A100)     128 cores/SM
#    7 -> Volta (V100) / Turing (RTX 20)    64 cores/SM (Volta) / 64 (Turing)
#    6 -> Pascal (GTX 10)                   64 cores/SM
#    5 -> Maxwell (GTX 9)                  128 cores/SM
# Numbers are CUDA-FP32-cores; the doubled-throughput claims (256/SM
# on Ada) refer to combined FP32+INT32 issue, not core count.
_CORES_PER_SM = {12: 128, 9: 128, 8: 128, 7: 64, 6: 64, 5: 128}


def _cuda_cores_from_cap(major: int, sm_count: int) -> int:
    """Estimate total CUDA cores. Returns 0 for unknown architectures
    rather than guessing -- caller treats 0 as 'unknown'."""
    cps = _CORES_PER_SM.get(int(major), 0)
    return int(cps * sm_count)


def hardware_profile() -> dict:
    """Probe the runtime hardware. CUDA-aware. Lazy imports so this
    module imports cleanly even on machines without torch installed."""
    out: dict = {
        "gpu_name": None,
        "compute_capability": None,
        "sm_count": 0,
        "cuda_cores": 0,
        "cores_per_sm": 0,
        "vram_gb": 0.0,
        "cpu_logical": os.cpu_count() or 1,
        "cpu_physical": None,
        "ram_gb": 0.0,
        "torch_version": None,
        "cuda_version": None,
    }
    try:
        import psutil
        out["cpu_physical"] = psutil.cpu_count(logical=False)
        out["ram_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
    except Exception:
        pass  # nolint:silent-except graceful-degradation -- hw_profile must return a usable dict on minimal installs without psutil
    try:
        import torch
        out["torch_version"] = torch.__version__
        out["cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            out["gpu_name"] = p.name
            out["compute_capability"] = (int(p.major), int(p.minor))
            out["sm_count"] = int(p.multi_processor_count)
            out["cores_per_sm"] = _CORES_PER_SM.get(int(p.major), 0)
            out["cuda_cores"] = _cuda_cores_from_cap(p.major,
                                                    p.multi_processor_count)
            out["vram_gb"] = round(p.total_memory / 1e9, 1)
    except Exception:
        pass  # nolint:silent-except graceful-degradation -- probe must work on torchless hosts; cuda_* fields stay 0/None
    return out


def auto_tune_for(profile: dict,
                  *,
                  cap_workers: int = 16,
                  cores_per_worker: int = 2) -> dict:
    """Derive (workers, batch, prefetch) from a hardware profile.

    Reference points (calibrated 2026-05-09 against actual measurements):
      * 5090 Laptop 24 GB / 24-core 275HX -> workers=12, batch=1536
      * 4090 24 GB / 16-core             -> workers=8,  batch=1536
      * 4080  16 GB / 12-core             -> workers=6,  batch=1024
      * 3080  10 GB /  8-core             -> workers=4,  batch=768
      * cpu-only / no GPU                  -> workers=2,  batch=64

    Workers heuristic: `cpu_logical // cores_per_worker`, capped at
    `cap_workers`. The 1-worker-per-2-cores rule is the empirical sweet
    spot for DataLoaders bottlenecked on PIL augmentation (PIL releases
    the GIL but the per-image work has diminishing returns past that
    ratio; the audit measured this on the 5090 box). Cap at 16 because
    Windows `spawn`-mode multi-proc has a ~0.3 s startup tax per worker
    that erodes the speedup beyond that.

    Batch: VRAM-banded thresholds tuned for MobileNetV3-small at NHWC +
    bf16 (~12.5 MB activation peak per sample, including augmentation
    copy). Beyond 24 GB we don't go higher because Phase C is CPU-bound
    on PIL augmentation past ~1536 -- a bigger batch just inflates
    per-batch CPU work without GPU benefit (verified by audit).
    """
    cpu_logical = max(1, int(profile.get("cpu_logical") or 1))
    # v9.10.3 (2026-05-09 incident-fix #2): cap workers at min(cap_workers, 8).
    # Even with prefetch=2 the cumulative shared-memory commit on Windows
    # grows monotonically across batches under heavy worker load (rate decays
    # 758 -> 590 img/s over 250 batches, then crashes at epoch boundary with
    # Win32 1455 ERROR_COMMITMENT_LIMIT). Eight workers proved stable across
    # the previous v9.9 production runs that produced patch_depth_fas_v8.pth
    # in 2.93 h. We keep the v9.10 win (going from 4 -> 8 vs the historical
    # baseline) while staying inside the proven-safe shared-memory envelope.
    _MAX_SAFE_WORKERS = 8
    workers = max(2, min(cap_workers, _MAX_SAFE_WORKERS,
                         cpu_logical // max(1, cores_per_worker)))

    vram_gb = float(profile.get("vram_gb") or 0)
    if vram_gb >= 22:
        # v9.10.3 (2026-05-09): batch dropped 1536 -> 1024 to match the
        # proven v9.9 footprint that produced the existing v8.pth. The
        # 1536 budget required ~21 GB train + ~10 GB val of shared
        # memory (12 workers x 4 prefetch); even at prefetch=2 the
        # cumulative leak still exhausted commit charge over an epoch.
        # 1024 + 8 workers + prefetch=2 => 9.3 GB train + 4.7 GB val =
        # 14 GB total, which v9.9 ran for 2.93 h without incident.
        batch = 1024
    elif vram_gb >= 14:
        batch = 1024
    elif vram_gb >= 10:
        batch = 768
    elif vram_gb >= 8:
        batch = 512
    elif vram_gb >= 4:
        batch = 256
    elif vram_gb > 0:
        batch = 128
    else:
        # No GPU detected -- fall back to a CPU-friendly tiny batch.
        batch = 64
        workers = min(workers, 4)

    # Prefetch heuristic (v9.10.1, 2026-05-09 incident-fix):
    # Originally we hard-coded prefetch=4. On Windows multiprocessing-spawn
    # mode the per-worker prefetch buffer lives in shared memory. With
    # workers=12 + batch=1536 the worst-case commit is
    #   workers x prefetch x batch x 3 x 224 x 224 x 4 bytes
    #   = 12 x 4 x 1536 x 0.57 MB = 42 GB
    # plus another ~10 GB for the val loader + ~10 GB OS/CUDA/python =>
    # ~62 GB out of a 68 GB Windows commit ceiling on this box. The first
    # OOM under shared-memory pressure surfaces as Win32 error 1455
    # (ERROR_COMMITMENT_LIMIT) during torch.UntypedStorage._new_shared,
    # which kills DataLoader workers silently and leaves Phase C stuck.
    #
    # Halving prefetch to 2 cuts shared-mem to ~21 GB (train) + ~5 GB
    # (val) = 26 GB, well under the ceiling, with negligible throughput
    # cost: at steady state the GPU consumes a 1536-batch in ~150 ms
    # while 12 workers can produce one in ~30 ms each, so prefetch=2 is
    # always enough to keep the GPU fed.
    prefetch = 4 if (workers <= 6 and batch <= 1024) else 2
    return {"workers": workers, "batch": batch, "prefetch": prefetch}


def format_profile_line(profile: dict, cfg: Optional[dict] = None) -> str:
    """One-line log-friendly summary. The training scripts emit this so
    the user can grep `[hw]` in any log to see which profile ran."""
    g = profile.get("gpu_name") or "no-cuda"
    cc = profile.get("compute_capability")
    cc_s = f"sm_{cc[0]}{cc[1]}" if cc else "?"
    sm = profile.get("sm_count", 0)
    cores = profile.get("cuda_cores", 0)
    vram = profile.get("vram_gb", 0)
    cpu_l = profile.get("cpu_logical", 0)
    cpu_p = profile.get("cpu_physical") or cpu_l
    ram = profile.get("ram_gb", 0)
    line = (f"[hw] gpu={g} {cc_s} sm={sm} cores={cores} vram={vram:.1f}GB "
            f"cpu={cpu_p}p/{cpu_l}l ram={ram:.1f}GB")
    if cfg:
        line += (f"  -> auto_tune workers={cfg['workers']} "
                 f"batch={cfg['batch']} prefetch={cfg['prefetch']}")
    return line
