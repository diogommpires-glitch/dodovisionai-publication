"""pad/run_overnight.py -- robust overnight queue orchestrator.

Replaces queue_mega_v3.cmd's :run_stage macro (which had a 9-arg cmd-parser
ceiling). Pure Python so quoting, long --flag chains, and Unicode just work.

Stages are gated by env vars (same names as before):
    MASTER_BLINK         -> stages 10, 11, 12
    MASTER_TEMPORAL_V3   -> stage 20
    MASTER_FXR_PGD16     -> stage 30
    MASTER_AUDIO_PGD     -> stage 40
    MASTER_FINAL         -> stages 50, 51, 52
    MASTER_NPU           -> stage 60

Per-stage idempotency via .done markers in queue/mega_v3/.
ntfy push fires at queue start, every stage start/end, every failure, and
queue end. Requires DODO_NTFY_TOPIC and DODO_NTFY_TOKEN private configuration.

Run as:
    .venv\\Scripts\\python.exe pad\\run_overnight.py
"""
from __future__ import annotations
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from pad.notify import ntfy as _ntfy

REPO = Path(__file__).resolve().parent  # script at repo root (moved out of pad/ to dodge pad/types.py stdlib shadowing)
PY = REPO / ".venv" / "Scripts" / "python.exe"


# ---- Belt-and-braces: hold the system-awake guard for the queue's lifetime.
# Even if black_screen.ps1 / keep_awake.ps1 aren't running, the orchestrator
# itself prevents Windows from sleeping during multi-hour training. Released
# automatically on process exit.
def _arm_keep_awake() -> None:
    if sys.platform != "win32":
        return
    try:
        import ctypes
        ES_CONTINUOUS      = 0x80000000
        ES_SYSTEM_REQUIRED = 0x00000001
        # Cast to c_uint to avoid the high-bit signed-int interpretation pitfall
        # on Python 3 + ctypes; explicit |= keeps both flags set.
        ctypes.windll.kernel32.SetThreadExecutionState(
            ctypes.c_uint(ES_CONTINUOUS | ES_SYSTEM_REQUIRED))
        print("[ok] orchestrator keep-awake guard armed (ES_SYSTEM_REQUIRED)")
    except Exception as e:
        print(f"[--] keep-awake guard failed (non-fatal): {e}")


_arm_keep_awake()
QDIR = REPO / "queue" / "mega_v3"
QDIR.mkdir(parents=True, exist_ok=True)
QLOG = QDIR / "queue.log"


def ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def log(msg: str) -> None:
    line = f"[{ts()}] {msg}"
    print(line, flush=True)
    with open(QLOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def ntfy(msg: str, priority: str = "default", title: str = "DodoVisionAI queue") -> None:
    _ntfy(title, msg, priority=priority, tags="hourglass", timeout_s=5, log_fn=log)


def run_stage(sid: str, desc: str, cmd: list[str]) -> int:
    marker = QDIR / f"{sid}.done"
    slog = QDIR / f"{sid}.log"
    if marker.exists():
        log(f"[skip] stage {sid} [{desc}] :: already .done")
        return 0
    log(f"")
    log(f"============================================================")
    log(f"[stage {sid}] {desc}")
    log(f"============================================================")
    log(f"  cmd: {' '.join(cmd)}")
    ntfy(f"stage {sid} START :: {desc}", priority="default")
    t0 = time.time()
    with open(slog, "w", encoding="utf-8") as logf:
        logf.write(f"[{ts()}] BEGIN {sid} {desc}\n")
        logf.write(f"  cmd: {' '.join(cmd)}\n\n")
        logf.flush()
        proc = subprocess.run(
            cmd, cwd=str(REPO), stdout=logf, stderr=subprocess.STDOUT
        )
    rc = proc.returncode
    dt = time.time() - t0
    log(f"[end ] stage {sid} rc={rc} dt={dt:.1f}s")
    if rc == 0:
        marker.write_text("done\n", encoding="utf-8")
        log(f"[done] stage {sid}")
        ntfy(f"stage {sid} DONE ({dt:.0f}s) :: {desc}", priority="default")
    else:
        log(f"[FAIL] stage {sid} rc={rc} -- see {slog}")
        # Print last 30 lines for visibility in the foreground console
        try:
            tail = slog.read_text(encoding="utf-8", errors="replace").splitlines()[-30:]
            for line in tail:
                print(f"  | {line}", flush=True)
        except Exception:  # nolint:silent-except log-tail-display-best-effort
            pass
        ntfy(f"stage {sid} FAILED rc={rc} :: {desc}", priority="urgent")
    return rc


def gate(env_name: str) -> bool:
    return os.environ.get(env_name, "0").strip() == "1"


def preflight() -> int:
    log("")
    log("============================================================")
    log("[stage 00] preflight (always runs, never marked done)")
    log("============================================================")
    if not Path(r"E:\pad_datasets").exists():
        log("[FAIL] E:\\pad_datasets missing :: mount the Lexar SSD as drive E:")
        ntfy("preflight FAILED :: E:\\pad_datasets missing", "urgent")
        return 2
    rc = subprocess.run([str(PY), "-c",
        "import torch; assert torch.cuda.is_available(), 'no CUDA'; "
        "n=torch.cuda.get_device_name(0); print('[OK] GPU:', n); "
        "assert torch.cuda.get_device_capability(0) >= (8,0), 'sm_80+ required'"
    ], cwd=str(REPO)).returncode
    if rc:
        ntfy("preflight FAILED :: GPU check", "urgent")
        return rc
    rc = subprocess.run([str(PY), "-m", "pytest",
        "tests/test_regression_guards.py", "tests/test_diagnose_env.py",
        "-q", "--tb=line", "--no-header", "-p", "no:cacheprovider"
    ], cwd=str(REPO)).returncode
    if rc:
        log("[FAIL] baseline pytest red")
        ntfy("preflight FAILED :: pytest baseline red", "urgent")
        return rc
    log("[OK] preflight passed")
    return 0


def main() -> int:
    if not PY.exists():
        log(f"[FAIL] venv missing at {PY}")
        return 1

    flags = {
        "BLINK":       gate("MASTER_BLINK"),
        "TEMPORAL_V3": gate("MASTER_TEMPORAL_V3"),
        "FXR_PGD16":   gate("MASTER_FXR_PGD16"),
        "AUDIO_PGD":   gate("MASTER_AUDIO_PGD"),
        "FINAL":       gate("MASTER_FINAL"),
        "NPU":         gate("MASTER_NPU"),
        # API_up v9.8 additions
        "MVIT":        gate("MASTER_MVIT"),         # stage 70: MViT-V2-S temporal-head experiment
        "FP16":        gate("MASTER_FP16"),         # stage 80: FP16 quant + A/B latency
    }
    if NTFY_TOPIC:
        log(f"[ntfy] pushing to https://ntfy.sh/{NTFY_TOPIC}")
    else:
        log("[ntfy] NTFY_TOPIC not set :: no push notifications")
    flags_str = " ".join(f"{k}={int(v)}" for k, v in flags.items())
    log(f"[run] flags: {flags_str}")
    ntfy(f"queue START :: {flags_str}", "default")

    rc = preflight()
    if rc:
        return rc

    PY_ = str(PY)
    stages: list[tuple[bool, str, str, list[str]]] = []

    if flags["BLINK"]:
        stages.append((True, "10", "fetch MRL Eye Dataset",
                       [PY_, "-m", "pad.fetch_mrl_eye", "--root", r"E:\pad_datasets\MRL"]))
        stages.append((True, "11", "train eye_state_v1 CNN",
                       [PY_, "-u", "-m", "pad.train_eye_state",
                        "--data", r"E:\pad_datasets\MRL\mrlEyes_2018_01",
                        "--epochs", "15", "--batch", "256", "--workers", "4",
                        "--out", r"pad\models\eye_state_v1.onnx",
                        "--out-pth", r"pad\models\eye_state_v1.pth"]))
        stages.append((True, "12", "bench eye_state ONNX (CPU vs CUDA vs NPU)",
                       [PY_, r"scripts\bench_eye_state.py"]))

    if flags["TEMPORAL_V3"]:
        stages.append((True, "20", "train deepfake_temporal_v3 (30 ep MC3-18)",
                       [PY_, "-u", "-m", "pad.train_temporal_deepfake",
                        "--data", r"E:\pad_datasets\video_clips_celebdf",
                        "--epochs", "30", "--batch", "16",
                        "--num-workers", "4", "--save-every-epoch",
                        "--out", r"pad\models\deepfake_temporal_v3.onnx"]))

    if flags["FXR_PGD16"]:
        # NOTE: --epsilon is int in /255 units per train_face_xray_adv.py:99
        # (eps = float(epsilon) / 255.0). So 16 means 16/255.
        stages.append((True, "30", "train face_xray_pgd eps=16/255 (PGD-10)",
                       [PY_, "-u", "-m", "pad.train_face_xray_adv",
                        "--data", r"E:\pad_datasets\face_xray",
                        "--attack", "pgd", "--epsilon", "16",
                        "--pgd-steps", "10", "--epochs", "5",
                        "--out", r"pad\models\face_xray_pgd_eps16.onnx"]))

    if flags["AUDIO_PGD"]:
        # NOTE: --epsilon is int in /100 units per train_audio_adv.py:218
        # (eps_amp = float(epsilon) / 100.0). 2 = 2/100 = 0.02 (FGSM only).
        stages.append((True, "40", "train audio_pad adv (FGSM eps=2/100)",
                       [PY_, "-u", "-m", "pad.train_audio_adv",
                        "--root", r"E:\pad_datasets\ASVspoof2019LA\LA\LA",
                        "--epochs", "5", "--epsilon", "2",
                        "--out", r"pad\models\audio_pad_adv.onnx"]))

    if flags["FINAL"]:
        stages.append((True, "50", "re-bench latency_profile on 5090",
                       [PY_, "-m", "pad.latency_profile", "--warmup", "30",
                        "--runs", "200", "--provider", "CUDAExecutionProvider"]))
        stages.append((True, "51", "re-run paired_bootstrap_auc n=10000",
                       [PY_, "-m", "pad.paired_bootstrap_auc",
                        "--inputs",
                        r"reports\meta_train_records_celeba_v6.jsonl",
                        r"reports\meta_train_records_replay_v5.jsonl",
                        r"reports\deepfake_temporal_v2_records_celebdf.jsonl",
                        r"reports\audio_asvspoof_records_v3.jsonl",
                        "--n-boot", "10000", "--seed", "2026"]))
        stages.append((True, "52", "re-run cross_dataset_matrix",
                       [PY_, "-m", "pad.cross_dataset_matrix",
                        "--sources",
                        r"celeba=reports\meta_train_records_celeba_v6.jsonl",
                        r"replay=reports\meta_train_records_replay_v5.jsonl",
                        r"celebdf=reports\meta_train_records_celebdf_v5.jsonl",
                        r"asvspoof=reports\audio_asvspoof_records_v3.jsonl",
                        "--out", r"reports\cross_dataset_matrix_post_v3.md",
                        "--seed", "2026"]))

    if flags["NPU"]:
        stages.append((True, "60", "OpenVINO bench audio_pad CPU vs NPU",
                       [PY_, r"scripts\bench_npu_audio.py"]))

    # API_up v9.8: stage 70 -- MViT-V2-S temporal-head training (~7-8 h)
    # Produces pad/models/deepfake_temporal_mvit.{onnx,pth}; the v3 MC3-18
    # remains in place. Compare AUC post-training before swapping.
    if flags["MVIT"]:
        stages.append((True, "70", "train deepfake_temporal_mvit (MViT-V2-S, 30 ep, 224x224)",
                       [PY_, "-u", "-m", "pad.train_temporal_mvit",
                        "--data", r"E:\pad_datasets\video_clips_celebdf",
                        "--epochs", "30", "--batch", "4",
                        "--num-workers", "4", "--save-every-epoch",
                        "--out", r"pad\models\deepfake_temporal_mvit.onnx"]))

    # API_up v9.8: stage 80 -- FP16 quantization batch + A/B latency bench.
    # Re-runs `scripts/quantize_fp16.py` (idempotent on already-quantized files
    # since it overwrites) and then a focused latency_profile --ab-fp16 pass.
    if flags["FP16"]:
        stages.append((True, "80", "FP16 quantize all heavy ONNX",
                       [PY_, r"scripts\quantize_fp16.py"]))
        stages.append((True, "81", "latency_profile --ab-fp16 on 5090",
                       [PY_, "-m", "pad.latency_profile",
                        "--warmup", "30", "--runs", "200",
                        "--provider", "CUDAExecutionProvider", "--ab-fp16"]))

    # Execute
    for _gate, sid, desc, cmd in stages:
        rc = run_stage(sid, desc, cmd)
        if rc != 0:
            log("")
            log("============================================================")
            log("  QUEUE ABORTED -- see queue\\mega_v3\\queue.log + the failed stage's log")
            log("============================================================")
            ntfy(f"queue ABORTED at stage {sid} :: see queue.log", "urgent",
                 title="DodoVisionAI queue ABORTED")
            return 1

    # Done
    (QDIR / "queue_done.marker").write_text("done\n", encoding="utf-8")
    log("")
    log("============================================================")
    log(f"  ALL STAGES COMPLETE  ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
    log(f"  log: {QLOG}")
    log(f"  per-stage logs: {QDIR}\\<sid>.log")
    log("============================================================")
    ntfy(f"queue ALL STAGES DONE :: safe to wake the panel", "high",
         title="DodoVisionAI queue DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())

