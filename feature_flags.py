"""
DODO VISION AI v9.3 — Feature Flag Registry

Every v9.3 "Same-Moment UI" change ships behind exactly one feature flag so
it can be rolled back individually. Flags are resolved in this order:

    URL query param (?ff_foo=1)  >  environment variable FF_FOO  >  default

`?safe=1` (or SAFE_MODE=1 env) forces every v9.3 flag OFF regardless of
other overrides — the mandatory incident-triage rollback path called out in
step 10.3 of the v9.3 plan.

This module is imported by both main.py (server-side flag evaluation) and
the UI template (client-side flags are serialized into the HTML so the
browser reads the same defaults).

Citations:
  - Liu & Heer 2014, "Effects of Interactive Latency on Exploratory
    Visual Analysis," IEEE InfoVis: motivation for per-source latency
    budgets that the flag set targets.
  - Crankshaw 2019 (UC Berkeley EECS-2019-171): flag-gated rollout of
    prediction-serving stages — the pattern we follow here.
"""
from __future__ import annotations
import os
from typing import Any, Dict

# ── Flag names and defaults ──
# Each tuple is (flag, default, description). Defaults are CONSERVATIVE —
# a v9.3 flag defaults ON only if the code path is guaranteed safe with
# graceful fallback; otherwise defaults OFF and requires explicit opt-in.
FLAGS: Dict[str, Dict[str, Any]] = {
    # Step 1 — split reactivity tiers in the UI + server envelopes.
    "SPLIT_TIERS":         {"default": True,  "step": 1, "desc": "Split INSTANT vs BEST-EFFORT data planes client-side"},
    # Step 2 — MediaStreamTrackProcessor + WebCodecs zero-copy capture.
    "MSTP_CAPTURE":        {"default": True,  "step": 2, "desc": "Zero-copy VideoFrame capture; requires MediaStreamTrackProcessor"},
    # Step 3 — WebGPU overlay. Feature-detected at runtime; flag gates opt-in.
    "WEBGPU":              {"default": True,  "step": 3, "desc": "WebGPU overlay renderer; falls back to WebGL2 then 2D canvas"},
    # Step 4 — On-device MediaPipe Face Landmarker (INSTANT tier).
    "CLIENT_TRACKER":      {"default": False, "step": 4, "desc": "On-device MediaPipe Face Landmarker (browser WASM+GPU)"},
    # Step 5 — New client→server binary WS protocol.
    "WS_MULTIPLEX":        {"default": True,  "step": 5, "desc": "Multiplexed WS envelopes: instant/corr/hr/spo2/decep/llm/vitals"},
    # Step 6 — Client-side ONNX FER head.
    "CLIENT_EMO":          {"default": False, "step": 6, "desc": "On-device emotion head via ONNX Runtime Web"},
    # Step 7 — MessagePack + Worker + SharedArrayBuffer ring.
    "SAB_RING":            {"default": False, "step": 7, "desc": "WS parsing in a Worker via SharedArrayBuffer ring (requires COOP/COEP)"},
    "COOP_COEP":           {"default": False, "step": 7, "desc": "Emit Cross-Origin-Opener/Embedder-Policy headers for SAB isolation"},
    # Step 8 — Solid.js signals.
    "SOLID_SIGNALS":       {"default": False, "step": 8, "desc": "SolidJS fine-grained signals for HTML pills"},
    # Step 9 — Compositor-thread CSS.
    "CSS_COMPOSITOR":      {"default": True,  "step": 9, "desc": "Compositor-thread transforms + content-visibility + Houdini @property"},
    # Step 10 — Latency HUD + back-pressure.
    "LAT_HUD":             {"default": True,  "step": 10, "desc": "Latency HUD panel + per-stage p50/p95/p99 telemetry"},
    "BACKPRESSURE":        {"default": True,  "step": 10, "desc": "Adaptive back-pressure: shed low-priority envelopes under load"},
    # ── v9.4 Presentation Attack Detection (PAD) stack ──
    # Every PAD step ships behind its own flag (§3.1 of the v9.4 plan).
    # All default OFF in production until Step 10 acceptance criteria met.
    "PAD_FUSION":          {"default": False, "step": "9.4.1",  "desc": "Log-odds fusion of PAD branches (master PAD flag)"},
    "PAD_RPPG":            {"default": False, "step": "9.4.2",  "desc": "rPPG liveness: pulse SNR + inter-ROI coherence + refresh-band"},
    "PAD_MOIRE":           {"default": False, "step": "9.4.3",  "desc": "Moiré branch: server GBT + client FFT fast-guard"},
    "PAD_CNN":             {"default": False, "step": "9.4.4",  "desc": "Patch+depth CNN (ONNX; no-op when weights missing)"},
    "PAD_DYNAMICS":        {"default": False, "step": "9.4.5",  "desc": "3D consistency + blink + vergence + pupil Bayes-net"},
    "PAD_DEEPFAKE":        {"default": False, "step": "9.4.6",  "desc": "Deepfake stream: Face X-ray + 3DMM + PRNU"},
    "PAD_AUDIO":           {"default": False, "step": "9.4.7",  "desc": "ASVspoof-style audio anti-spoofing on /audio/chunk"},
    "PAD_ADV_DEFENSE":     {"default": False, "step": "9.4.8",  "desc": "Adversarial patch defense (F-Cutout + JPEG + optional diffusion)"},
    "PAD_UI":              {"default": False, "step": "9.4.9",  "desc": "PAD UI: INSTANT-tier ring + BEST-EFFORT drawer + challenge"},
    "PAD_EVAL":            {"default": False, "step": "9.4.10", "desc": "PAD eval harness (ISO/IEC 30107-3 reports) + /pad/stats"},
    "PAD_SCREEN_FLASH":    {"default": True, "step": "9.8.3",  "desc": "Active screen-flash liveness branch (on with PAD; set FF_PAD_SCREEN_FLASH=0 to disable)"},
    # ── v9.5 open-source stack upgrades (2026 report) ─────────────
    "SILERO_VAD":          {"default": False, "step": "9.5.1", "desc": "Silero VAD (MIT, 1 MB, 6k+ langs) replacing the energy VAD"},
    "LLM_COMPRESS":        {"default": False, "step": "9.5.2", "desc": "LLMLingua prompt compression kicks in when budget > 70%"},
    "FAISS_MEMORY":        {"default": False, "step": "9.5.3", "desc": "Per-person FAISS retrieval memory for LLM context"},
    "PYANNOTE":            {"default": False, "step": "9.5.4", "desc": "Real pyannote.audio 3.1 diarization (gated HF model; needs HUGGING_FACE_HUB_TOKEN)"},
    "PAD_META_FUSE":       {"default": False, "step": "9.5.5", "desc": "LightGBM meta-classifier fusion (pad.meta_fuse); falls back to log-odds when pickle missing"},
    "PAD_CALIBRATE":       {"default": False, "step": "9.5.6", "desc": "Apply per-branch Platt/isotonic calibrators (pad.calibrate); no-op when pickle missing"},
    # ── v9.8 API_up upgrade fork (2026-05) ────────────────────────
    # Registered here so /flags surfaces them alongside the v9.3+ set.
    # Each is read directly from os.environ by the consumer module
    # (the FF_* prefix is the canonical interface) -- this registry is
    # for OBSERVABILITY only. Defaults are False so a fresh clone runs
    # paper-canonical without these toggled on.
    "USE_FP16":            {"default": False, "step": "9.8.1", "desc": "Route deepfake/face_xray/patch_depth ONNX inference to FP16 weights (1.27-1.62x speedup, parity-verified)"},
    "BLINK_CNN":           {"default": False, "step": "9.8.2", "desc": "Add eye_state_v1.onnx as a 3rd OR-vote channel in BlinkDetector (helps glasses-wearers)"},
    "AUDIO_NPU":           {"default": False, "step": "9.8.3", "desc": "Route audio_pad to OpenVINO INT8 IR on Intel NPU/CPU (frees dGPU for PAD pipeline)"},
}

# ── Server-side evaluation ──
_SAFE_MODE_ENV = os.environ.get("SAFE_MODE", "").lower() in ("1", "true", "yes", "on")

def get(flag: str, default: Any = None) -> bool:
    """Server-side flag resolution. Env overrides defaults; SAFE_MODE overrides all."""
    if _SAFE_MODE_ENV:
        return False
    entry = FLAGS.get(flag)
    fallback = entry["default"] if entry else default
    env_key = f"FF_{flag}"
    if env_key in os.environ:
        raw = os.environ[env_key].lower()
        return raw in ("1", "true", "yes", "on")
    return bool(fallback) if fallback is not None else False

def snapshot() -> Dict[str, bool]:
    """Return the current state of every flag for the /flags endpoint."""
    return {name: get(name) for name in FLAGS}

def client_js_literal(query_string: str = "") -> str:
    """Render a JS literal the HTML page can embed so the browser reads the
    same flag set the server does. query_string is the raw URL query from
    the client; ?safe=1 or ?ff_foo=0 in the query overrides env/defaults
    for that session only (no server state mutation)."""
    from urllib.parse import parse_qs
    parsed = parse_qs(query_string) if query_string else {}
    safe_in_qs = any(
        v.lower() in ("1", "true", "yes", "on")
        for v in parsed.get("safe", [])
    )
    import json as _json
    out: Dict[str, bool] = {}
    for name in FLAGS:
        if safe_in_qs or _SAFE_MODE_ENV:
            out[name] = False
            continue
        qs_key = f"ff_{name.lower()}"
        if qs_key in parsed:
            out[name] = parsed[qs_key][0].lower() in ("1", "true", "yes", "on")
        else:
            out[name] = get(name)
    return _json.dumps(out)

def is_safe_mode() -> bool:
    return _SAFE_MODE_ENV

# ── CLI ──
if __name__ == "__main__":
    print("Dodo Vision AI v9.3 Feature Flags")
    print("=" * 50)
    for name, meta in FLAGS.items():
        state = "ON" if get(name) else "OFF"
        _step = meta["step"]
        _step_s = f"{_step:2d}" if isinstance(_step, int) else str(_step)
        print(f"  FF_{name:20s} step {_step_s} [{state}]  {meta['desc']}")
    if _SAFE_MODE_ENV:
        print("\n  SAFE_MODE is ACTIVE — all flags forced OFF.")
