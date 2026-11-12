"""
DODO VISION AI v9.4 PAD Step 7 — Audio-deepfake branch.

Guards the three voice-based deception cues (vocal pitch, jitter,
speech hesitation) against TTS / voice-clone attacks.

Model options (first found wins):
  1. pad/models/asvspoof_rawnet2.onnx  — RawNet2 baseline
  2. pad/models/asvspoof_aasist.onnx   — AASIST (2024)
  3. vocoder-artifact head from Sun et al. CVPRW 2023

When no model is shipped we emit reliability=0.0 so fusion ignores
this branch and the deception pipeline keeps working on rPPG + text.

Runs on 2-second rolling windows per diarized speaker. Only emits
when VAD has latched `speech_probability > 0.7` for ≥ 500 ms so
silent chunks don't dilute the fused score.

Refs:
  - ASVspoof 2024 Workshop program.
  - Sun et al., CVPRW 2023 — Neural Vocoder Artifacts for detection.
"""
from __future__ import annotations
import os
import time
from typing import Optional, Dict

import numpy as np

from pad.types import BranchScore

_MOD_DIR = os.path.dirname(os.path.abspath(__file__))
# v9.8.1 2026-05-12 (9th-pass deep audit): the previous candidate list
# pointed at `asvspoof_rawnet2.onnx` / `asvspoof_aasist.onnx` /
# `asvspoof_vocoder.onnx` -- NONE of these filenames exist on disk.
# `fetch_audio_pad.py` and the pack4 trainer produce `audio_pad_*.onnx`.
# Result: when FF_AUDIO_NPU was unset the ORT loader hit the FileNotFound
# trap and the audio branch silently no-op'd (reliability=0 every frame).
# Updated to the actual filenames produced by the pipeline, with
# `audio_pad_adv.onnx` (v9.11 adversarial training) first since it's
# the most recent / best-AUC model on disk.
_MODEL_CANDIDATES = [
    os.path.join(os.path.dirname(_MOD_DIR), "models", "audio_pad_adv.onnx"),
    os.path.join(os.path.dirname(_MOD_DIR), "models", "audio_pad_rawnet2.onnx"),
    os.path.join(os.path.dirname(_MOD_DIR), "models", "audio_pad.onnx"),
]
# API_up v9.8: optional OpenVINO INT8 dispatch using the IR from
# `scripts/quantize_audio_npu.py`. Three modes:
#   FF_AUDIO_NPU=0     -> default: ORT-CUDA on FP32 ONNX (paper-cited, fastest)
#   FF_AUDIO_NPU=1|npu -> OpenVINO NPU (works but ~80 ms; AASIST attention is
#                         the bottleneck, see AASIST_NPU_ROADMAP.md)
#   FF_AUDIO_NPU=cpu   -> OpenVINO CPU on INT8 IR (~22 ms, 1.34x faster than
#                         FP32 CPU; no GPU contention with PAD pipeline)
_OV_IR_DIR = os.path.join(os.path.dirname(_MOD_DIR), "models", "audio_pad_int8_ir")
_FF_AUDIO_NPU_RAW = os.environ.get("FF_AUDIO_NPU", "0").strip().lower()
if _FF_AUDIO_NPU_RAW in ("1", "true", "yes", "npu"):
    _FF_AUDIO_NPU = "NPU"
elif _FF_AUDIO_NPU_RAW in ("cpu", "ov_cpu", "openvino_cpu"):
    _FF_AUDIO_NPU = "CPU"
else:
    _FF_AUDIO_NPU = ""

_SESSION = None
_SESSION_NAME: Optional[str] = None
_LOADED = False
_TARGET_SR = 16000

# v9.8: once-only error logging. The score() exception path used to
# print nothing (silent except status=-2.0) which made debugging hard,
# but printing every frame would flood the log when the model is broken
# (e.g., shape mismatch with a new ONNX). Track the first error per
# error class and only print on the transition.
_logged_errors: set = set()


class _OvSessionShim:
    """ORT-compatible thin wrapper around an OpenVINO compiled model so the
    rest of audio_branch.py keeps calling `session.get_inputs()` /
    `session.run(None, {name: x})` unchanged.
    Designed for NPU dispatch but works on any OV device.

    v9.8: built-in thermal watchdog. The NPU 4 in Arrow Lake-HX throttles
    when sustained-power exceeds the chassis limit, and the OV runtime
    silently extends inference latency 2-3x without raising. We track a
    rolling 60-second mean of inference latency; if `current >=
    NPU_THROTTLE_FACTOR * baseline`, we re-compile to CPU and stay
    there for the rest of the session. Self-healing on next process
    boot (the cache dir is reused so warm-up is fast).
    """
    NPU_THROTTLE_FACTOR = 2.0      # 2x baseline = throttling
    NPU_BASELINE_FRAMES = 50       # need this many before judging
    NPU_WINDOW_SECS     = 60.0     # rolling window for current mean

    def __init__(self, ir_dir: str, device: str = "NPU"):
        import openvino as ov
        self._ov = ov
        self._core = ov.Core()
        self._ir_dir = ir_dir
        self._core.set_property({"CACHE_DIR": os.path.join(ir_dir, "_cache")})
        m = self._core.read_model(os.path.join(ir_dir, "model.xml"))
        # Force static input shape (NPU requirement). audio_pad expects
        # (batch, 64600) -- 4 s @ 16 kHz waveform.
        in_name = m.input(0).any_name
        m.reshape({in_name: [1, 64600]})
        try:
            self._compiled = self._core.compile_model(m, device)
            self._device_used = device
        except Exception as e:
            # Fallback to CPU if NPU compile fails (e.g., op not supported)
            print(f"[PAD audio][NPU] compile failed on {device}, falling "
                  f"back to CPU: {str(e).splitlines()[0][:100]}")
            self._compiled = self._core.compile_model(m, "CPU")
            self._device_used = "CPU"
        self._req = self._compiled.create_infer_request()
        self._in_name = self._compiled.input(0).any_name
        # Watchdog state -- only meaningful when we're on NPU; CPU path
        # just collects telemetry without ever throttling.
        from collections import deque as _dq
        self._lat_history = _dq()    # (timestamp, latency_ms) tuples
        self._baseline_ms = None
        self._throttle_warned = False

    def get_inputs(self):
        # Mimic ORT's get_inputs() shape -- branch just reads .name
        class _I:
            def __init__(self, name):
                self.name = name
        return [_I(self._in_name)]

    def get_providers(self):
        return [f"OpenVINO-{self._device_used}"]

    def run(self, output_names, feeds):
        # ORT contract: run(None or output_names, {input_name: ndarray}) -> list of ndarrays
        x = list(feeds.values())[0]
        # Reshape to static [1, 64600] if needed
        if x.ndim == 1:
            x = x[None, :]
        if x.shape != (1, 64600):
            # Truncate/pad to match the NPU's static shape
            n = x.shape[-1]
            if n > 64600:
                x = x[..., :64600]
            elif n < 64600:
                pad = np.zeros((1, 64600), dtype=x.dtype)
                pad[:, :n] = x
                x = pad
        x = np.ascontiguousarray(x.astype(np.float32))
        # v9.8: time the inference for the watchdog.
        import time as _time
        _t0 = _time.perf_counter()
        self._req.set_tensor(self._compiled.input(0), self._ov.Tensor(x))
        self._req.infer()
        _lat_ms = (_time.perf_counter() - _t0) * 1000.0
        self._maybe_throttle_check(_lat_ms)
        # Return all outputs in declaration order
        outs = []
        for o in self._compiled.outputs:
            outs.append(self._req.get_tensor(o).data.copy())
        return outs

    def _maybe_throttle_check(self, lat_ms: float) -> None:
        """v9.8: NPU thermal-throttle watchdog. Silently swap the
        compiled model to CPU if sustained latency over the last 60 s
        exceeds NPU_THROTTLE_FACTOR x baseline. Only acts on NPU --
        CPU is the fallback target so it can't be throttled away from.
        """
        import time as _time
        now = _time.monotonic()
        self._lat_history.append((now, lat_ms))
        # Drop old samples
        cutoff = now - self.NPU_WINDOW_SECS
        while self._lat_history and self._lat_history[0][0] < cutoff:
            self._lat_history.popleft()
        # Establish baseline (median of first NPU_BASELINE_FRAMES)
        if self._baseline_ms is None and len(self._lat_history) >= self.NPU_BASELINE_FRAMES:
            firsts = sorted(s[1] for s in list(self._lat_history)[:self.NPU_BASELINE_FRAMES])
            self._baseline_ms = firsts[len(firsts) // 2]
            print(f"[PAD audio][watchdog] {self._device_used} baseline = "
                  f"{self._baseline_ms:.1f} ms (over {self.NPU_BASELINE_FRAMES} samples)")
        # Throttle check (NPU only -- CPU path can't fall further back)
        if (self._device_used == "NPU"
                and self._baseline_ms is not None
                and not self._throttle_warned
                and len(self._lat_history) >= 30):
            recent = [s[1] for s in list(self._lat_history)[-30:]]
            recent_med = sorted(recent)[len(recent) // 2]
            if recent_med >= self.NPU_THROTTLE_FACTOR * self._baseline_ms:
                self._throttle_warned = True
                print(f"[PAD audio][watchdog] NPU throttle detected: "
                      f"recent median {recent_med:.0f} ms vs baseline "
                      f"{self._baseline_ms:.0f} ms ({recent_med/self._baseline_ms:.1f}x). "
                      f"Falling back to CPU for the rest of the session.")
                try:
                    m = self._core.read_model(
                        os.path.join(self._ir_dir, "model.xml"))
                    in_name = m.input(0).any_name
                    m.reshape({in_name: [1, 64600]})
                    self._compiled = self._core.compile_model(m, "CPU")
                    self._req = self._compiled.create_infer_request()
                    self._device_used = "CPU"
                    print("[PAD audio][watchdog] re-compiled to CPU successfully")
                except Exception as e:
                    print(f"[PAD audio][watchdog] CPU re-compile failed: {e}")


def _load_session():
    global _SESSION, _SESSION_NAME, _LOADED
    if _LOADED:
        return _SESSION
    _LOADED = True

    # API_up v9.8: OpenVINO INT8 shortcut. If the IR is built and the flag
    # is "npu" or "cpu", use the INT8 OV model instead of FP32 ONNX. Falls
    # through silently to the CUDA/CPU ORT path if anything goes wrong.
    if _FF_AUDIO_NPU and os.path.isdir(_OV_IR_DIR):
        try:
            _SESSION = _OvSessionShim(_OV_IR_DIR, device=_FF_AUDIO_NPU)
            _SESSION_NAME = "audio_pad_int8_ir"
            mode_label = "NPU" if _FF_AUDIO_NPU == "NPU" else "INT8-CPU"
            print(f"[PAD audio] loaded INT8 IR: {_SESSION.get_providers()} "
                  f"(FF_AUDIO_NPU={_FF_AUDIO_NPU_RAW}, device={_SESSION._device_used})")
            return _SESSION
        except Exception as e:
            print(f"[PAD audio][OV] shim init failed, falling back to ONNX: "
                  f"{str(e).splitlines()[0][:150]}")

    for path in _MODEL_CANDIDATES:
        if os.path.exists(path):
            try:
                import onnxruntime as ort
                provs = [p for p in ort.get_available_providers()
                         if p in ("DmlExecutionProvider", "CUDAExecutionProvider")]
                provs = provs + ["CPUExecutionProvider"] if provs else ["CPUExecutionProvider"]
                _SESSION = ort.InferenceSession(path, providers=provs)
                _SESSION_NAME = os.path.basename(path)
                print(f"[PAD audio] loaded {_SESSION_NAME}: {_SESSION.get_providers()}")
                return _SESSION
            except Exception as e:
                print(f"[PAD audio] load failed for {path}: {e}")
    print("[PAD audio] no ASVspoof ONNX found — run pad/fetch_audio_pad.py")
    _SESSION = None
    return None


def _resample_to_target(audio: np.ndarray, sr: int) -> Optional[np.ndarray]:
    if sr == _TARGET_SR:
        return audio
    try:
        import librosa
        return librosa.resample(audio, orig_sr=sr, target_sr=_TARGET_SR)
    except Exception:
        # Very crude decimation fallback
        if sr <= 0:
            return None
        ratio = sr / _TARGET_SR
        if ratio < 1:
            return None
        idx = (np.arange(0, len(audio) / ratio).astype(int))
        idx = idx[idx < len(audio)]
        return audio[idx]


def _center_2s_window(audio: np.ndarray) -> np.ndarray:
    """ASVspoof expects a 2-s input. Trim or pad."""
    want = _TARGET_SR * 2
    if audio.size == want:
        return audio
    if audio.size > want:
        start = (audio.size - want) // 2
        return audio[start:start + want]
    # Pad with silence
    pad = want - audio.size
    return np.concatenate([audio, np.zeros(pad, dtype=audio.dtype)])


def score(audio: Optional[np.ndarray], sr: int = 16000,
          speech_probability: float = 1.0,
          speech_frames: int = 100) -> BranchScore:
    """Score a 2-s audio chunk.

    Args:
        audio: float32 [-1, 1] PCM, mono.
        sr: sample rate of the input (will be resampled to 16 kHz).
        speech_probability: VAD output in [0, 1].
        speech_frames: number of VAD frames with speech in this window;
                       used to require ≥ 500 ms of speech.
    """
    t0 = time.monotonic()
    # Require ≥ 0.5 s of speech at ≥ 0.7 probability to emit.
    min_speech_frames = int(0.5 * 100)  # assuming VAD hop 10 ms
    if (audio is None or audio.size < _TARGET_SR
            or speech_probability < 0.7
            or speech_frames < min_speech_frames):
        return BranchScore(
            name="audio",
            p_attack=0.5,
            reliability=0.0,
            latency_ms=(time.monotonic() - t0) * 1000.0,
            meta={"reason": 0.0},
        )
    sess = _load_session()
    if sess is None:
        return BranchScore(
            name="audio",
            p_attack=0.5,
            reliability=0.0,
            latency_ms=(time.monotonic() - t0) * 1000.0,
            meta={"status": 0.0},
        )
    try:
        x = _resample_to_target(audio.astype(np.float32), sr)
        if x is None or x.size < _TARGET_SR:
            return BranchScore(
                name="audio",
                p_attack=0.5,
                reliability=0.0,
                latency_ms=(time.monotonic() - t0) * 1000.0,
                meta={"status": -1.0},
            )
        x = _center_2s_window(x)
        # RawNet2 / AASIST both accept [1, N] raw PCM input.
        model_input = sess.get_inputs()[0]
        x_in = x[None, :].astype(np.float32)
        outs = sess.run(None, {model_input.name: x_in})
        logits = np.asarray(outs[0]).reshape(-1)
        # Model output is commonly [live_logit, spoof_logit]; take softmax.
        if logits.size >= 2:
            ex = np.exp(logits - logits.max())
            probs = ex / ex.sum()
            p_attack = float(probs[1])
        else:
            p_attack = float(1.0 / (1.0 + np.exp(-logits[0])))
    except Exception as e:
        # v9.8: log once per distinct error so the user sees broken-model
        # situations on first occurrence without flooding the log on
        # every subsequent frame.
        ek = f"{type(e).__name__}:{str(e).splitlines()[0][:80]}"
        if ek not in _logged_errors:
            _logged_errors.add(ek)
            print(f"[PAD audio] inference error (will be silent on repeats): {ek}")
        return BranchScore(
            name="audio",
            p_attack=0.5,
            reliability=0.0,
            latency_ms=(time.monotonic() - t0) * 1000.0,
            meta={"status": -2.0},
        )
    return BranchScore(
        name="audio",
        p_attack=max(0.0, min(1.0, p_attack)),
        reliability=float(min(1.0, 0.6 + 0.4 * speech_probability)),
        latency_ms=(time.monotonic() - t0) * 1000.0,
        meta={"sr": float(sr), "speech_prob": speech_probability},
        # v9.7.3: raised from 0.6 to 0.80 to match the moire/deepfake
        # branch tag thresholds — borderline 0.6-0.8 predictions no
        # longer label themselves AUDIO_DEEPFAKE in the panel.
        dominant_class="AUDIO_DEEPFAKE" if p_attack > 0.80 else None,
    )
