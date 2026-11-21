"""
DODO VISION AI v8 — Speaker Diarization (Skeleton)

Optional speaker identification via embeddings. Real implementations require:
  - pyannote.audio (for state-of-the-art ECAPA-TDNN)
  - speechbrain (for alternative speaker embeddings)

If both are missing, this module falls back to a naive energy+F0 clustering
on the voice_monitor metrics.

Public API:
    diarizer = Diarizer(backend="auto")
    diarizer.ingest(audio_float32, sample_rate)  # push 3-sec chunks
    speakers = diarizer.active_speakers()        # [{speaker_id, confidence, last_seen}]
    diarizer.attach_to_person(speaker_id, person_id)  # cross-link
"""
from __future__ import annotations
import time
from typing import List, Dict, Optional
from collections import deque

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore


class Diarizer:
    """Multi-speaker tracking with optional embeddings."""

    def __init__(self, backend: str = "auto") -> None:
        """
        Parameters
        ----------
        backend : str
            "auto" (prefer pyannote, fallback to speechbrain, then energy),
            "pyannote", "speechbrain", or "energy".
        """
        self.backend = backend
        self._speakers: Dict[str, Dict] = {}  # speaker_id -> {embedding, last_seen, confidence}
        self._speaker_person_map: Dict[str, str] = {}  # speaker_id -> person_id
        self._audio_buffer: deque = deque(maxlen=48000 * 3)  # ~3 sec at 16kHz
        self._sample_rate: Optional[int] = None
        # v9.4.1 step 4: per-speaker rolling audio buffers so the
        # pad.branches.audio_branch can score each diarized speaker
        # independently. Capped at ~3 s at 16 kHz per speaker.
        self._speaker_audio: Dict[str, deque] = {}
        self._SPK_AUDIO_MAX = 48000 * 3

        # Try to load embedding models
        self._pyannote_model = None
        self._pyannote_real: Optional[object] = None
        self._speechbrain_model = None
        self._use_embeddings = False

        if backend in ("auto", "pyannote"):
            try:
                import pyannote.audio  # noqa: F401 — only checking availability
                # v9.5.4: lazy real load gated behind FF_PYANNOTE so the
                # ~350 MB model + HF auth only happens when operators
                # explicitly opt in. On failure (no token, offline, etc.)
                # we fall back to the deterministic pseudo-embedding
                # stub — same behaviour as pre-v9.5.4.
                if self._maybe_load_real_pyannote():
                    # Real model loaded; mark so _update_speakers_embedding
                    # routes to it rather than the hash-energy proxy.
                    self._pyannote_model = "real"
                else:
                    self._pyannote_model = "stub"
                self._use_embeddings = True
            except ImportError:
                pass

        if not self._use_embeddings and backend in ("auto", "speechbrain"):
            try:
                from speechbrain.pretrained import SpeakerRecognition
                # Stub: self._speechbrain_model = SpeakerRecognition.from_hparams(...)
                self._speechbrain_model = "stub"
                self._use_embeddings = True
            except ImportError:
                pass

    def _maybe_load_real_pyannote(self) -> bool:
        """Attempt to load pyannote.audio 3.1 diarization pipeline.

        Gated by FF_PYANNOTE (defaults OFF) so the large download +
        HuggingFace auth token requirement are explicit opt-ins. Returns
        True if a real pipeline is attached, False on any failure (so
        the caller falls back to the stub).
        """
        # FF_PYANNOTE gate — env var override, defaults off.
        try:
            import feature_flags as _ff
            if not _ff.get("PYANNOTE"):
                return False
        except Exception:
            # feature_flags may not be importable in standalone unit
            # tests; fall back to env only.
            import os as _os
            if _os.environ.get("FF_PYANNOTE", "").lower() not in ("1", "true", "yes", "on"):
                return False

        import os as _os
        token = (_os.environ.get("HUGGING_FACE_HUB_TOKEN")
                 or _os.environ.get("HF_TOKEN"))
        if not token:
            # Real model is a gated HF artefact — no token = no load.
            return False

        try:
            from pyannote.audio import Pipeline
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=token,
            )
            # Optional GPU speedup — guard so CPU-only boxes still work.
            try:
                import torch
                if torch.cuda.is_available():
                    pipeline.to(torch.device("cuda"))
            except Exception:
                pass
            self._pyannote_real = pipeline
            return True
        except Exception as e:
            # Silent failure — operator can inspect by calling diarize
            # with FF_PYANNOTE=1 in a dev shell to see the exception.
            self._pyannote_real = None
            self._last_pyannote_err = repr(e)
            return False

    @property
    def has_embeddings(self) -> bool:
        """Whether a real embedding backend is available."""
        return self._use_embeddings

    def ingest(self, audio_float32: "np.ndarray", sr: int) -> None:
        """
        Push a chunk of audio (typically 3 seconds) for diarization.

        Parameters
        ----------
        audio_float32 : ndarray
            Audio samples in range [-1, 1].
        sr : int
            Sample rate (e.g., 16000 Hz).
        """
        if np is None:
            return
        self._sample_rate = sr
        # Append to rolling buffer
        for sample in audio_float32:
            self._audio_buffer.append(float(sample))
        # v8.2: if embedding backend is live, extract and cluster speaker embeddings
        if self._use_embeddings and len(self._audio_buffer) >= (sr or 16000):
            self._update_speakers_embedding()

    def active_speakers(self) -> List[Dict[str, object]]:
        """
        Return list of currently active speakers.

        Returns
        -------
        speakers : List[Dict]
            Each dict has:
              - speaker_id: str
              - confidence: float in [0, 1]
              - last_seen: float (timestamp)
        """
        now = time.time()
        # Prune stale speakers (not seen for >10 seconds)
        stale = [sid for sid, info in self._speakers.items()
                 if (now - info["last_seen"]) > 10.0]
        for sid in stale:
            del self._speakers[sid]

        return [
            {
                "speaker_id": sid,
                "confidence": info.get("confidence", 0.5),
                "last_seen": info.get("last_seen"),
            }
            for sid, info in self._speakers.items()
        ]

    def attach_to_person(self, speaker_id: str, person_id: str) -> None:
        """
        Link a detected speaker to a tracked person (for cross-modal fusion).

        Parameters
        ----------
        speaker_id : str
            Speaker cluster ID from active_speakers().
        person_id : str
            Person ID from face tracking.
        """
        self._speaker_person_map[speaker_id] = person_id

    def _update_speakers_embedding(self) -> None:
        """v8.2: extract speaker embeddings and cluster via cosine similarity.
        Uses pyannote or speechbrain backend when available."""
        if np is None or not self._use_embeddings:
            return
        sr = self._sample_rate or 16000
        if len(self._audio_buffer) < sr:
            return
        audio_arr = np.array(list(self._audio_buffer), dtype=np.float32)
        embedding = None
        # v9.5.4: real pyannote path first — skips the internal clustering
        # entirely because the pipeline already returns speaker labels.
        if self._pyannote_real is not None:
            self._update_speakers_pyannote_real(audio_arr, sr)
            return
        try:
            if self._pyannote_model is not None:
                # pyannote: extract d-vector from audio chunk
                # In production this would be: Pipeline("speaker-diarization-3.1")
                # For now, use a simple hash-based embedding proxy when stub
                if self._pyannote_model == "stub":
                    # Deterministic pseudo-embedding from audio energy profile
                    chunk_size = sr // 4
                    features = []
                    for i in range(0, len(audio_arr), chunk_size):
                        c = audio_arr[i:i+chunk_size]
                        if len(c) > 0:
                            features.append(float(np.sqrt(np.mean(c**2))))
                    if features:
                        embedding = np.array(features[:16] + [0.0] * max(0, 16 - len(features)))
                        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
                else:
                    # Real pyannote path handled above via
                    # _update_speakers_pyannote_real — this branch is
                    # unreachable now but kept for forward compat.
                    pass
            elif self._speechbrain_model is not None:
                if self._speechbrain_model == "stub":
                    chunk_size = sr // 4
                    features = []
                    for i in range(0, len(audio_arr), chunk_size):
                        c = audio_arr[i:i+chunk_size]
                        if len(c) > 0:
                            features.append(float(np.mean(np.abs(c))))
                    if features:
                        embedding = np.array(features[:16] + [0.0] * max(0, 16 - len(features)))
                        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
                else:
                    pass
        except Exception:
            return
        if embedding is None:
            return
        # Cosine similarity against cached speaker embeddings
        best_id = None
        best_sim = 0.6  # similarity threshold for same-speaker
        for sid, info in self._speakers.items():
            cached_emb = info.get("embedding")
            if cached_emb is not None:
                sim = float(np.dot(embedding, cached_emb))
                if sim > best_sim:
                    best_sim = sim
                    best_id = sid
        if best_id is not None:
            # Update existing speaker
            self._speakers[best_id]["last_seen"] = time.time()
            self._speakers[best_id]["confidence"] = min(1.0, best_sim)
            # Running average of embedding
            old_emb = self._speakers[best_id]["embedding"]
            self._speakers[best_id]["embedding"] = 0.8 * old_emb + 0.2 * embedding
        else:
            # New speaker
            new_id = f"spk_{len(self._speakers)}"
            self._speakers[new_id] = {
                "embedding": embedding,
                "confidence": 0.7,
                "last_seen": time.time(),
            }

    def _update_speakers_pyannote_real(self, audio_arr: "np.ndarray",
                                        sr: int) -> None:
        """Invoke the real pyannote 3.1 pipeline on the rolling buffer.

        The pipeline returns an `Annotation` with time-stamped speaker
        labels like ``SPEAKER_00``; we remap those to ``spk_<n>`` to
        stay compatible with the rest of the module, then update
        per-speaker audio buffers + last-seen metadata.
        """
        if np is None or self._pyannote_real is None:
            return
        try:
            import torch
            wav = torch.from_numpy(audio_arr).unsqueeze(0)  # (1, N)
            diarization = self._pyannote_real(
                {"waveform": wav, "sample_rate": int(sr)})
        except Exception as e:
            self._last_pyannote_err = repr(e)
            return

        now = time.time()
        # Iterate turns: each `segment` is a pyannote.core.Segment,
        # `_track` is an internal id, `label` is SPEAKER_{nn}.
        for segment, _track, label in diarization.itertracks(yield_label=True):
            spk_id = f"spk_{label.split('_')[-1]}" if isinstance(label, str) else "spk_0"
            start = max(0, int(segment.start * sr))
            end = min(len(audio_arr), int(segment.end * sr))
            if end <= start:
                continue
            slice_ = audio_arr[start:end]
            info = self._speakers.setdefault(spk_id, {
                "embedding": None,
                "confidence": 0.8,
                "last_seen": now,
            })
            info["last_seen"] = now
            info["confidence"] = 0.9  # real pipeline — trust more than the stub
            # Per-speaker audio buffer for downstream audio-branch scoring
            self.ingest_speaker_audio(spk_id, slice_)

    def _update_speakers_energy_fallback(self) -> None:
        """
        Fallback: naive energy-based speaker clustering.
        Divides audio into chunks, computes RMS energy and (stub) F0,
        clusters similar segments.
        """
        if np is None or len(self._audio_buffer) < 8000:  # <0.5 sec
            return
        audio_arr = np.array(list(self._audio_buffer), dtype=np.float32)
        # Chunk into ~1-sec windows
        chunk_size = (self._sample_rate or 16000)
        chunks = [
            audio_arr[i : i + chunk_size]
            for i in range(0, len(audio_arr), chunk_size)
        ]
        if not chunks:
            return
        # Compute energy per chunk
        energies = [float(np.sqrt(np.mean(c ** 2))) if len(c) > 0 else 0.0
                    for c in chunks]
        # Naive clustering: high energy -> speaker 1, low -> speaker 2
        threshold = (max(energies) + min(energies)) / 2 if energies else 0.5
        for i, nrg in enumerate(energies):
            speaker_id = "high_energy" if nrg > threshold else "low_energy"
            self._speakers[speaker_id] = {
                "confidence": min(1.0, nrg * 2),
                "last_seen": time.time(),
            }


    # ── v9.4.1 step 4: per-speaker audio buffers ─────────────────
    def ingest_speaker_audio(self, speaker_id: str,
                             audio_float32: "np.ndarray") -> None:
        """Append audio to a named speaker's rolling buffer. Used by the
        PAD audio branch to score each diarized speaker independently.
        """
        if np is None or speaker_id is None:
            return
        buf = self._speaker_audio.get(speaker_id)
        if buf is None:
            buf = deque(maxlen=self._SPK_AUDIO_MAX)
            self._speaker_audio[speaker_id] = buf
        for s in audio_float32:
            buf.append(float(s))

    def get_speaker_audio(self, speaker_id: str) -> Optional["np.ndarray"]:
        """Return the rolling audio array for a speaker, or None."""
        if np is None:
            return None
        buf = self._speaker_audio.get(speaker_id)
        if buf is None or len(buf) == 0:
            return None
        return np.asarray(list(buf), dtype=np.float32)


_singleton: "Diarizer | None" = None


def get_diarizer() -> Diarizer:
    global _singleton
    if _singleton is None:
        _singleton = Diarizer()
    return _singleton


if __name__ == "__main__":
    # Test stub
    dz = Diarizer()
    print(f"[OK] Diarizer instantiated (backend={dz.backend}, embeddings={dz._use_embeddings})")
