"""
DODO VISION AI v7.1 — Face Identity Persistence (Next Step #2)

Matches a live face crop against faces/registry.json (Facenet512
embeddings, averaged per person) and, on match, carries the previous
PersonState forward so per-person vitals, deception baseline, and
Dodo LLM message history survive a re-entry.

Lazy, best-effort, crash-safe: if deepface or the registry is
missing, every call returns (None, 0.0) and the caller proceeds
with a fresh PersonState.

Public API:
    identify(face_bgr) -> (name_or_None, similarity)
    cache_state(name, person_state)
    pop_cached_state(name) -> PersonState or None
"""
from __future__ import annotations
import json, os, threading, time
from typing import Optional, Tuple, Dict

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore

import config as cfg

_REGISTRY_CACHE: Dict[str, "np.ndarray"] = {}
_REGISTRY_MTIME: float = 0.0
_STATE_CACHE: Dict[str, Tuple[float, object]] = {}   # name -> (evict_at, PersonState)
_LOCK = threading.Lock()
_DEEPFACE_OK: Optional[bool] = None

STATE_TTL = 300.0          # drop a cached identity after 5 minutes
SIM_THRESHOLD = getattr(cfg, "FACE_RECOGNITION_THRESHOLD", 0.42)
REGISTRY_PATH = os.path.join(getattr(cfg, "FACES_DIR", "faces"), "registry.json")


def _load_registry() -> Dict[str, "np.ndarray"]:
    """Reload registry.json if it changed on disk."""
    global _REGISTRY_MTIME, _REGISTRY_CACHE
    if np is None or not os.path.exists(REGISTRY_PATH):
        return {}
    mt = os.path.getmtime(REGISTRY_PATH)
    if mt == _REGISTRY_MTIME and _REGISTRY_CACHE:
        return _REGISTRY_CACHE
    try:
        with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception:
        return _REGISTRY_CACHE
    out: Dict[str, "np.ndarray"] = {}
    for name, rec in raw.items():
        emb = rec.get("embedding") if isinstance(rec, dict) else rec
        if emb and isinstance(emb, list):
            v = np.asarray(emb, dtype=np.float32)
            n = np.linalg.norm(v)
            if n > 0:
                out[name] = v / n
    _REGISTRY_CACHE = out
    _REGISTRY_MTIME = mt
    return out


def _deepface_embed(face_bgr) -> Optional["np.ndarray"]:
    global _DEEPFACE_OK
    if _DEEPFACE_OK is False or np is None:
        return None
    try:
        from deepface import DeepFace
        _DEEPFACE_OK = True
    except Exception:
        _DEEPFACE_OK = False
        return None
    try:
        reps = DeepFace.represent(
            face_bgr,
            model_name="Facenet512",
            enforce_detection=False,
            detector_backend="skip",
        )
        if not reps:
            return None
        emb = np.asarray(reps[0]["embedding"], dtype=np.float32)
        n = np.linalg.norm(emb)
        return emb / n if n > 0 else None
    except Exception:
        return None


def identify(face_bgr) -> Tuple[Optional[str], float]:
    """Return (name, cosine_similarity) for a BGR face crop, or (None, 0.0)."""
    reg = _load_registry()
    if not reg or np is None:
        return (None, 0.0)
    emb = _deepface_embed(face_bgr)
    if emb is None:
        return (None, 0.0)
    best_name, best_sim = None, -1.0
    for name, ref in reg.items():
        sim = float(np.dot(emb, ref))
        if sim > best_sim:
            best_name, best_sim = name, sim
    if best_sim >= (1.0 - SIM_THRESHOLD):
        return (best_name, best_sim)
    return (None, best_sim)


def cache_state(name: str, state) -> None:
    """Park a PersonState keyed by identity so we can restore it on re-entry."""
    if not name:
        return
    with _LOCK:
        _STATE_CACHE[name] = (time.time() + STATE_TTL, state)


def pop_cached_state(name: str):
    """Retrieve and remove a cached PersonState. Returns None if absent/expired."""
    if not name:
        return None
    now = time.time()
    with _LOCK:
        for n in list(_STATE_CACHE):
            evict_at, _ = _STATE_CACHE[n]
            if evict_at < now:
                _STATE_CACHE.pop(n, None)
        item = _STATE_CACHE.pop(name, None)
    return item[1] if item else None


def registry_size() -> int:
    return len(_load_registry())
