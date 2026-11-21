"""
DODO VISION AI v9.5 — FAISS-backed per-person memory

Provides a lightweight retrieval-augmented context for the LLM:
each tracked person gets their own IndexFlatIP (cosine similarity
after normalisation). At inference, the last N seconds of transcribed
speech + affect summary are encoded by a small Sentence-Transformer
and inserted into the index. When the LLM is called, we pull the top-k
most relevant past snippets as additional context instead of relying
on a sliding window that forgets.

Memory is PER PERSON (same `PersonState` isolation contract as
rPPG/pad_state). Nothing leaves disk; FAISS is in-process.

Public API:
    m = PersonMemory(pid)
    m.add(text, ts=None, tag="voice")
    ctx = m.retrieve(query, k=3)      # -> list of (text, score, tag, ts)
    m.save(path)                      # persist the FAISS index + sidecar
    m.load(path)                      # restore

Install (one-time):
    pip install faiss-cpu sentence-transformers

Sentence-Transformer default: `all-MiniLM-L6-v2` (22 MB, 384-D
embeddings). Falls back to hash-based sparse embedding if neither
library is available — retrieval quality drops but the API works.

Reference: Johnson, Douze, Jégou 2019 (FAISS).
"""
from __future__ import annotations
import os
import json
import time
import hashlib
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple, Dict

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore


_MODEL = None
_MODEL_LOADED = False
_DIM = 384  # matches all-MiniLM-L6-v2 default


def _lazy_encoder():
    """Load the sentence-transformer once. Falls back to hash embeddings."""
    global _MODEL, _MODEL_LOADED
    if _MODEL_LOADED:
        return _MODEL
    _MODEL_LOADED = True
    try:
        from sentence_transformers import SentenceTransformer
        _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        print("[memory-faiss] all-MiniLM-L6-v2 loaded (384-D)")
    except Exception as e:
        print(f"[memory-faiss] sentence-transformers unavailable ({e}); "
              "using hash-based fallback embeddings")
        _MODEL = None
    return _MODEL


def _hash_embed(text: str, dim: int = _DIM) -> "np.ndarray":
    """Cheap 384-D feature hash. Zero-dependency fallback. Quality is
    poor but adequate for debugging and tests.
    """
    h = hashlib.sha256(text.lower().encode("utf-8")).digest()
    # Expand the 32-byte hash into a dim-vector by tiling + sign flip
    chunk = np.frombuffer(h, dtype=np.uint8).astype(np.float32)
    v = np.tile(chunk, (dim // chunk.size) + 1)[:dim]
    v = (v - 127.5) / 127.5
    # L2 normalise
    nrm = np.linalg.norm(v) or 1.0
    return v / nrm


def _encode(text: str) -> "np.ndarray":
    m = _lazy_encoder()
    if m is None:
        return _hash_embed(text)
    try:
        vec = m.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return vec.astype(np.float32)
    except Exception:
        return _hash_embed(text)


@dataclass
class MemoryEntry:
    text: str
    ts: float
    tag: str = ""


class PersonMemory:
    """Per-person FAISS index + metadata sidecar.

    Designed for bounded growth — at `max_entries`, drops the oldest
    entries so long sessions don't leak RAM.
    """
    def __init__(self, pid, dim: int = _DIM, max_entries: int = 1024):
        self.pid = pid
        self.dim = dim
        self.max_entries = max_entries
        self._entries: List[MemoryEntry] = []
        self._index = None
        self._lazy_init_index()

    def _lazy_init_index(self):
        try:
            import faiss
            self._index = faiss.IndexFlatIP(self.dim)
        except ImportError:
            self._index = None  # fall back to linear scan

    def add(self, text: str, ts: Optional[float] = None, tag: str = "") -> None:
        if not text or not text.strip():
            return
        if ts is None:
            ts = time.time()
        vec = _encode(text).astype(np.float32)
        self._entries.append(MemoryEntry(text=text, ts=ts, tag=tag))
        if self._index is not None:
            self._index.add(vec[None, :])
        # Bound memory
        while len(self._entries) > self.max_entries:
            # Drop-first. For FAISS IndexFlatIP we can't cheaply drop one
            # entry so we rebuild when the ring fills.
            self._entries.pop(0)
            if self._index is not None:
                try:
                    import faiss
                    self._index = faiss.IndexFlatIP(self.dim)
                    if self._entries:
                        mat = np.stack([_encode(e.text) for e in self._entries])
                        self._index.add(mat.astype(np.float32))
                except Exception:
                    self._index = None

    def retrieve(self, query: str, k: int = 3) -> List[Tuple[str, float, str, float]]:
        if not self._entries:
            return []
        k = max(1, min(k, len(self._entries)))
        qv = _encode(query).astype(np.float32)[None, :]
        if self._index is not None:
            try:
                scores, idxs = self._index.search(qv, k)
                out = []
                for rank in range(k):
                    i = int(idxs[0, rank])
                    if i < 0 or i >= len(self._entries):
                        continue
                    s = float(scores[0, rank])
                    e = self._entries[i]
                    out.append((e.text, s, e.tag, e.ts))
                return out
            except Exception:
                pass
        # Fallback: linear cosine
        mat = np.stack([_encode(e.text) for e in self._entries])
        sims = (mat @ qv[0]).astype(np.float32)
        top = np.argsort(-sims)[:k]
        return [(self._entries[i].text, float(sims[i]),
                 self._entries[i].tag, self._entries[i].ts) for i in top]

    def save(self, path: str) -> None:
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        data = {
            "pid": str(self.pid),
            "dim": self.dim,
            "max_entries": self.max_entries,
            "entries": [asdict(e) for e in self._entries],
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh)

    def load(self, path: str) -> None:
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        self.dim = data.get("dim", self.dim)
        self.max_entries = data.get("max_entries", self.max_entries)
        self._entries = [MemoryEntry(**e) for e in data.get("entries", [])]
        # Rebuild index
        self._lazy_init_index()
        if self._index is not None and self._entries:
            mat = np.stack([_encode(e.text) for e in self._entries])
            self._index.add(mat.astype(np.float32))


# Global registry keyed by pid so main.py can grab a memory per person
_REGISTRY: Dict[str, PersonMemory] = {}


def get_memory(pid) -> PersonMemory:
    key = str(pid)
    m = _REGISTRY.get(key)
    if m is None:
        m = PersonMemory(pid)
        _REGISTRY[key] = m
    return m


def reset_all() -> None:
    _REGISTRY.clear()
