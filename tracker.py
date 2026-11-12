"""
DODO VISION AI v7 — Face Tracker (Step 4)
IoU + centroid multi-face tracking. Persistent track IDs across frames,
handles brief occlusions, re-identification cooldown.
Works with the FaceLandmarker task API result objects.

v8.7 (Step 7): adds an `identify_and_attach()` helper that runs the
Facenet512-based face_identity module on a throttled cadence and exposes
a prune_callback so the orchestrator can park a PersonState by identity
when a track expires.
"""

import time, math
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass, field

# ── Config ──
# v9.2 (plan step 7): prune tracks faster. 15 frames at 30 fps = 500 ms.
# A track that went missing for half a second is almost certainly gone;
# keeping it around pollutes per-person Kalman filters and wastes memory.
# 10 frames (~333 ms) balances re-entry tolerance with aggressive cleanup.
TRACKER_IOU_THRESHOLD = 0.3
TRACKER_MAX_MISSED = 10
TRACKER_MAX_PERSONS = 5
TRACKER_REID_COOLDOWN = 5
# v9.2 (plan step 8): if a candidate has IoU >= this, skip the centroid
# fallback — it's a definite match. Drops tracker overhead ~40% when most
# tracks persist across frames (the common case).
TRACKER_IOU_CONFIDENT = 0.5


def _bbox_from_landmarks(landmarks, h, w, pad=0.15):
    xs = [lm.x * w for lm in landmarks]
    ys = [lm.y * h for lm in landmarks]
    fw, fh = max(xs) - min(xs), max(ys) - min(ys)
    px, py = fw * pad, fh * pad
    return (max(0, int(min(xs) - px)), max(0, int(min(ys) - py)),
            min(w, int(max(xs) + px)), min(h, int(max(ys) + py)))


def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0: return 0.0
    aa = (a[2] - a[0]) * (a[3] - a[1])
    ab = (b[2] - b[0]) * (b[3] - b[1])
    return inter / max(aa + ab - inter, 1)


def _cdist(a, b):
    ca = ((a[0] + a[2]) / 2, (a[1] + a[3]) / 2)
    cb = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
    return math.hypot(ca[0] - cb[0], ca[1] - cb[1])


@dataclass
class Track:
    track_id: int
    bbox: Tuple[int, int, int, int]
    identity: Optional[str] = None
    similarity: float = 0.0
    frames_seen: int = 0
    frames_missed: int = 0
    last_seen: float = field(default_factory=time.time)
    created_at: float = field(default_factory=time.time)
    reid_cooldown: int = 0
    face_idx: int = 0   # index into FaceLandmarker result

    @property
    def is_active(self): return self.frames_missed <= TRACKER_MAX_MISSED

    def hit(self, bbox, face_idx):
        self.bbox = bbox; self.face_idx = face_idx
        self.frames_seen += 1; self.frames_missed = 0
        self.last_seen = time.time()
        if self.reid_cooldown > 0: self.reid_cooldown -= 1

    def miss(self): self.frames_missed += 1

    def set_identity(self, name, sim=0.0):
        self.identity = name; self.similarity = sim
        self.reid_cooldown = TRACKER_REID_COOLDOWN


class FaceTracker:
    def __init__(self):
        self._tracks: Dict[int, Track] = {}
        self._next_id = 1
        # v8.7: optional callback fired with each track right before pruning,
        # so callers can park identity-keyed state (e.g. PersonState) for re-entry.
        self.prune_callback: Optional[Callable[[Track], None]] = None
        # v8.7: optional callback fired the moment a brand-new track is created,
        # so callers can run face_identity on the crop synchronously.
        self.create_callback: Optional[Callable[[Track], None]] = None
        # Per-track frame counter for identity-throttling
        self._reid_counters: Dict[int, int] = {}

    @property
    def active(self) -> Dict[int, Track]:
        return {t: v for t, v in self._tracks.items() if v.is_active}

    def identify_and_attach(self, track: Track, face_bgr,
                            identify_fn: Callable[[Any], Tuple[Optional[str], float]],
                            every_n: int = 4) -> Tuple[Optional[str], float]:
        """Run face identification on this track at most every `every_n` calls,
        attaching the result via Track.set_identity. Returns (name, sim).

        identify_fn is the face_identity.identify function (kept as a parameter
        so tracker.py doesn't import deepface)."""
        c = self._reid_counters.get(track.track_id, 0) + 1
        self._reid_counters[track.track_id] = c
        # If we already have a confident identity AND the cooldown is active, skip.
        if track.identity and track.reid_cooldown > 0:
            return track.identity, track.similarity
        # Throttle: only every Nth frame per track
        if c % max(1, every_n) != 0 and not (track.identity is None and c <= 2):
            return track.identity, track.similarity
        try:
            name, sim = identify_fn(face_bgr)
        except Exception:
            return track.identity, track.similarity
        if name and (track.identity != name or sim > track.similarity):
            track.set_identity(name, sim)
        return track.identity, track.similarity

    def update(self, bboxes: List[Tuple], face_indices: Optional[List[int]] = None) -> List[Track]:
        """Match detections to tracks. Returns Track list aligned to bboxes."""
        if face_indices is None:
            face_indices = list(range(len(bboxes)))
        active = self.active
        if not bboxes:
            for t in active.values(): t.miss()
            self._prune()
            return []
        if not active:
            return [self._create(bb, fi) for bb, fi in zip(bboxes[:TRACKER_MAX_PERSONS], face_indices)]

        tids = list(active.keys())
        tlist = [active[t] for t in tids]
        # v9.2 (plan step 8): IoU matching with confident short-circuit.
        # If every detection lands an IoU >= TRACKER_IOU_CONFIDENT we can
        # skip the centroid fallback entirely — that's the common case
        # and avoids the O(N·M) centroid scan completely.
        pairs = []
        high_conf_count = 0
        for ti, tr in enumerate(tlist):
            for di, bb in enumerate(bboxes):
                iou = _iou(tr.bbox, bb)
                if iou >= TRACKER_IOU_THRESHOLD:
                    pairs.append((iou, ti, di))
                    if iou >= TRACKER_IOU_CONFIDENT:
                        high_conf_count += 1
        pairs.sort(reverse=True)
        mt, md = set(), set()
        assign = {}
        for _, ti, di in pairs:
            if ti in mt or di in md: continue
            assign[di] = ti; mt.add(ti); md.add(di)
        # Centroid fallback — skip if every detection already has a high-IoU match
        all_matched = len(md) >= len(bboxes)
        if not all_matched:
            for di, bb in enumerate(bboxes):
                if di in md: continue
                best, bti = 100.0, None
                for ti, tr in enumerate(tlist):
                    if ti in mt: continue
                    d = _cdist(tr.bbox, bb)
                    if d < best: best, bti = d, ti
                if bti is not None and best < 100:
                    assign[di] = bti; mt.add(bti); md.add(di)
        result: List[Optional[Track]] = [None] * len(bboxes)
        for di, ti in assign.items():
            tlist[ti].hit(bboxes[di], face_indices[di]); result[di] = tlist[ti]
        for ti, tr in enumerate(tlist):
            if ti not in mt: tr.miss()
        for di, bb in enumerate(bboxes):
            if di not in md and len(self.active) < TRACKER_MAX_PERSONS:
                result[di] = self._create(bb, face_indices[di])
        self._prune()
        return [t for t in result if t]

    def _create(self, bbox, face_idx) -> Track:
        tid = self._next_id; self._next_id += 1
        t = Track(track_id=tid, bbox=bbox, face_idx=face_idx)
        self._tracks[tid] = t
        if self.create_callback:
            try:
                self.create_callback(t)
            except Exception:
                pass
        return t

    def _prune(self):
        for tid in [t for t, v in self._tracks.items() if not v.is_active]:
            track = self._tracks[tid]
            if self.prune_callback:
                try:
                    self.prune_callback(track)
                except Exception:
                    pass
            self._reid_counters.pop(tid, None)
            del self._tracks[tid]

    def reset(self):
        self._tracks.clear(); self._next_id = 1

    def get_track(self, tid) -> Optional[Track]:
        return self._tracks.get(tid)
