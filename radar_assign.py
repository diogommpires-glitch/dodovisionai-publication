"""
DODO VISION AI v7.1 — Spatial Radar-to-Person Assignment (Next Step #3)

Replaces the old positional hack (radar A -> person 0, radar B -> person 1)
with a cost-minimizing assignment over (radar distance, face depth proxy).

Face depth proxy: inverse of bounding-box area normalized to frame area.
The bigger the face on screen, the closer it is to the camera, which
correlates with lower radar distance.

Mapping is cached for CACHE_SECS to prevent flapping when two people
stand at similar distances.
"""
from __future__ import annotations
import time
from typing import Dict, List, Optional, Tuple

CACHE_SECS = 3.0

_last_mapping: Dict[str, int] = {}    # radar_key -> person_idx
_last_mapping_t: float = 0.0


def _face_depth_proxy(bbox: Tuple[int, int, int, int], frame_wh: Tuple[int, int]) -> float:
    """Return a pseudo-distance in metres. Larger face -> closer -> smaller number."""
    w = max(1, bbox[2] - bbox[0])
    h = max(1, bbox[3] - bbox[1])
    fw, fh = frame_wh
    face_frac = (w * h) / max(1, fw * fh)
    # Empirical: a face filling 25% of the frame is ~0.5m away;
    # a 2% face is ~3m away. Simple inverse-sqrt interpolation.
    if face_frac <= 0:
        return 5.0
    return max(0.3, min(5.0, 0.25 / (face_frac ** 0.5)))


def assign(
    radar_readings: Dict[str, Optional[dict]],
    face_bboxes: List[Tuple[int, int, int, int]],
    frame_wh: Tuple[int, int],
) -> Dict[str, int]:
    """
    Parameters
    ----------
    radar_readings : e.g. {"A": {"distance": 0.8, ...}, "B": {"distance": 1.5, ...}}
                     Values may be None for disconnected radars.
    face_bboxes    : list of (x1,y1,x2,y2) in frame pixels, aligned to person indices.
    frame_wh       : (width, height) of the analyzed frame.

    Returns
    -------
    mapping : {"A": person_idx, "B": person_idx}, omitting radars that
              couldn't be assigned (no reading, or no faces visible).
    """
    global _last_mapping, _last_mapping_t

    if not face_bboxes or not radar_readings:
        return {}

    # Use cache if fresh and still valid (all mapped persons still present)
    if (time.time() - _last_mapping_t) < CACHE_SECS and _last_mapping:
        if all(0 <= idx < len(face_bboxes) for idx in _last_mapping.values()):
            return dict(_last_mapping)

    # Build depth proxy per face
    face_depths = [_face_depth_proxy(bb, frame_wh) for bb in face_bboxes]

    # Cost matrix: |radar_dist - face_depth|
    radar_keys = [k for k, v in radar_readings.items()
                  if v and v.get("distance") is not None]
    if not radar_keys:
        return {}

    # Greedy minimum-cost assignment (small N, Hungarian is overkill)
    costs: List[Tuple[float, str, int]] = []
    for k in radar_keys:
        rd = float(radar_readings[k]["distance"])
        for pi, fd in enumerate(face_depths):
            costs.append((abs(rd - fd), k, pi))
    costs.sort()
    used_r: set = set()
    used_p: set = set()
    mapping: Dict[str, int] = {}
    for _cost, k, pi in costs:
        if k in used_r or pi in used_p:
            continue
        mapping[k] = pi
        used_r.add(k); used_p.add(pi)

    _last_mapping = mapping
    _last_mapping_t = time.time()
    return mapping


def invalidate_cache() -> None:
    global _last_mapping, _last_mapping_t
    _last_mapping = {}
    _last_mapping_t = 0.0
