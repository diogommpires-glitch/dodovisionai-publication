"""
MediaPipe Face Landmarker (478-point) topology constants and helpers.

v9.8.3 — single place for PAD / dynamics nose–ear motion assumptions so
call sites do not scatter magic indices. `analysis._ear_from_landmarks`
uses a separate minimum length (387) for EAR geometry; PAD nose–oval
anchors need the full oval set (indices up to 454).
"""
from __future__ import annotations

from typing import Any, Optional, Tuple

# EAR path in analysis.py requires at least these landmark slots.
MIN_LANDMARKS_FOR_EAR_MESH = 387

# 478-point topology (MediaPipe tasks) — nose tip + face-oval proxies for ears.
NOSE_TIP_IDX = 4
FACE_OVAL_LEFT_IDX = 234
FACE_OVAL_RIGHT_IDX = 454
# max(FACE_OVAL_RIGHT_IDX, ...) + 1
MIN_LANDMARKS_FOR_NOSE_OVAL_TRIO = 455

PixelPair = Tuple[float, float]
NoseEarPixels = Tuple[PixelPair, PixelPair, PixelPair]


def nose_ear_frame_pixels(
    fl: Any,
    frame_w: int,
    frame_h: int,
) -> Optional[NoseEarPixels]:
    """Return (nose_xy, left_ear_xy, right_ear_xy) in full-frame pixel coords.

    None if the landmark list is too short or indexing fails.
    """
    if fl is None:
        return None
    try:
        n = len(fl)
    except Exception:
        return None
    if n < MIN_LANDMARKS_FOR_NOSE_OVAL_TRIO:
        return None
    fw = float(frame_w)
    fh = float(frame_h)

    def _px(idx: int) -> PixelPair:
        p = fl[idx]
        return (float(p.x * fw), float(p.y * fh))

    try:
        return (
            _px(NOSE_TIP_IDX),
            _px(FACE_OVAL_LEFT_IDX),
            _px(FACE_OVAL_RIGHT_IDX),
        )
    except Exception:
        return None
