"""
DODO VISION AI v9.4 — PAD branches.

Each submodule exports a `score(...)` callable that returns a
`BranchScore`. Branches are INDEPENDENT: no branch reads another
branch's output. Fusion is log-odds-weighted in `pad.fusion.fuse`.

Current roster:
  - rppg_branch      — pulse presence + inter-ROI phase coherence + refresh-band
  - moire_branch     — screen-replay moiré via DCT+LBP gradient-boosted tree
  - dynamics_branch  — 3D consistency + blink + vergence + pupil Bayes net
  - patch_depth_branch — patch-pool CNN + pseudo-depth (ONNX inference)
  - deepfake_branch  — Face X-ray blend boundary + 3DMM + PRNU
  - audio_branch     — ASVspoof-style RawNet2/AASIST on /audio/chunk
  - blink_gaze_branch — temporal liveness from blink rate + gaze stability
  - radar_presence_branch — 60 GHz radar chest-motion presence detector
"""
from pad.branches.rppg_branch import score as score_rppg
from pad.branches.moire_branch import score as score_moire
from pad.branches.dynamics_branch import score as score_dynamics
from pad.branches.patch_depth_branch import score as score_patch_depth
from pad.branches.deepfake_branch import score as score_deepfake
from pad.branches.audio_branch import score as score_audio
# v9.8.1 2026-05-12 (9th-pass deep audit): the live PAD pipeline calls
# both of these directly via their module paths in main.py, but they
# weren't in __all__ or top-level imports, so `from pad.branches import *`
# (used by some test/inspection scripts) silently got 6 branches not 8.
from pad.branches.blink_gaze_branch import score as score_blink_gaze
from pad.branches.radar_presence_branch import score as score_radar_presence

__all__ = [
    "score_rppg", "score_moire", "score_dynamics",
    "score_patch_depth", "score_deepfake", "score_audio",
    "score_blink_gaze", "score_radar_presence",
]
