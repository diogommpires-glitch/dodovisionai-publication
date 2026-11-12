"""pad/train_screen_flash.py — scaffold (data not yet recorded).

When data/screen_flash/{live,spoof}/*.mp4 exists, run:
    python -m pad.train_screen_flash --epochs 8 --out pad/models/screen_flash.pkl

Until then, screen_flash_branch.score emits reliability=0 (no model).
"""
import argparse
import sys
from pathlib import Path

DATA_LIVE = Path("data/screen_flash/live")
DATA_SPOOF = Path("data/screen_flash/spoof")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--out", default="pad/models/screen_flash.pkl")
    args = ap.parse_args()
    if not (DATA_LIVE.is_dir() and DATA_SPOOF.is_dir()):
        print("[screen_flash] data not present; see "
              "docs/BLOCKED_ROADMAP_ITEMS.md#12 for recording protocol")
        return 2
    # TODO: per-frame reflectance-delta GBT or small MLP.
    # Use pad/screen_flash_protocol.py to drive the flash sequence.
    print("[screen_flash] training scaffolded but not implemented yet")
    return 0


if __name__ == "__main__":
    sys.exit(main())
