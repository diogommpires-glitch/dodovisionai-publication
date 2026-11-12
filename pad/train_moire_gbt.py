"""train_moire_gbt.py
===================

Trains (and caches) a LightGBM classifier that flags moire patterns in face
images. Moire shows up when someone points a camera at a screen, so it's a
cheap but effective replay-attack detector.

Legacy 20-D features (`extract_features`):
  - radial FFT energy in 4 annular bins (replay moire lives 0.2-0.35 of Nyquist)
  - spectral peak count, peak prominence statistics
  - Laplacian variance (blur)
  - color channel stats
  - wavelet LH/HL sub-band energy ratios

API_unq enhanced 374-D features (`extract_features_enhanced`):
  - legacy 20-D
  - + uniform LBP^u2_{8,1} histograms (59 bins) across 6 channels
    {H,S,V,Y,Cb,Cr} = 354 features
  - Rationale: per the 2026 PAD survey, multi-color-space LBP^u2 captures
    pixel-grid texture independently of high-frequency Moiré aliasing.
    Closes the OLPF-blurred phone-replay false-negative gap (R10 + EOS
    Webcam Utility hardware-blurs moire below the FFT detection band).

Usage:
    python -m pad.train_moire_gbt                              # train from fixtures, 20-D legacy
    python -m pad.train_moire_gbt --feature-mode enhanced     # 374-D LBP-augmented (default in API_unq)
    python -m pad.train_moire_gbt --records <jsonl> ...        # train from a meta-records jsonl
    python -m pad.train_moire_gbt --force                      # retrain
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np

from . import MODEL_DIR, PAD_ROOT

OUT = MODEL_DIR / "moire_gbt.pkl"
DEFAULT_FIXTURES = PAD_ROOT.parent / "tests" / "fixtures"


# ----------------------------------------------------------------------
# feature extraction
# ----------------------------------------------------------------------
def _radial_fft_energy(gray: np.ndarray, bins: int = 4) -> np.ndarray:
    f = np.fft.fftshift(np.fft.fft2(gray.astype(np.float32)))
    mag = np.log1p(np.abs(f))
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    y, x = np.indices(mag.shape)
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    r_max = float(r.max())
    out = np.zeros(bins, dtype=np.float32)
    for i in range(bins):
        lo = i * r_max / bins
        hi = (i + 1) * r_max / bins
        mask = (r >= lo) & (r < hi)
        out[i] = mag[mask].mean() if mask.any() else 0.0
    return out


def _spectral_peaks(gray: np.ndarray) -> np.ndarray:
    from scipy.signal import find_peaks
    # 1D radial profile
    f = np.fft.fftshift(np.fft.fft2(gray.astype(np.float32)))
    mag = np.abs(f)
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    y, x = np.indices(mag.shape)
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(np.int32)
    tbin = np.bincount(r.ravel(), mag.ravel())
    nr = np.bincount(r.ravel())
    profile = tbin / np.maximum(nr, 1)
    profile = profile[: len(profile) // 2]
    peaks, props = find_peaks(profile, prominence=profile.mean())
    n = len(peaks)
    prom = props["prominences"]
    return np.array([
        n,
        prom.mean() if n else 0.0,
        prom.max() if n else 0.0,
        prom.std() if n else 0.0,
    ], dtype=np.float32)


def _color_and_blur(img_rgb: np.ndarray) -> np.ndarray:
    import cv2
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F).var()
    means = img_rgb.reshape(-1, 3).mean(axis=0)
    stds = img_rgb.reshape(-1, 3).std(axis=0)
    return np.concatenate([[lap], means, stds]).astype(np.float32)  # 7-D


def _wavelet_bands(gray: np.ndarray) -> np.ndarray:
    try:
        import pywt
    except ImportError:
        return np.zeros(5, dtype=np.float32)  # graceful degrade
    coeffs = pywt.wavedec2(gray.astype(np.float32), "db2", level=2)
    # coeffs = [cA2, (cH2,cV2,cD2), (cH1,cV1,cD1)]
    _, (h2, v2, d2), (h1, v1, d1) = coeffs
    e = lambda a: float(np.mean(np.abs(a)))
    return np.array([e(h1), e(v1), e(h2), e(v2), e(d1 + d2)], dtype=np.float32)


def extract_features(img_rgb: np.ndarray) -> np.ndarray:
    """20-D legacy feature vector for a single RGB image."""
    import cv2
    if img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
        raise ValueError("expected HxWx3 RGB uint8")
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    feats = np.concatenate([
        _radial_fft_energy(gray, bins=4),   # 4
        _spectral_peaks(gray),              # 4
        _color_and_blur(img_rgb),           # 7
        _wavelet_bands(gray),               # 5
    ])
    assert feats.shape == (20,), feats.shape
    return feats


# ----------------------------------------------------------------------
# API_unq: HSV/YCbCr LBP^u2_{8,1} feature extension (354-D)
# ----------------------------------------------------------------------
_LBP_BINS_PER_CHANNEL = 59  # nri_uniform LBP for P=8 → 58 uniform + 1 non-uniform
_LBP_CHANNEL_ORDER = ("H", "S", "V", "Y", "Cb", "Cr")  # 6 channels


def _nri_uniform_lbp_hist(channel_u8: np.ndarray) -> np.ndarray:
    """Non-rotation-invariant uniform LBP^{u2}_{8,1} normalized histogram.

    Returns a 59-bin probability mass vector. Falls back to zeros when
    skimage is unavailable (graceful degrade — caller can detect via
    sum-near-zero check).
    """
    try:
        from skimage.feature import local_binary_pattern
    except Exception:
        return np.zeros(_LBP_BINS_PER_CHANNEL, dtype=np.float32)
    lbp = local_binary_pattern(channel_u8, P=8, R=1, method="nri_uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=_LBP_BINS_PER_CHANNEL,
                           range=(0, _LBP_BINS_PER_CHANNEL))
    hist = hist.astype(np.float32)
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist


def _lbp_color_histograms(img_rgb: np.ndarray) -> np.ndarray:
    """Concatenated uniform-LBP histograms across HSV + YCbCr channels.

    Returns 354-D (6 channels × 59 bins). Captures pixel-grid texture
    in color spaces where moiré is most discriminative even when the
    sensor's OLPF has blurred high-frequency aliasing out of FFT range.
    """
    import cv2
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    channels = [
        hsv[:, :, 0],  # H (0-179 in OpenCV — rescale to 0-255 for LBP stability)
        hsv[:, :, 1],  # S
        hsv[:, :, 2],  # V
        ycc[:, :, 0],  # Y
        ycc[:, :, 2],  # Cb (OpenCV YCrCb order: Y, Cr, Cb — index 2 is Cb)
        ycc[:, :, 1],  # Cr
    ]
    # Rescale H to full uint8 range so LBP threshold operates consistently.
    channels[0] = np.clip(channels[0].astype(np.float32) * (255.0 / 179.0), 0, 255).astype(np.uint8)
    hists = [_nri_uniform_lbp_hist(c) for c in channels]
    return np.concatenate(hists).astype(np.float32)


def _rolling_shutter_band_score(img_rgb: np.ndarray) -> float:
    """Single-frame rolling-shutter banding score.

    API_unq 2026-05-13 (P1.4): the canonical screen-replay signature
    that ACTUALLY survives at 30 fps front-facing capture is rolling-
    shutter banding -- the CMOS row-sequential readout aliases the
    display's PWM/refresh waveform into horizontal stripes WITHIN a
    single frame. This shows up as a peak in the vertical-axis (row-
    summed) FFT of the cropped face region.

    Per Patel, Han, Jain & Ott (ICB 2015) + Patel, Han & Jain
    (IEEE TIFS 11(10), 2016) section IV-A, this is distinct from
    spatial Moiré (LCD pixel grid beating against Bayer grid -- which
    the LBP features capture). The two signals stack.

    Implementation: sum each row of grayscale → 1D column profile →
    rFFT → drop the first 3 bins (DC + ultra-low) → return peak /
    total energy ratio. Score in [0, 1]; higher = more banding.
    """
    import cv2
    if img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
        return 0.0
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    if gray.shape[0] < 16:
        return 0.0
    col = gray.mean(axis=1)
    col -= col.mean()
    if col.size < 8:
        return 0.0
    spec = np.abs(np.fft.rfft(col)) ** 2
    if spec.size <= 3:
        return 0.0
    spec[:3] = 0.0
    peak = float(spec.max())
    total = float(spec.sum()) + 1e-8
    return peak / total


def extract_features_enhanced(img_rgb: np.ndarray) -> np.ndarray:
    """375-D enhanced feature vector:
        - legacy 20-D
        - 6-channel uniform LBP (354-D HSV/YCbCr)
        - rolling-shutter banding score (1-D, API_unq P1.4)

    Use this when the trained model's metadata reports feature_mode='enhanced'
    (LightGBM bundle version >= 2). The branch loader in pad/branches/
    moire_branch.py dispatches on the metadata flag.
    """
    legacy = extract_features(img_rgb)
    lbp = _lbp_color_histograms(img_rgb)
    rolling_shutter = np.array([_rolling_shutter_band_score(img_rgb)],
                               dtype=np.float32)
    out = np.concatenate([legacy, lbp, rolling_shutter]).astype(np.float32)
    assert out.shape == (375,), out.shape
    return out


def extract_features_for_mode(img_rgb: np.ndarray, mode: str) -> np.ndarray:
    """Dispatcher: 'legacy' → 20-D, 'enhanced' → 374-D."""
    if mode == "enhanced":
        return extract_features_enhanced(img_rgb)
    return extract_features(img_rgb)


# ----------------------------------------------------------------------
# synthetic data if no fixtures supplied
# ----------------------------------------------------------------------
def _synth_clean(n: int, rng: np.random.Generator, size: int = 128) -> np.ndarray:
    """Synthesize clean-ish face-like images: low-freq blobs + small noise."""
    out = np.zeros((n, size, size, 3), dtype=np.uint8)
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    for i in range(n):
        img = np.zeros((size, size, 3), dtype=np.float32)
        for _ in range(rng.integers(3, 7)):
            cx, cy = rng.uniform(0, size), rng.uniform(0, size)
            sigma = rng.uniform(size / 8, size / 3)
            color = rng.uniform(60, 220, size=3)
            g = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma * sigma))
            img += g[..., None] * color
        img += rng.normal(0, 2.0, img.shape)
        out[i] = np.clip(img, 0, 255).astype(np.uint8)
    return out


def _add_moire(imgs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Overlay sinusoidal interference: the canonical moire pattern."""
    out = imgs.astype(np.float32).copy()
    _, h, w, _ = out.shape
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    for i in range(len(out)):
        theta = rng.uniform(0, np.pi)
        freq = rng.uniform(0.15, 0.35)   # cycles per pixel in the moire band
        amp = rng.uniform(15, 40)
        pattern = amp * np.sin(2 * np.pi * freq * (np.cos(theta) * x + np.sin(theta) * y))
        out[i] += pattern[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def _load_fixtures(fixtures_dir: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Load real/screen fixtures if the user has them.
    Expects: fixtures_dir/real/*.jpg and fixtures_dir/screen/*.jpg."""
    import cv2
    real_dir = fixtures_dir / "real"
    fake_dir = fixtures_dir / "screen"
    if not (real_dir.is_dir() and fake_dir.is_dir()):
        return None

    def _load(d: Path) -> list[np.ndarray]:
        xs = []
        for p in sorted(d.glob("*")):
            if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            img = cv2.imread(p.as_posix())
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            xs.append(cv2.resize(img, (128, 128)))
        return xs

    real = _load(real_dir)
    fake = _load(fake_dir)
    if not real or not fake:
        return None
    X = np.stack(real + fake).astype(np.uint8)
    y = np.array([0] * len(real) + [1] * len(fake), dtype=np.int32)
    return X, y


def build_dataset(fixtures_dir: Path, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    fixtures = _load_fixtures(fixtures_dir)
    if fixtures is not None:
        X, y = fixtures
        print(f"[moire] using {len(X)} fixture images ({int((y==0).sum())} real / "
              f"{int((y==1).sum())} screen)")
        return X, y

    print("[moire] no fixtures found; synthesizing a starter set (200+200)")
    clean = _synth_clean(200, rng)
    moire = _add_moire(_synth_clean(200, rng), rng)
    X = np.concatenate([clean, moire])
    y = np.array([0] * 200 + [1] * 200, dtype=np.int32)
    return X, y


def _featurize(X: np.ndarray, mode: str = "legacy") -> np.ndarray:
    from tqdm import tqdm
    dim = 375 if mode == "enhanced" else 20
    out = np.zeros((len(X), dim), dtype=np.float32)
    for i, img in enumerate(tqdm(X, desc=f"[moire] featurize ({mode})", unit="img")):
        out[i] = extract_features_for_mode(img, mode)
    return out


def _load_records_dataset(records_path: Path, max_per_class: int = 5000,
                          img_size: int = 128) -> tuple[np.ndarray, np.ndarray] | None:
    """Load (X, y) from a meta-records jsonl where each row has
    {"path": ..., "label": "live"|"spoof", ...}. Labels are mapped to
    {0: live, 1: spoof}. Balances per-class up to max_per_class.

    Returns None if the file is missing or no readable images load.
    """
    import json
    import cv2
    if not records_path.is_file():
        print(f"[moire] records file not found: {records_path}")
        return None
    live: list[np.ndarray] = []
    spoof: list[np.ndarray] = []
    with records_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            label = str(rec.get("label", "")).lower()
            path = rec.get("path") or rec.get("image_path") or rec.get("file")
            if not path:
                continue
            if label not in ("live", "spoof"):
                continue
            bucket = live if label == "live" else spoof
            if len(bucket) >= max_per_class:
                if len(live) >= max_per_class and len(spoof) >= max_per_class:
                    break
                continue
            img = cv2.imread(str(path))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (img_size, img_size))
            bucket.append(img)
    if not live or not spoof:
        print(f"[moire] records dataset empty: live={len(live)} spoof={len(spoof)}")
        return None
    X = np.stack(live + spoof).astype(np.uint8)
    y = np.array([0] * len(live) + [1] * len(spoof), dtype=np.int32)
    print(f"[moire] loaded records: {len(live)} live + {len(spoof)} spoof "
          f"from {records_path.name}")
    return X, y


def train(fixtures_dir: Path, seed: int = 0, mode: str = "legacy",
          records_path: Path | None = None, out_path: Path = OUT,
          max_per_class: int = 5000) -> None:
    import joblib
    import lightgbm as lgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, accuracy_score

    rng = np.random.default_rng(seed)
    if records_path is not None:
        rec = _load_records_dataset(records_path, max_per_class=max_per_class)
        if rec is None:
            print("[moire] records mode failed; falling back to fixtures/synthetic")
            X_img, y = build_dataset(fixtures_dir, rng)
        else:
            X_img, y = rec
    else:
        X_img, y = build_dataset(fixtures_dir, rng)
    X = _featurize(X_img, mode=mode)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=seed)

    # Regularization scales with feature_dim: 374-D needs tighter leaves +
    # higher min_child_samples vs 20-D legacy to avoid overfitting.
    feature_dim = X.shape[1]
    if feature_dim >= 200:
        num_leaves = 63
        min_child = 20
        n_est = 500
    else:
        num_leaves = 31
        min_child = 5
        n_est = 300

    clf = lgb.LGBMClassifier(
        n_estimators=n_est,
        learning_rate=0.05,
        num_leaves=num_leaves,
        min_child_samples=min_child,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=seed,
        verbose=-1,
    )
    clf.fit(Xtr, ytr)
    probs = clf.predict_proba(Xte)[:, 1]
    preds = (probs > 0.5).astype(int)
    print(f"[moire] feature_mode={mode} dim={feature_dim} "
          f"val_acc={accuracy_score(yte, preds):.3f} "
          f"AUC={roc_auc_score(yte, probs):.3f} "
          f"n_train={len(Xtr)} n_val={len(Xte)}")

    bundle = {
        "model": clf,
        "feature_dim": int(feature_dim),
        "feature_mode": mode,
        "version": 2 if mode == "enhanced" else 1,
        "n_train": int(len(Xtr)),
        "val_auc": float(roc_auc_score(yte, probs)),
        "val_acc": float(accuracy_score(yte, preds)),
        "channels": list(_LBP_CHANNEL_ORDER) if mode == "enhanced" else None,
    }
    joblib.dump(bundle, out_path)
    print(f"[moire] saved -> {out_path} ({out_path.stat().st_size/1e6:.2f} MB)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES,
                        help="dir with real/ and screen/ subfolders (default: tests/fixtures)")
    parser.add_argument("--records", type=Path, default=None,
                        help="meta-records jsonl with path+label rows (overrides fixtures)")
    parser.add_argument("--feature-mode", choices=["legacy", "enhanced"], default="enhanced",
                        help="20-D legacy or 374-D HSV/YCbCr LBP-augmented (default: enhanced)")
    parser.add_argument("--max-per-class", type=int, default=5000,
                        help="cap samples per class when reading --records")
    parser.add_argument("--out", type=Path, default=OUT,
                        help="output pkl path (default: pad/models/moire_gbt.pkl)")
    parser.add_argument("--force", action="store_true", help="retrain even if cached")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.out.exists() and not args.force:
        print(f"[moire] already trained: {args.out} ({args.out.stat().st_size/1e6:.2f} MB) "
              f"(pass --force to retrain)")
        return 0

    train(args.fixtures, seed=args.seed, mode=args.feature_mode,
          records_path=args.records, out_path=args.out,
          max_per_class=args.max_per_class)
    return 0


if __name__ == "__main__":
    sys.exit(main())
