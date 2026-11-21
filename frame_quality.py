"""
DODO VISION AI v7 — Frame Quality Assessment (Step 9)
Pre-pipeline quality gate: brightness, contrast, blur.
Returns confidence multiplier for biometric readings.
"""

import cv2
import numpy as np


class FrameQuality:
    """Assess frame quality before biometric processing."""

    MIN_BRIGHTNESS = 30
    MAX_BRIGHTNESS = 230
    MIN_CONTRAST = 20
    BLUR_THRESHOLD = 50

    @staticmethod
    def assess(frame):
        """Returns dict with quality metrics and confidence multiplier."""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_ch = lab[:, :, 0].astype(float)
        brightness = float(np.mean(l_ch))
        contrast = float(np.std(l_ch))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        dark = brightness < FrameQuality.MIN_BRIGHTNESS
        bright = brightness > FrameQuality.MAX_BRIGHTNESS
        low_c = contrast < FrameQuality.MIN_CONTRAST
        blurry = blur < FrameQuality.BLUR_THRESHOLD

        warnings = []
        if dark: warnings.append(f"Too dark ({brightness:.0f})")
        if bright: warnings.append(f"Overexposed ({brightness:.0f})")
        if low_c: warnings.append(f"Low contrast ({contrast:.1f})")
        if blurry: warnings.append(f"Blurry ({blur:.1f})")

        ok = not (dark or bright or low_c or blurry)
        n_bad = sum([dark, bright, low_c, blurry])
        mul = max(0.1, 1.0 - n_bad * 0.25)

        return {
            "brightness": round(brightness, 1), "contrast": round(contrast, 1),
            "blur": round(blur, 1), "overall_ok": ok,
            "confidence_mul": round(mul, 2), "warnings": warnings,
        }

    @staticmethod
    def assess_crop(crop):
        """Quick quality check on a face crop. Returns (multiplier, warnings)."""
        if crop is None or crop.size == 0:
            return 0.0, ["Empty crop"]
        h, w = crop.shape[:2]
        warnings = []
        mul = 1.0
        if w < 48 or h < 48:
            warnings.append(f"Small face ({w}x{h})")
            mul *= 0.5
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
        m = float(np.mean(gray))
        if m < 40: warnings.append("Dark face"); mul *= 0.6
        elif m > 220: warnings.append("Bright face"); mul *= 0.7
        if cv2.Laplacian(gray, cv2.CV_64F).var() < 30:
            warnings.append("Blurry face"); mul *= 0.7
        return mul, warnings


frame_quality = FrameQuality()
