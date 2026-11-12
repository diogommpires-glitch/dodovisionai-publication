"""
DODO VISION AI — Medical Reference Ranges
Normal/expected values for vitals by age group and ethnicity.
Displayed as "real_value\expected_avg" in the UI.
"""

# ── Heart Rate (BPM) by age group ──
# Source: AHA guidelines, resting HR
HR_BY_AGE = {
    (0, 1): (100, 160),    # newborn
    (1, 5): (80, 130),
    (6, 12): (70, 110),
    (13, 17): (60, 100),
    (18, 25): (60, 95),
    (26, 35): (58, 92),
    (36, 45): (58, 94),
    (46, 55): (60, 96),
    (56, 65): (60, 98),
    (66, 75): (62, 100),
    (76, 100): (64, 104),
}

# ── SpO2 by age ──
SPO2_BY_AGE = {
    (0, 17): (95, 100),
    (18, 45): (95, 100),
    (46, 65): (94, 99),
    (66, 80): (93, 98),
    (81, 100): (92, 97),
}

# ── Respiratory Rate (breaths/min) by age ──
RESP_BY_AGE = {
    (0, 1): (30, 60),
    (1, 5): (20, 30),
    (6, 12): (16, 24),
    (13, 17): (12, 20),
    (18, 65): (12, 20),
    (66, 100): (12, 22),
}

# ── HRV RMSSD (ms) by age — healthy resting ──
HRV_BY_AGE = {
    (18, 25): (30, 80),
    (26, 35): (25, 70),
    (36, 45): (20, 60),
    (46, 55): (15, 50),
    (56, 65): (12, 40),
    (66, 100): (10, 35),
    (0, 17): (30, 90),
}

# ── Blink Rate (blinks/min) — mostly age-independent ──
BLINK_NORMAL = (15, 20)  # 15-20 blinks/min is normal for adults
BLINK_BY_AGE = {
    (0, 5): (3, 15),
    (6, 17): (10, 18),
    (18, 100): (15, 20),
}

# ── PERCLOS thresholds (same for all) ──
PERCLOS_NORMAL = (0, 0.15)  # <15% is normal/alert

# ── Stress Index (composite) ──
STRESS_NORMAL = (0.1, 0.35)  # 0.1-0.35 is relaxed-normal range

# ── Ethnicity adjustments ──
# SpO2: studies show slightly lower baseline in some populations
# Source: Sjoding et al. 2020, NEJM
SPO2_ETHNICITY_OFFSET = {
    "white": 0,
    "black": -1.5,        # pulse ox tends to overread by ~2% in darker skin
    "asian": -0.5,
    "indian": -0.5,
    "middle eastern": 0,
    "latino hispanic": -0.5,
}

# HR ethnicity adjustments (minor, from population studies)
HR_ETHNICITY_OFFSET = {
    "white": 0,
    "black": 2,
    "asian": -1,
    "indian": 1,
    "middle eastern": 0,
    "latino hispanic": 0,
}


def _find_range(table, age):
    """Find the matching range for an age in a lookup table."""
    if age is None:
        age = 30  # default assumption
    for (lo, hi), vals in table.items():
        if lo <= age <= hi:
            return vals
    return list(table.values())[-1]  # fallback to last entry


def get_reference(age=None, ethnicity=None):
    """
    Get expected normal ranges for all health metrics given age and ethnicity.
    Returns dict: { metric: { "lo": x, "hi": y, "avg": z, "unit": str } }
    """
    if age is None:
        age = 30
    eth = (ethnicity or "").lower().strip()

    hr_lo, hr_hi = _find_range(HR_BY_AGE, age)
    hr_off = HR_ETHNICITY_OFFSET.get(eth, 0)
    hr_lo += hr_off
    hr_hi += hr_off
    hr_avg = round((hr_lo + hr_hi) / 2)

    spo2_lo, spo2_hi = _find_range(SPO2_BY_AGE, age)
    spo2_off = SPO2_ETHNICITY_OFFSET.get(eth, 0)
    spo2_avg = round((spo2_lo + spo2_hi) / 2 + spo2_off, 1)

    resp_lo, resp_hi = _find_range(RESP_BY_AGE, age)
    resp_avg = round((resp_lo + resp_hi) / 2)

    hrv_lo, hrv_hi = _find_range(HRV_BY_AGE, age)
    hrv_avg = round((hrv_lo + hrv_hi) / 2)

    blink_lo, blink_hi = _find_range(BLINK_BY_AGE, age)
    blink_avg = round((blink_lo + blink_hi) / 2)

    return {
        "hr": {"lo": round(hr_lo), "hi": round(hr_hi), "avg": hr_avg, "unit": "bpm"},
        "spo2": {"lo": round(spo2_lo + spo2_off, 1), "hi": spo2_hi, "avg": spo2_avg, "unit": "%"},
        "resp": {"lo": resp_lo, "hi": resp_hi, "avg": resp_avg, "unit": "rpm"},
        "hrv": {"lo": hrv_lo, "hi": hrv_hi, "avg": hrv_avg, "unit": "ms"},
        "blink": {"lo": blink_lo, "hi": blink_hi, "avg": blink_avg, "unit": "/min"},
        "stress": {"lo": STRESS_NORMAL[0], "hi": STRESS_NORMAL[1], "avg": round((STRESS_NORMAL[0] + STRESS_NORMAL[1]) / 2, 2), "unit": ""},
        "perclos": {"lo": PERCLOS_NORMAL[0], "hi": PERCLOS_NORMAL[1], "avg": round(PERCLOS_NORMAL[1] * 100), "unit": "%"},
    }
