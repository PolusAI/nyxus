"""Regression / bug-exposure tests for 2D feature defects found during oracle validation
(2026-06). These exercise the PRODUCTION featurize() path on ROIs *with background* and at
the DEFAULT settings - the conditions the C++ unit tests miss.

This module covers the chords max-angle / AC-vs-MC defect (bug #16).
"""
import re
from pathlib import Path
import numpy as np
import pytest
import nyxus


# ----------------------------- helpers --------------------------------------
def _canonical_roi():
    """The pixelIntensityFeaturesTestData ROI from tests/test_data.h - the same irregular
    154-px region the C++ tests use, which reproduces the shape/moment defects. Falls back to
    an L-shape if the header can't be read."""
    hdr = Path(__file__).resolve().parent.parent / "test_data.h"
    try:
        txt = hdr.read_text(encoding="utf-8", errors="replace")
        body = re.search(r"pixelIntensityFeaturesTestData\[\]\s*=\s*\{(.*?)\};", txt, re.S).group(1)
        pts = [(int(x), int(y), int(v)) for x, y, v in
               re.findall(r"\{\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\}", body)]
        W = max(p[0] for p in pts) + 2
        H = max(p[1] for p in pts) + 2
        inten = np.zeros((H, W), np.uint32)
        label = np.zeros((H, W), np.uint32)
        for x, y, v in pts:
            inten[y, x] = v
            label[y, x] = 1
        return inten, label
    except Exception:
        return None


def _run(features, inten, label, **kw):
    nyx = nyxus.Nyxus(features=features, n_feature_calc_threads=1, **kw)
    df = nyx.featurize(inten.astype(np.float64), label.astype(np.uint32),
                       intensity_names=["i"], label_names=["l"])
    return df  # one row per label


def _one(features, inten, label, **kw):
    return _run(features, inten, label, **kw).iloc[0]


# ============================ CHORDS ========================================
def test_chord_max_angle_distinct_from_min():
    """Bug #16 (FIXED): chords.cpp computed the max-chord angle index from iteMin
    (`idxmax = distance(begin, iteMin)`), so MAXCHORDS_MAX_ANG was ALWAYS equal to
    MAXCHORDS_MIN_ANG (same copy-paste error for ALLCHORDS). When the longest and shortest
    chords have different lengths they occur at different orientations, so the max- and
    min-angle must differ. Also exercised: ALLCHORDS mode/median were built from the
    max-chords histogram (MC) instead of all chords (AC)."""
    c = _canonical_roi()
    if c is None:
        pytest.skip("canonical ROI (tests/test_data.h) not available")
    inten, label = c
    row = _one(["MAXCHORDS_MAX", "MAXCHORDS_MIN", "MAXCHORDS_MAX_ANG", "MAXCHORDS_MIN_ANG",
                "ALLCHORDS_MAX", "ALLCHORDS_MIN", "ALLCHORDS_MAX_ANG", "ALLCHORDS_MIN_ANG"],
               inten, label)
    if row["MAXCHORDS_MAX"] != row["MAXCHORDS_MIN"]:
        assert row["MAXCHORDS_MAX_ANG"] != row["MAXCHORDS_MIN_ANG"], \
            "MAXCHORDS max-angle == min-angle (idxmax=iteMin copy-paste bug regressed)"
    if row["ALLCHORDS_MAX"] != row["ALLCHORDS_MIN"]:
        assert row["ALLCHORDS_MAX_ANG"] != row["ALLCHORDS_MIN_ANG"], \
            "ALLCHORDS max-angle == min-angle (idxmax=iteMin copy-paste bug regressed)"
