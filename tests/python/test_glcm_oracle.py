"""Regression / bug-exposure tests for 2D feature defects found during oracle validation
(2026-06). These exercise the PRODUCTION featurize() path on ROIs *with background* and at
the DEFAULT settings - the conditions the C++ unit tests miss (test_glcm.h hard-codes
offset=1 on a fully-masked phantom, which hid the GLCM defect).

This module covers the GLCM co-occurrence-offset default (bug #1) and GLCM background
pollution (bug #2).
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


# ============================ GLCM ==========================================
def test_glcm_contrast_nonzero_by_default():
    """Bug #1 (FIXED): glcm/offset defaulted to 0 -> dx=dy=0 -> CONTRAST=0 for any image.
    A horizontal intensity ramp must have nonzero 0deg contrast and zero 90deg contrast."""
    H, W = 9, 9
    inten = np.zeros((H, W), np.uint32)
    label = np.zeros((H, W), np.uint32)
    inten[2:7, 2:7] = np.tile(np.arange(1, 6) * 10, (5, 1))   # columns 10,20,30,40,50
    label[2:7, 2:7] = 1
    row = _one(["*ALL_GLCM*"], inten, label, coarse_gray_depth=8, ibsi=False)
    assert row["GLCM_CONTRAST_0"] > 0.0,  "0deg contrast must be > 0 (offset=0 regression)"
    assert row["GLCM_CONTRAST_90"] == pytest.approx(0.0, abs=1e-9), "constant columns -> 90deg contrast 0"


def test_glcm_background_not_counted():
    """Bug #2 (FIXED): the MATLAB binning path mapped background (0) -> level 1 and counted it,
    polluting the matrix with spurious diagonal mass. This only manifests for a CONCAVE ROI
    (background inside the bounding box). On the canonical irregular ROI, quantized to 1..64 so
    binning is identity, GLCM_CONTRAST_AVE must match the transparent numpy reference / MIRP /
    PyRadiomics (~133.9). Pre-fix this ROI gave ~99 (background-diluted); post-fix ~133."""
    c = _canonical_roi()
    if c is None:
        pytest.skip("canonical ROI (tests/test_data.h) not available")
    inten, label = c
    roi = label > 0
    v = inten[roi].astype(float)
    lvl = np.clip(np.floor(64 * (v - v.min()) / (v.max() - v.min())).astype(int) + 1, 1, 64)
    q = np.zeros_like(inten, np.uint32)
    q[roi] = lvl                                  # quantized to integer levels 1..64
    row = _one(["*ALL_GLCM*"], q, label.astype(np.uint32), coarse_gray_depth=64, ibsi=False)
    assert row["GLCM_CONTRAST_AVE"] == pytest.approx(133.9, rel=0.12), \
        "concave-ROI GLCM contrast must match the reference (background must be excluded)"
    assert row["GLCM_CONTRAST_AVE"] > 115.0, "contrast still diluted by background (bug #2 regressed)"
