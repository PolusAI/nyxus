"""Regression / bug-exposure tests for 2D feature defects found during oracle validation
(2026-06). These exercise the PRODUCTION featurize() path on ROIs *with background* and at
the DEFAULT settings - the conditions the C++ unit tests miss.

This module covers the convex-hull / SOLIDITY defect (proposed fix #6): SOLIDITY must be <= 1.
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


def _blob():
    """Irregular ROI for shape/moment bug exposure. Prefers the canonical 154-px ROI (which
    reproduces solidity>1 and the ROI-radius anomaly); else an L-shape."""
    c = _canonical_roi()
    if c is not None:
        return c
    H, W = 16, 16
    inten = np.zeros((H, W), np.uint32)
    label = np.zeros((H, W), np.uint32)
    # L-shape: a vertical bar + a horizontal foot (clearly non-convex -> hull > area)
    for y in range(3, 13):
        for x in range(3, 6):
            label[y, x] = 1
    for y in range(10, 13):
        for x in range(6, 13):
            label[y, x] = 1
    # graded intensity (varies in x and y) over the ROI
    ys, xs = np.nonzero(label)
    inten[ys, xs] = (100 + 10 * xs + 7 * ys).astype(np.uint32)
    return inten, label


# ============================ CONVEX HULL / SOLIDITY ========================
def test_solidity_le_one():
    """Proposed fix #6 (Pick's-theorem pixel-count hull area): SOLIDITY = ROI area / hull area
    must be <= 1. The bare shoelace hull area runs through pixel CENTRES and under-counts
    coverage, so for small/elongated ROIs it fell below the ROI pixel count -> SOLIDITY > 1
    (e.g. 1.3 on the canonical ROI, which is impossible)."""
    inten, label = _blob()
    row = _one(["*ALL_MORPHOLOGY*"], inten, label)
    assert row["SOLIDITY"] <= 1.0 + 1e-6
