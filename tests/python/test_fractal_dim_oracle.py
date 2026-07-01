"""Regression / bug-exposure tests for 2D feature defects found during oracle validation
(2026-06). These exercise the PRODUCTION featurize() path on ROIs *with background* and at
the DEFAULT settings - the conditions the C++ unit tests miss.

This module covers the fractal-dimension defect (proposed fix #8): FRACT_DIM_BOXCOUNT in [1, 2].
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
    """Irregular ROI for shape/moment bug exposure. Prefers the canonical 154-px ROI; else an L-shape."""
    c = _canonical_roi()
    if c is not None:
        return c
    H, W = 16, 16
    inten = np.zeros((H, W), np.uint32)
    label = np.zeros((H, W), np.uint32)
    # L-shape: a vertical bar + a horizontal foot (clearly non-convex)
    for y in range(3, 13):
        for x in range(3, 6):
            label[y, x] = 1
    for y in range(10, 13):
        for x in range(6, 13):
            label[y, x] = 1
    ys, xs = np.nonzero(label)
    inten[ys, xs] = (100 + 10 * xs + 7 * ys).astype(np.uint32)
    return inten, label


# ============================ FRACTAL DIMENSION =============================
def test_fractal_dimension_in_range():
    """Proposed fix #8 (fractal dim: mean log-log slope, not slope-of-slopes): the box-counting
    fractal dimension is the MEAN of the local log(count)-vs-log(scale) slopes. The old code
    returned the least-squares slope of those slopes *against their index* (~0 for a clean power
    law), so FRACT_DIM_BOXCOUNT came out ~-0.07..-0.83 - outside the valid [1, 2] range for a 2D
    shape. (FRACT_DIM_PERIMETER similarly needed the Richardson D = 1 - slope convention.)"""
    inten, label = _blob()
    row = _one(["*ALL_MORPHOLOGY*"], inten, label)
    assert 1.0 <= float(row["FRACT_DIM_BOXCOUNT"]) <= 2.0
