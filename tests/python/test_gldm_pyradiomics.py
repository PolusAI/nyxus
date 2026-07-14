"""Regression tests for the GLDM background-pollution defect (bug #14b, fixed 2026-06).

These exercise the PRODUCTION featurize() path on a GLDM ROI *with background* at the
DEFAULT (MATLAB grey-binning) settings - the conditions the C++ phantom unit tests miss,
because those run on fully-masked ROIs where no off-ROI background sits inside the
bounding box. The C++ counterpart of this file is tests/test_gldm_oracle.h.
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


# ============================ GLDM ==========================================
def test_gldm_background_not_counted():
    """Bug #14b (FIXED): the MATLAB binning path maps background (0) -> level 1; the GLDM zone
    loop's `pi==0` guard tests the BINNED value, so background voxels in the bounding box were
    counted as zones AND as dependent neighbours - inflating Nz (154 -> 234 on this ROI) and every
    count/dependence feature. Same root cause as GLCM #2. On the canonical concave ROI quantized to
    1..64, the count features must match the PyRadiomics reference (LDE 1.88, GLN 3.90, DN 95.7);
    pre-fix this ROI gave LDE ~16.6, GLN ~30, DN ~65 (background-polluted)."""
    c = _canonical_roi()
    if c is None:
        pytest.skip("canonical ROI (tests/test_data.h) not available")
    inten, label = c
    roi = label > 0
    v = inten[roi].astype(float)
    lvl = np.clip(np.floor(64 * (v - v.min()) / (v.max() - v.min())).astype(int) + 1, 1, 64)
    q = np.zeros_like(inten, np.uint32)
    q[roi] = lvl                                  # quantized to integer levels 1..64
    row = _one(["*ALL_GLDM*"], q, label.astype(np.uint32), coarse_gray_depth=64, ibsi=False)
    assert row["GLDM_LDE"] == pytest.approx(1.883, rel=0.20), \
        "GLDM Large-Dependence-Emphasis must match the reference (background must be excluded)"
    assert row["GLDM_LDE"] < 5.0,  "GLDM_LDE still inflated by background zones (bug regressed)"
    assert row["GLDM_GLN"] == pytest.approx(3.896, rel=0.15)
    assert row["GLDM_DN"]  == pytest.approx(95.70, rel=0.15)
