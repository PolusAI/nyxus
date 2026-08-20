"""Mechanics guard for the GLDM background-pollution defect (bug #14b, fixed 2026-06).

Exercises the PRODUCTION featurize() path on a GLDM ROI *with background* at the DEFAULT
(MATLAB grey-binning) settings - the condition the C++ phantom oracle tests miss, because those
run on fully-masked ROIs where no off-ROI background sits inside the bounding box. The C++
counterpart is tests/test_2d_gldm_mechanics.h.

SPEC 2 mechanics tier: the pins below are Nyxus output at this config, not an oracle claim. A
PyRadiomics run on this fixture is NOT comparable - Nyxus re-bins the ROI with the MATLAB scheme
at coarse_gray_depth=64 while PyRadiomics bins at binWidth=1, so the two build their dependence
matrices over different level assignments and disagree by up to 108%. The measurement is in
tests/vetting/audit/gldm_2d_pyradiomics_vetting_report.md; the family's oracle assertions live on
the IBSI digital phantom in IBSI mode, where the two agree to 1e-15.
"""
import re
from pathlib import Path
import numpy as np
import pytest
import nyxus


# ----------------------------- helpers --------------------------------------
def _canonical_roi():
    """The pixelIntensityFeaturesTestData ROI from tests/test_data.h - the same irregular
    154-px region the C++ tests use, which reproduces the shape/moment defects. Returns None if
    the header can't be read."""
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
# Nyxus output on the quantized canonical ROI at coarse_gray_depth=64, ibsi=False, with the
# background correctly excluded. Pre-fix the background pixels in the bounding box were counted
# both as their own zones and as dependent neighbours, inflating Nz from 154 to 234 and giving
# GLDM_LDE ~16.6, GLDM_GLN ~30, GLDM_DN ~65 - an order of magnitude away from every pin here.
GLDM_BACKGROUND_EXCLUDED_REF_VALS = {
    "GLDM_SDE": 0.8025342712842713,
    "GLDM_LDE": 2.064935064935065,
    "GLDM_GLN": 3.948051948051948,
    "GLDM_DN": 92.44155844155844,
    "GLDM_DNN": 0.6002698600101197,
}


def test_2d_gldm_background_not_counted_mechanics():
    """Bug #14b (FIXED): the MATLAB binning path maps background (0) -> level 1, so the GLDM zone
    loop's `pi==0` guard tested the BINNED value and let background voxels inside the bounding box
    count as zones AND as dependent neighbours. Same root cause as GLCM #2."""
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

    for feature_name, ref in GLDM_BACKGROUND_EXCLUDED_REF_VALS.items():
        assert row[feature_name] == pytest.approx(ref, rel=1e-9), \
            f"{feature_name} moved off its pinned value - background exclusion may have regressed"

    # the bug's signature is an inflated dependence axis; this bound fails on the pre-fix ~16.6
    assert row["GLDM_LDE"] < 5.0, "GLDM_LDE inflated by background zones (bug regressed)"
