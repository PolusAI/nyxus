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
    binning is identity, GLCM_CONTRAST_AVE is an ORACLE-grade match to PyRadiomics: CONTRAST depends
    only on the grey-level difference |i-j|, so it is invariant to matrix symmetrization AND to the
    absolute level values -> nyxus and PyRadiomics genuinely agree.

    Measured provenance (PR #356 review, rebuilt backend 2026-07-09, this exact quantized ROID):
        Nyxus  GLCM_CONTRAST_AVE = 133.4338   (ibsi=False, coarse_gray_depth=64)
        PyRadiomics Contrast     = 133.8569   (radiomics v3.0.1, symmetricalGLCM=True, binCount=64,
                                               distances=[1], force2D=True, weightingNorm=None)
        => agreement 0.32% (was asserted with a ~12% band before; tightened to 1%).
    Pre-fix this ROI gave ~99 (background-diluted, ~26% low), which rel=1e-2 decisively excludes."""
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
    # Tight oracle check against the measured PyRadiomics value. The residual 0.32% (nyxus 133.43 vs
    # PyRadiomics 133.86) is definitional (grey-binning edges + directional-angle aggregation between
    # estimators), not floating-point; rel=1e-2 leaves ~3x margin over that while still excluding the
    # background-polluted ~99 that bug #2 produced.
    assert row["GLCM_CONTRAST_AVE"] == pytest.approx(133.8569, rel=1e-2), \
        "concave-ROI GLCM contrast must match the PyRadiomics oracle (133.86) to ~1%, not the " \
        "~99 background-polluted value that bug #2 produced"


def test_glcm_acor_family_ibsi_oracle():
    """PR #356 review (Comment 2): ACOR, SUMAVERAGE, IDN, IDMN depend on the *absolute* grey-level
    values / Ng (unlike CONTRAST, which depends only on |i-j|). Under the MATLAB-binning path
    (ibsi=False, the config tests/test_glcm.h uses) those absolute levels are re-mapped, so these
    four DIVERGE from PyRadiomics by up to ~43% (ACOR) and are NOT oracle-vetted there. They ARE
    genuinely third-party-vetted on the IBSI path (symmetric matrix + identity binning), which this
    test pins tightly.

    Dense fixture -- every grey level 1..8 occurs, so PyRadiomics does not drop/re-index levels
    (which would shift ACOR/SUMAVERAGE and Ng for IDN/IDMN). Frozen phantom img[i,j]=((i+2j)%8)+1,
    8x8, one-pixel background border; the goldens are specific to it.

    PyRadiomics reference (generated offline, radiomics v3.0.1, symmetricalGLCM=True, binWidth=1
    [identity on the integer image], distances=[1], force2D=True, force2Ddimension=0,
    weightingNorm=None, label=1; SumAverage taken as 2 x JointAverage) on the identical array+mask:
        ACOR=20.512755, SUMAVERAGE=9.020408, IDN=0.779479, IDMN=0.887342
    (also reproducible to 6 dp by a transparent numpy standard-Haralick reference).
    Nyxus (ibsi=True) reproduces all four exactly; assert at rel=1e-3."""
    ii, jj = np.meshgrid(np.arange(8), np.arange(8), indexing="ij")
    arr = ((ii + 2 * jj) % 8) + 1
    assert set(np.unique(arr).tolist()) == set(range(1, 9)), \
        "dense phantom must contain every grey level 1..8 (else PyRadiomics re-indexes)"
    inten = np.zeros((10, 10), np.uint32)     # 1-px background border around the 8x8 fixture
    label = np.zeros((10, 10), np.uint32)
    inten[1:9, 1:9] = arr
    label[1:9, 1:9] = 1
    # ibsi=True -> symmetric matrix + identity binning; this is the load-bearing setting (do NOT use
    # the MATLAB-binning config here, under which these four are convention-bound, not oracle values).
    row = _one(["*ALL_GLCM*"], inten, label, coarse_gray_depth=8, ibsi=True)
    goldens = {"GLCM_ACOR_AVE": 20.512755, "GLCM_SUMAVERAGE_AVE": 9.020408,
               "GLCM_IDN_AVE": 0.779479, "GLCM_IDMN_AVE": 0.887342}
    for key, gold in goldens.items():
        assert row[key] == pytest.approx(gold, rel=1e-3), \
            f"{key}: nyxus(ibsi=True) {row[key]} must match the PyRadiomics/IBSI oracle {gold} to 1e-3"
