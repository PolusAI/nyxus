"""Regression / bug-exposure tests for 2D feature defects found during oracle validation
(2026-06). These exercise the PRODUCTION featurize() path on ROIs *with background* and at
the DEFAULT settings - the conditions the C++ unit tests miss (test_glcm.h hard-codes
offset=1 on a fully-masked phantom, which hid the GLCM defect).

GLCM tests are live regression guards (the offset=0 + background-pollution defect is fixed).
The remaining tests assert invariants that must hold for ANY ROI; the ones marked xfail
correspond to defects that are not yet fixed - when a fix lands they XPASS and the xfail can
be removed.
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


def _solid_square(side=10, border=4, val=500):
    H = W = side + 2 * border
    inten = np.zeros((H, W), np.uint32)
    label = np.zeros((H, W), np.uint32)
    inten[border:border + side, border:border + side] = val   # UNIFORM intensity
    label[border:border + side, border:border + side] = 1
    return inten, label


# ============================ GLCM (FIXED) ==================================
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


def test_compactness_uses_mean_centroid():
    """Bug #18 (FIXED): COMPACTNESS (std of centroid-to-pixel distances / area) measured distances
    from the coordinate SUM (cen_x/cen_y were never divided by n) instead of the mean centroid, and
    used Pixel2::sqdist(int,int) which truncates the fractional centroid. Same fix as PR #350.
    On a 10x10 square it must equal the independent numpy value (std(dist, ddof=1)/n ~ 0.0141);
    pre-fix it was ~0.029 (sum-centroid)."""
    inten, label = _solid_square(side=10, border=4)
    row = _one(["*ALL_MORPHOLOGY*"], inten, label)
    ys, xs = np.nonzero(label > 0)
    n = len(xs)
    d = np.hypot(xs - xs.mean(), ys - ys.mean())
    expected = d.std(ddof=1) / n                  # Nyxus Moments2.std() uses Bessel's correction
    assert float(row["COMPACTNESS"]) == pytest.approx(expected, rel=1e-3), \
        "COMPACTNESS must use the MEAN centroid with double-precision distances"


# ===================== OPEN bugs (xfail until fixed) ========================
# regression guard for PROPOSED fix #6 (Pick's-theorem pixel-count hull area); validated locally
def test_solidity_le_one():
    inten, label = _blob()
    row = _one(["*ALL_MORPHOLOGY*"], inten, label)
    assert row["SOLIDITY"] <= 1.0 + 1e-6


# regression guard for PROPOSED fix #7 (euler: pad mask + Cd '||' + mode formula); validated locally
def test_euler_number_simply_connected():
    inten, label = _solid_square()
    row = _one(["*ALL_MORPHOLOGY*"], inten, label)
    assert row["EULER_NUMBER"] == 1


# regression guard for PROPOSED fix #5 (DIAMETER_EQUAL_AREA: add sqrt); validated locally
def test_diameter_equal_area_is_sqrt():
    inten, label = _solid_square(side=10, border=4)
    row = _one(["*ALL_MORPHOLOGY*"], inten, label)
    area = float(row["AREA_PIXELS_COUNT"])
    assert row["DIAMETER_EQUAL_AREA"] == pytest.approx(np.sqrt(4 * area / np.pi), rel=0.05)


# regression guard for PROPOSED fix #10 (weighted centroid 0-based, drop +1); validated locally
def test_weighted_centroid_matches_plain_for_uniform_intensity():
    inten, label = _solid_square()       # uniform intensity -> weighted == geometric centroid
    row = _one(["*ALL_INTENSITY*", "*ALL_MORPHOLOGY*"], inten, label)
    assert row["WEIGHTED_CENTROID_X"] == pytest.approx(row["CENTROID_X"], abs=0.25)
    assert row["WEIGHTED_CENTROID_Y"] == pytest.approx(row["CENTROID_Y"], abs=0.25)


# regression guard for PROPOSED fix #4 (central moment: drop (int) centroid truncation); validated locally
def test_first_central_moment_zero():
    inten, label = _blob()
    row = _one(["*SGEOMOMS*"], inten, label)
    m00 = abs(float(row["CENTRAL_MOMENT_00"])) or 1.0
    assert abs(float(row["CENTRAL_MOMENT_01"])) / m00 < 1e-3
    assert abs(float(row["CENTRAL_MOMENT_10"])) / m00 < 1e-3


def test_plain_intensity_zeroth_moment_positive():
    # NOTE: WEIGHTED_*/IMOM_W* are distance-to-contour LOG-weighted moments (weight =
    # intensity*log(dist+eps)); their zeroth moment can legitimately be negative -> NOT a bug.
    # The PLAIN intensity raw zeroth moment IMOM_RM_00 must be > 0 (= sum of intensities).
    inten, label = _blob()
    row = _one(["*IGEOMOMS*"], inten, label)
    assert float(row["IMOM_RM_00"]) > 0.0


# regression guard for PROPOSED fix #8 (fractal dim: mean log-log slope, not slope-of-slopes); validated locally
def test_fractal_dimension_in_range():
    inten, label = _blob()
    row = _one(["*ALL_MORPHOLOGY*"], inten, label)
    assert 1.0 <= float(row["FRACT_DIM_BOXCOUNT"]) <= 2.0


# regression guard for PROPOSED fix #11 (ROI_RADIUS = sqrt(min_sqdist), was squared); validated locally
def test_roi_radius_within_image():
    inten, label = _blob()
    H, W = label.shape
    row = _one(["*ALL_MORPHOLOGY*"], inten, label)
    assert float(row["ROI_RADIUS_MAX"]) <= np.hypot(H, W)


# #9 RECLASSIFIED (not a bug): DIAMETER_INSCRIBING_CIRCLE is Nyxus's CENTROID-based inscribing circle
# (2*min centroid-to-contour distance), the companion of the centroid-based CIRCUMSCRIBING circle - NOT
# the largest inscribed circle (that is imea's max_inclosing, the oracle the tracker mistakenly compared
# against). The PROPOSED fix here is only the dropped -1 centroid offset; assert the centroid-based
# sanity invariants rather than the largest-inscribed value.
def test_inscribing_circle_centroid_based_sane():
    inten, label = _solid_square(side=10, border=4)
    row = _one(["*ALL_MORPHOLOGY*"], inten, label)
    insc = float(row["DIAMETER_INSCRIBING_CIRCLE"])
    circ = float(row["DIAMETER_CIRCUMSCRIBING_CIRCLE"])
    assert 0.0 < insc <= circ + 1e-6              # inscribing <= circumscribing (both centroid-centered)


# regression guard for PROPOSED fix #17 (circle.cpp undoes the contour (+1,+1) frame offset from
# contour.cpp's missing -1 pad correction); validated locally
def test_inscribing_circumscribing_circle_correct_for_square():
    """A 10x10 square (centroid 8.5,8.5): the inscribing diameter must be ~2*min centroid-to-edge
    distance (~9.06) and the circumscribing ~2*max centroid-to-corner (~12.73) == the min-enclosing.
    Pre-fix the +1 contour offset gave INSCRIBING 7.07 (=5*sqrt2, too small) and CIRCUMSCRIBING 15.56."""
    inten, label = _solid_square(side=10, border=4)
    row = _one(["*ALL_MORPHOLOGY*"], inten, label)
    insc = float(row["DIAMETER_INSCRIBING_CIRCLE"])
    circ = float(row["DIAMETER_CIRCUMSCRIBING_CIRCLE"])
    menc = float(row["DIAMETER_MIN_ENCLOSING_CIRCLE"])
    assert insc == pytest.approx(9.06, abs=0.3),  f"inscribing {insc} (expect ~9.06; 7.07 == +1 contour-offset bug)"
    assert circ == pytest.approx(12.73, abs=0.3), f"circumscribing {circ} (expect ~12.73; 15.56 == +1 contour-offset bug)"
    assert circ == pytest.approx(menc, abs=0.3),  "circumscribing ~ min-enclosing for a square (both reach the corners)"


# regression guard for PROPOSED fixes #12 (closest-neighbor dist/ang need CENTROID, now forced when
# neighbors requested) and #13 (PERCENT_TOUCHING deduped + adjacency, was >100%); validated locally
def test_neighbor_distance_and_percent_touching():
    # two 4x4 squares separated by a 2px gap (neighbors within neighbor_distance=5, but NOT touching)
    H, W = 12, 20
    inten = np.zeros((H, W), np.uint32)
    label = np.zeros((H, W), np.uint32)
    inten[4:8, 3:7] = 100;  label[4:8, 3:7] = 1
    inten[4:8, 9:13] = 100; label[4:8, 9:13] = 2
    df = _run(["*ALL_NEIGHBOR*"], inten, label, neighbor_distance=5)
    assert (df["NUM_NEIGHBORS"] >= 1).all()
    assert (df["CLOSEST_NEIGHBOR1_DIST"] > 0).all()        # was 0 (CENTROID not computed) == bug #12
    assert (df["PERCENT_TOUCHING"] <= 100.0).all()         # was >100% == bug #13
    assert (df["PERCENT_TOUCHING"] == 0.0).all()           # 2px gap -> not touching (adjacency-based)
