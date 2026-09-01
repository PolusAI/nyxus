"""Invariant tests for the out-of-core (oversized-ROI) path: trivial == non-trivial.

An ROI whose memory footprint reaches ram_limit is streamed through the disk-backed
OutOfRamPixelCloud instead of an in-memory pixel vector, and its features are computed by the
osized_calculate path rather than calculate(). The two paths must return identical values, so each
test here featurizes the same input twice -- once in-RAM, once out-of-core -- and requires every
feature column to agree.

Kind: *invariant* per tests/vetting/SPEC.md 2 -- these assert a required relation between two Nyxus
code paths, not agreement with an external tool, so they establish no vetting. The correct reading is
"out-of-core equals in-RAM, which is separately vetted", never "out-of-core is correct". The plumbing
/ gating checks for this path live in test_2d_ooc_mechanics.py.

ram_limit is a process-global in Nyxus, so each test sets it explicitly on both sides to stay
order-independent.
"""
import os
import numpy as np
import pytest

import nyxus

import test_data

tifffile = pytest.importorskip("tifffile")

# The "large" ram_limit for the in-RAM side of these comparisons. The biggest fixture here has a
# footprint of a few MB, so this only has to clear that -- it must NOT be sized generously, because
# Nyxus rejects a limit above currently available RAM (see _set_ram_limit_mb).
RAM_LIMIT_LARGE_MB = 64


def _set_ram_limit_mb(nyx, mb):
    """Set the process-global ram_limit and verify it was accepted.

    Nyxus rejects a limit above the RAM currently available and keeps the previous value, reporting
    the refusal without raising. ram_limit being a process-global, a silently rejected setting leaves
    an earlier test's 0 or 1 MB limit in place, the "in-RAM" side of a comparison then runs
    out-of-core, and the test fails on an unrelated assertion. Read the value back so a refusal fails
    here, naming the cause."""
    nyx.set_params(ram_limit=mb)
    got = nyx.get_params("ram_limit")["ram_limit"]
    assert got == mb, (
        "ram_limit=%d MB was not accepted (still %d MB) -- Nyxus refuses a limit above available "
        "RAM. Free memory or lower RAM_LIMIT_LARGE_MB." % (mb, got)
    )


def _make_pair(tmp_path):
    # Deterministic, non-degenerate: per-row offset + per-column gradient. 500x500 gives an
    # in-memory footprint well above 1 MB, so ram_limit=1 forces the out-of-core path.
    Y, X = 500, 500
    xg = (np.arange(X) % 256).astype(np.uint32)
    yg = ((np.arange(Y) % 200) * 256).astype(np.uint32)
    inten = (1 + xg[None, :] + yg[:, None]).astype(np.uint32)
    mask = np.ones((Y, X), np.uint32)  # a single ROI covering the whole image
    intdir = tmp_path / "int"
    segdir = tmp_path / "seg"
    intdir.mkdir()
    segdir.mkdir()
    tifffile.imwrite(str(intdir / "img.tif"), inten)
    tifffile.imwrite(str(segdir / "img.tif"), mask)
    return str(intdir) + os.sep, str(segdir) + os.sep


def _feature_cols(df):
    cols = [
        c
        for c in df.select_dtypes(include=[np.number]).columns
        if c not in ("ROI_label", "t_index", "c_index")
    ]
    return cols, df[cols].to_numpy(dtype=float).ravel()


def test_2d_ooc_2d_matches_in_ram_invariant(tmp_path):
    intdir, segdir = _make_pair(tmp_path)
    feats = ["*ALL_INTENSITY*"]

    n_ram = nyxus.Nyxus(feats)
    _set_ram_limit_mb(n_ram, RAM_LIMIT_LARGE_MB)  # large -> in-RAM (trivial); explicit so test is order-independent
    df_ram = n_ram.featurize_directory(intdir, segdir)

    n_ooc = nyxus.Nyxus(feats)
    _set_ram_limit_mb(n_ooc, 1)  # 1 MB -> forces the oversized / out-of-core path
    df_ooc = n_ooc.featurize_directory(intdir, segdir)

    cols, a = _feature_cols(df_ram)
    _, b = _feature_cols(df_ooc)
    assert a.size > 0 and a.shape == b.shape

    # Tolerance: both sides are the same build and the same formula, so the only admissible
    # difference is float summation order (the two paths visit pixels in a different sequence).
    # rel 1e-6 is far tighter than SPEC 7's rel=1e-3 floor for a same-definition comparison; it is
    # not an oracle tolerance, and nothing here claims agreement with an external tool.
    bad = [
        (c, p, q)
        for c, p, q in zip(cols, a, b)
        if abs(p - q) > 1e-6 * max(abs(p), abs(q), 1.0) + 1e-9
    ]
    assert not bad, "out-of-core intensity features diverge from in-RAM: %r" % (bad[:8],)


def _make_pair_2d(tmp_path, size=128, const=None):
    """A 2D intensity+mask pair. `const` gives every pixel the same intensity (a degenerate ROI,
    aux_min == aux_max); otherwise intensity is a deterministic non-degenerate gradient. Small on
    purpose: the out-of-core side below forces oversizing with ram_limit=0 rather than by exceeding
    a byte threshold, so the fixture does not have to be megabytes to reach the streaming path."""
    Y = X = size
    if const is None:
        xg = (np.arange(X) % 256).astype(np.uint32)
        yg = ((np.arange(Y) % 200) * 256).astype(np.uint32)
        inten = (1 + xg[None, :] + yg[:, None]).astype(np.uint32)
    else:
        inten = np.full((Y, X), const, dtype=np.uint32)
    mask = np.ones((Y, X), np.uint32)  # a single ROI covering the whole image
    intdir = tmp_path / "int"
    segdir = tmp_path / "seg"
    intdir.mkdir()
    segdir.mkdir()
    tifffile.imwrite(str(intdir / "img.tif"), inten)
    tifffile.imwrite(str(segdir / "img.tif"), mask)
    return str(intdir) + os.sep, str(segdir) + os.sep


def _ooc_vs_ram_2d(tmp_path, feats, const=None, size=128):
    """Featurize the same file pair twice, in-RAM and out-of-core, and require every feature column
    to agree. ram_limit=0 forces the oversized branch for any ROI (roiFootprint >= 0 is always
    true), which is what routes the ROI through the disk-backed OutOfRamPixelCloud and each
    feature's osized_calculate."""
    intdir, segdir = _make_pair_2d(tmp_path, size=size, const=const)

    n_ram = nyxus.Nyxus(feats)
    _set_ram_limit_mb(n_ram, RAM_LIMIT_LARGE_MB)  # large -> in-RAM (trivial); explicit so test is order-independent
    df_ram = n_ram.featurize_directory(intdir, segdir)

    n_ooc = nyxus.Nyxus(feats)
    _set_ram_limit_mb(n_ooc, 0)  # 0 -> every ROI oversized -> out-of-core
    df_ooc = n_ooc.featurize_directory(intdir, segdir)

    cols, a = _feature_cols(df_ram)
    _, b = _feature_cols(df_ooc)
    assert a.size > 0 and a.shape == b.shape

    # Tolerance: both sides are the same build and the same formula, so the only admissible
    # difference is float summation order (the two paths visit pixels in a different sequence).
    # rel 1e-6 is far tighter than SPEC 7's rel=1e-3 floor for a same-definition comparison; it is
    # not an oracle tolerance, and nothing here claims agreement with an external tool.
    bad = [
        (c, p, q)
        for c, p, q in zip(cols, a, b)
        if not (np.isnan(p) and np.isnan(q))
        and (
            np.isnan(p) != np.isnan(q)
            or abs(p - q) > 1e-6 * max(abs(p), abs(q), 1.0) + 1e-9
        )
    ]
    assert not bad, "2D out-of-core features diverge from in-RAM: %r" % (bad[:8],)


def test_2d_ooc_2d_texture_matches_in_ram_invariant(tmp_path):
    """All seven 2D texture families out-of-core must match the in-RAM path. Each of these used to
    bin intensities with to_grayscale() instead of TextureFeature::bin_intensities(), so GLCM threw,
    GLRLM/NGTDM indexed out of bounds and crashed, and GLSZM/GLDM/GLDZM returned wrong values."""
    _ooc_vs_ram_2d(
        tmp_path,
        ["*ALL_GLCM*", "*ALL_GLRLM*", "*ALL_GLSZM*", "*ALL_GLDZM*", "*ALL_GLDM*", "*ALL_NGLDM*", "*ALL_NGTDM*"],
    )


def test_2d_ooc_2d_moments_matches_in_ram_invariant(tmp_path):
    """2D geometric moments out-of-core must match in-RAM. Intensity- and shape-moments used to
    share one osized_calculate that dropped the intenfunction, so shape moments summed raw
    intensities (SPAT_MOMENT_00 returned the intensity sum rather than the ROI area)."""
    _ooc_vs_ram_2d(tmp_path, ["*GEOMOMS*"])


def test_2d_ooc_2d_gabor_matches_in_ram_invariant(tmp_path):
    """Gabor out-of-core must match in-RAM. Its streaming variant never assigned the
    'originalScore' baseline, so each frequency was divided by the tiny-number floor and returned an
    astronomically large value instead of a ratio in [0,1]."""
    _ooc_vs_ram_2d(tmp_path, ["GABOR"])


def test_2d_ooc_2d_morphology_matches_in_ram_invariant(tmp_path):
    """2D morphology out-of-core must match in-RAM. Chords used to step over columns instead of
    scanning every one, reporting shorter max/min/median chord lengths than the in-RAM path."""
    _ooc_vs_ram_2d(tmp_path, ["*ALL_MORPHOLOGY*"])


# The fixture every _ooc_vs_ram_2d test above uses is a full-image RECTANGLE, and that shape cannot
# discriminate the contour features: every step around a rectangle's boundary is an axis-aligned unit
# step, so the contour pixel COUNT and the sum of Euclidean step lengths are the same number. The
# out-of-core path computes PERIMETER the first way and the in-RAM path the second
# (contour.cpp: osized_calculate sets fval_PERIMETER = K.size(), calculate sums sqrt(sqdist)), and on
# a rectangle that difference is invisible. It stayed invisible here for as long as this file has
# existed. A disk has a genuinely diagonal boundary, so the two definitions separate.
def _disk_pair(tmp_path):
    """bench_disk64_diagonal_boundary, built by test_data.disk64_arrays() so the three modules that
    read this fixture cannot drift apart. ram_limit=0 forces the oversized branch for any ROI, so it
    does not have to be large to reach the out-of-core path."""
    return test_data.write_disk64_pair(tmp_path, tifffile, as_dirs=True)


def assert_ooc_agrees(df_ram, df_ooc, feature):
    """One feature, both paths. Same build and same helper, different pixel visit order, so the only
    admissible difference is float summation order."""
    assert feature in df_ram.columns, "%s missing from the frame" % feature
    a = float(df_ram[feature].iloc[0])
    b = float(df_ooc[feature].iloc[0])
    assert abs(a - b) <= 1e-6 * max(abs(a), abs(b), 1.0) + 1e-9, (
        "out-of-core %s diverges from in-RAM: %r vs %r" % (feature, a, b))


def test_2d_ooc_2d_contour_intensity_matches_in_ram_on_diagonal_boundary_invariant(tmp_path):
    """The five EDGE_* statistics and MASS_DISPLACEMENT must agree out-of-core on a shape whose
    boundary is diagonal, not only on the rectangle the other tests use.

    This is the cell tests/vetting/matrix/morphology.md records for the out-of-core contour path.
    PERIMETER is deliberately NOT among them: it does NOT agree, and that divergence is pinned as a
    known defect in test_2d_ooc_regression.py rather than hidden by leaving this fixture rectangular.
    """
    intdir, segdir = _disk_pair(tmp_path)
    feats = ["*ALL_MORPHOLOGY*", "*BASIC_MORPHOLOGY*"]

    n_ram = nyxus.Nyxus(feats)
    _set_ram_limit_mb(n_ram, RAM_LIMIT_LARGE_MB)
    df_ram = n_ram.featurize_directory(intdir, segdir)

    n_ooc = nyxus.Nyxus(feats)
    _set_ram_limit_mb(n_ooc, 0)          # 0 forces the oversized branch for any ROI
    df_ooc = n_ooc.featurize_directory(intdir, segdir)

    assert_ooc_agrees(df_ram, df_ooc, "MASS_DISPLACEMENT")
    assert_ooc_agrees(df_ram, df_ooc, "EDGE_MEAN_INTENSITY")
    assert_ooc_agrees(df_ram, df_ooc, "EDGE_STDDEV_INTENSITY")
    assert_ooc_agrees(df_ram, df_ooc, "EDGE_MAX_INTENSITY")
    assert_ooc_agrees(df_ram, df_ooc, "EDGE_MIN_INTENSITY")
    assert_ooc_agrees(df_ram, df_ooc, "EDGE_INTEGRATED_INTENSITY")


def test_2d_ooc_2d_zernike_matches_in_ram_invariant(tmp_path):
    """Zernike out-of-core must match in-RAM on an ordinary ROI as well as on the degenerate one
    covered below (its streaming variant lacked calculate()'s constant-ROI guard)."""
    _ooc_vs_ram_2d(tmp_path, ["ZERNIKE2D"])


ALL_2D_FEATURE_GROUPS = [
    "*ALL_INTENSITY*", "*ALL_IH*", "*BASIC_MORPHOLOGY*", "*ALL_MORPHOLOGY*", "*ALL_GLCM*",
    "*ALL_GLRLM*", "*ALL_GLSZM*", "*ALL_GLDZM*", "*ALL_GLDM*", "*ALL_NGLDM*", "*ALL_NGTDM*",
    "*GEOMOMS*", "GABOR", "ZERNIKE2D",
]


def test_2d_ooc_2d_blank_matches_in_ram_invariant(tmp_path):
    """A degenerate (constant-intensity) 2D ROI, across every 2D feature group at once. This is the
    case each feature's blank-ROI guard covers, and the one the bespoke out-of-core bodies used to
    get wrong in their own ways:

    - intensity intercepted aux_min == aux_max and replaced INTEGRATED_INTENSITY, ENERGY, MODE,
      ROOT_MEAN_SQUARED and the percentiles with the soft-NAN sentinel even though all of them are
      well defined on a constant ROI, and its excess kurtosis came from Moments4 in-core but from
      KURTOSIS-3 out-of-core (equal on ordinary data, not on a constant ROI);
    - erosion has no in-RAM value for a constant ROI (its driver skips one) but out-of-core ran the
      chain to the sanity cap and reported that instead;
    - zernike's streaming variant lacked calculate()'s constant-ROI guard and produced moments where
      the in-RAM path reports the soft-NAN sentinel."""
    _ooc_vs_ram_2d(tmp_path, ALL_2D_FEATURE_GROUPS, const=42)


def test_2d_ooc_2d_all_groups_match_in_ram_invariant(tmp_path):
    """The whole 2D feature surface at once on an ordinary ROI -- the umbrella guard for the
    trivial == out-of-core invariant."""
    _ooc_vs_ram_2d(tmp_path, ALL_2D_FEATURE_GROUPS)


def _make_pair_2d_two_rois(tmp_path, size=128):
    """One image with two side-by-side ROIs: label 1 is a non-degenerate gradient, label 2 is
    constant-intensity (a degenerate ROI). Both are oversized under ram_limit=0."""
    Y = X = size
    half = X // 2
    inten = np.empty((Y, X), np.uint32)
    xg = (np.arange(half) % 256).astype(np.uint32)
    yg = ((np.arange(Y) % 200) * 256).astype(np.uint32)
    inten[:, :half] = (1 + xg[None, :] + yg[:, None]).astype(np.uint32)  # gradient ROI
    inten[:, half:] = 42                                                 # constant ROI
    mask = np.empty((Y, X), np.uint32)
    mask[:, :half] = 1
    mask[:, half:] = 2
    intdir = tmp_path / "int"
    segdir = tmp_path / "seg"
    intdir.mkdir()
    segdir.mkdir()
    tifffile.imwrite(str(intdir / "img.tif"), inten)
    tifffile.imwrite(str(segdir / "img.tif"), mask)
    return str(intdir) + os.sep, str(segdir) + os.sep


def test_2d_ooc_2d_two_rois_no_state_leak_invariant(tmp_path):
    """A non-degenerate ROI and a degenerate (constant) ROI in the same image, all feature groups.
    The out-of-core feature loop reuses one persistent feature-method instance across ROIs (unlike
    the in-RAM path, which uses a fresh instance per ROI), so a feature that only pads-rather-than-
    resets its output buffer on the degenerate branch would report the previous ROI's values for the
    constant ROI. This caught ZERNIKE2D emitting the gradient ROI's moments for the constant ROI
    because its blank-ROI guard used vector::resize() (a no-op at unchanged length) instead of
    assign(). Also guards the ellipse-fitting family, whose out-of-core body had drifted from the
    in-RAM formulas for eccentricity/elongation and disagreed on any non-circular ROI.

    Scoped to families whose out-of-core body this change makes match the in-RAM path. Neighbor,
    enclosing/inscribing-circle, ROI-radius and distance-weighted-moment families have separate,
    pre-existing out-of-core gaps (the streaming path does not run the cross-ROI neighbor reduce and
    discards the contour data those need); they are intentionally not asserted here."""
    intdir, segdir = _make_pair_2d_two_rois(tmp_path)

    n_ram = nyxus.Nyxus(ALL_2D_FEATURE_GROUPS)
    _set_ram_limit_mb(n_ram, RAM_LIMIT_LARGE_MB)
    df_ram = n_ram.featurize_directory(intdir, segdir)

    n_ooc = nyxus.Nyxus(ALL_2D_FEATURE_GROUPS)
    _set_ram_limit_mb(n_ooc, 0)
    df_ooc = n_ooc.featurize_directory(intdir, segdir)

    # Align rows by ROI_label so a per-ROI comparison is order-independent
    df_ram = df_ram.sort_values("ROI_label").reset_index(drop=True)
    df_ooc = df_ooc.sort_values("ROI_label").reset_index(drop=True)

    # Scope to the families this change makes agree out-of-core: Zernike's moment buffer and
    # erosion's count (per-instance state a degenerate ROI must reset), and the ellipse-fitting
    # family (its out-of-core eccentricity/elongation formulas had drifted from the in-RAM path).
    ellipse = ("MAJOR_AXIS_LENGTH", "MINOR_AXIS_LENGTH", "ECCENTRICITY", "ELONGATION",
               "ORIENTATION", "ROUNDNESS")
    cols = [c for c in df_ram.columns
            if c.startswith("ZERNIKE2D") or c.startswith("EROSIONS_2_VANISH") or c in ellipse]
    assert cols, "expected ZERNIKE2D/EROSIONS/ellipse columns in the frame"

    # Tolerance: both sides are the same build and the same formula, so the only admissible
    # difference is float summation order (the two paths visit pixels in a different sequence).
    # rel 1e-6 is far tighter than SPEC 7's rel=1e-3 floor for a same-definition comparison; it is
    # not an oracle tolerance, and nothing here claims agreement with an external tool.
    bad = []
    for row in range(df_ram.shape[0]):
        lab = int(df_ram["ROI_label"].iloc[row])
        for c in cols:
            p = float(df_ram[c].iloc[row])
            q = float(df_ooc[c].iloc[row])
            if np.isnan(p) and np.isnan(q):
                continue
            if np.isnan(p) != np.isnan(q) or abs(p - q) > 1e-6 * max(abs(p), abs(q), 1.0) + 1e-9:
                bad.append((lab, c, p, q))
    assert not bad, "Zernike/erosion/ellipse out-of-core values diverge from in-RAM across ROIs: %r" % (bad[:8],)
