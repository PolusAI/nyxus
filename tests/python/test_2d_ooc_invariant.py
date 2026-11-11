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


def _set_metaparams(nyx, metaparams):
    """Apply each "<family>/<name>=<value>" setting and read it back, so a setting the backend did
    not take fails here rather than surfacing as a value mismatch further on."""
    for mp in metaparams:
        name, val = mp.split("=")
        nyx.set_metaparam(mp)
        assert nyx.get_metaparam(name) == float(val), "%s was not applied" % mp


def _ooc_vs_ram_2d(tmp_path, feats, const=None, size=128, metaparams=()):
    """Featurize the same file pair twice, in-RAM and out-of-core, and require every feature column
    to agree. ram_limit=0 forces the oversized branch for any ROI (roiFootprint >= 0 is always
    true), which is what routes the ROI through the disk-backed OutOfRamPixelCloud and each
    feature's osized_calculate. `metaparams` are applied to both instances alike. Returns the
    in-RAM frame."""
    intdir, segdir = _make_pair_2d(tmp_path, size=size, const=const)

    n_ram = nyxus.Nyxus(feats)
    _set_ram_limit_mb(n_ram, RAM_LIMIT_LARGE_MB)  # large -> in-RAM (trivial); explicit so test is order-independent
    _set_metaparams(n_ram, metaparams)
    df_ram = n_ram.featurize_directory(intdir, segdir)

    n_ooc = nyxus.Nyxus(feats)
    _set_ram_limit_mb(n_ooc, 0)  # 0 -> every ROI oversized -> out-of-core
    _set_metaparams(n_ooc, metaparams)
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
    return df_ram


def test_2d_ooc_2d_texture_matches_in_ram_invariant(tmp_path):
    """All seven 2D texture families out-of-core must match the in-RAM path. Each of these used to
    bin intensities with to_grayscale() instead of TextureFeature::bin_intensities(), so GLCM threw,
    GLRLM/NGTDM indexed out of bounds and crashed, and GLSZM/GLDM/GLDZM returned wrong values."""
    _ooc_vs_ram_2d(
        tmp_path,
        ["*ALL_GLCM*", "*ALL_GLRLM*", "*ALL_GLSZM*", "*ALL_GLDZM*", "*ALL_GLDM*", "*ALL_NGLDM*", "*ALL_NGTDM*"],
    )


# Non-default GLCM settings: co-occurrence offset 2 against the default 1, and 16 grey levels against
# the default 64. Both are written only to the GLCM family's own settings vector, so each shows which
# vector a path was given. Metaparameters live on the instance's own Environment, so nothing set here
# reaches another instance or a later test.
GLCM_METAPARAMS = ("glcm/offset=2", "glcm/greydepth=16")


def _glcm_in_ram(tmp_path, metaparams):
    intdir, segdir = _make_pair_2d(tmp_path)
    nyx = nyxus.Nyxus(["*ALL_GLCM*"])
    _set_ram_limit_mb(nyx, RAM_LIMIT_LARGE_MB)
    _set_metaparams(nyx, metaparams)
    return _feature_cols(nyx.featurize_directory(intdir, segdir))[1]


def test_2d_ooc_2d_glcm_metaparam_matches_in_ram_invariant(tmp_path):
    """GLCM out-of-core must match in-RAM under a non-default glcm/offset and glcm/greydepth, i.e.
    the oversized-ROI path must be handed the GLCM family's own settings vector, which is where
    set_metaparam writes them. At the defaults every family's vector holds the same values, so the
    texture test above cannot tell which vector the out-of-core path was given.

    The fixture is a 128 px gradient spanning every one of the 64 default grey levels, so doubling
    the offset or binning into 16 levels moves the GLCM values. The control half asserts exactly
    that, per setting, so that an out-of-core path running at the defaults cannot agree with the
    in-RAM one by the fixture being insensitive to the setting."""
    base = tmp_path / "default"
    base.mkdir()
    default_vals = _glcm_in_ram(base, ())
    for mp in GLCM_METAPARAMS:
        d = tmp_path / mp.split("/")[1].split("=")[0]
        d.mkdir()
        vals = _glcm_in_ram(d, (mp,))
        assert vals.shape == default_vals.shape
        assert not np.allclose(vals, default_vals, rtol=1e-6, atol=1e-9, equal_nan=True), (
            "%s leaves the in-RAM GLCM values unchanged on this fixture" % mp)

    _ooc_vs_ram_2d(tmp_path, ["*ALL_GLCM*"], metaparams=GLCM_METAPARAMS)


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


# The fixture every _ooc_vs_ram_2d test above uses is a full-image RECTANGLE at the image origin, and
# that shape cannot discriminate the contour features: every step around a rectangle's boundary is an
# axis-aligned unit step, so the contour's pixel count and its length are the same number, and its
# bounding box starts where the image does. A disk away from the origin has a diagonal boundary and a
# bounding box that does not.
def _disk_pair(tmp_path):
    """bench_disk64_diagonal_boundary, built by test_data.disk64_arrays() so the modules that read
    this fixture cannot drift apart. ram_limit=0 forces the oversized branch for any ROI, so it does
    not have to be large to reach the out-of-core path."""
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
    """The contour's own features, MASS_DISPLACEMENT and ROI_RADIUS_* must agree out-of-core on a
    shape whose boundary is diagonal, not only on the rectangle the other tests use.

    This is the cell tests/vetting/matrix/morphology.md records for the out-of-core contour path. On
    a diagonal boundary the contour's pixel count and its length are different numbers, so PERIMETER
    tells a path that counts pixels from one that sums step lengths. The disk lies away from the
    image origin, so ROI_RADIUS_MEAN and ROI_RADIUS_MEDIAN tell a contour placed in absolute
    coordinates from one left relative to the bounding box.
    """
    intdir, segdir = _disk_pair(tmp_path)
    feats = ["*ALL_MORPHOLOGY*", "*BASIC_MORPHOLOGY*"]

    n_ram = nyxus.Nyxus(feats)
    _set_ram_limit_mb(n_ram, RAM_LIMIT_LARGE_MB)
    df_ram = n_ram.featurize_directory(intdir, segdir)

    n_ooc = nyxus.Nyxus(feats)
    _set_ram_limit_mb(n_ooc, 0)          # 0 forces the oversized branch for any ROI
    df_ooc = n_ooc.featurize_directory(intdir, segdir)

    assert_ooc_agrees(df_ram, df_ooc, "PERIMETER")
    assert_ooc_agrees(df_ram, df_ooc, "DIAMETER_EQUAL_PERIMETER")
    assert_ooc_agrees(df_ram, df_ooc, "ROI_RADIUS_MEAN")
    assert_ooc_agrees(df_ram, df_ooc, "ROI_RADIUS_MAX")
    assert_ooc_agrees(df_ram, df_ooc, "ROI_RADIUS_MEDIAN")
    assert_ooc_agrees(df_ram, df_ooc, "MASS_DISPLACEMENT")
    assert_ooc_agrees(df_ram, df_ooc, "EDGE_MEAN_INTENSITY")
    assert_ooc_agrees(df_ram, df_ooc, "EDGE_STDDEV_INTENSITY")
    assert_ooc_agrees(df_ram, df_ooc, "EDGE_MAX_INTENSITY")
    assert_ooc_agrees(df_ram, df_ooc, "EDGE_MIN_INTENSITY")
    assert_ooc_agrees(df_ram, df_ooc, "EDGE_INTEGRATED_INTENSITY")


# Every 2D feature whose value is computed from the ROI's contour: the contour's own features, those
# built on PERIMETER, the circles, the ROI radii, the fractal dimensions, the moments weighted by the
# distance to the contour and the radial distribution. Matched by exact code or by prefix.
CONTOUR_FEATURE_CODES = (
    "PERIMETER", "DIAMETER_EQUAL_PERIMETER", "EDGE_MEAN_INTENSITY", "EDGE_STDDEV_INTENSITY",
    "EDGE_MAX_INTENSITY", "EDGE_MIN_INTENSITY", "EDGE_INTEGRATED_INTENSITY",
    "CIRCULARITY", "GEODETIC_LENGTH", "THICKNESS",
    "DIAMETER_MIN_ENCLOSING_CIRCLE", "DIAMETER_INSCRIBING_CIRCLE", "DIAMETER_CIRCUMSCRIBING_CIRCLE",
    "ROI_RADIUS_MEAN", "ROI_RADIUS_MAX", "ROI_RADIUS_MEDIAN",
    "FRACT_DIM_BOXCOUNT", "FRACT_DIM_PERIMETER",
)
CONTOUR_FEATURE_PREFIXES = (
    "WEIGHTED_SPAT_MOMENT_", "WEIGHTED_CENTRAL_MOMENT_", "WT_NORM_CTR_MOM_", "WEIGHTED_HU_M",
    "IMOM_WRM_", "IMOM_WCM_", "IMOM_WNCM_", "IMOM_WHU",
    "FRAC_AT_D_", "MEAN_FRAC_", "RADIAL_CV_",
)

# The neighbour features are measured between ROIs, by a pass over every ROI's contour at once that
# the out-of-core path does not run, so out of core they keep their initial values. POLYGONALITY_AVE
# and HEXAGONALITY_* are computed only for a ROI with neighbours. All of them are left out here.
NEIGHBOR_FEATURE_PREFIXES = (
    "NUM_NEIGHBORS", "PERCENT_TOUCHING", "CLOSEST_NEIGHBOR", "ANG_BW_NEIGHBORS", "POLYGONALITY_",
    "HEXAGONALITY_",
)


def _is_contour_feature(col):
    return col in CONTOUR_FEATURE_CODES or col.startswith(CONTOUR_FEATURE_PREFIXES)


def _write_pair(tmp_path, mask, inten=None, tile=None):
    """Write one intensity + mask pair as TIFFs, tiled `tile` x `tile` when `tile` is given. The
    default intensity is a gradient, so the edge and intensity-weighted features see distinct
    pixel values."""
    if inten is None:
        yy, xx = np.mgrid[0:mask.shape[0], 0:mask.shape[1]]
        inten = 1 + xx + 3 * yy
    intdir = tmp_path / "int"
    segdir = tmp_path / "seg"
    intdir.mkdir()
    segdir.mkdir()
    kw = {"tile": (tile, tile)} if tile else {}
    tifffile.imwrite(str(intdir / "img.tif"), inten.astype(np.uint32), **kw)
    tifffile.imwrite(str(segdir / "img.tif"), mask.astype(np.uint32), **kw)
    if tile:
        with tifffile.TiffFile(str(segdir / "img.tif")) as tf:
            assert tf.pages[0].is_tiled
    return str(intdir) + os.sep, str(segdir) + os.sep


def _contour_mask(name):
    """(mask, tile, anisotropy_x) of one contour fixture."""
    if name == "offset_rectangle":
        # a 3 x 5 rectangle away from the image origin: a contour left relative to the bounding box
        # lies (4, 7) pixels from the ROI it is measured against
        m = np.zeros((20, 20), np.uint32)
        m[7:10, 4:9] = 1
        return m, None, 1.0
    if name == "diagonal_disk":
        # a disk: its boundary steps diagonally, so a pixel count is not its length
        yy, xx = np.mgrid[0:64, 0:64]
        return (((xx - 31.5) ** 2 + (yy - 31.5) ** 2) <= 20 ** 2).astype(np.uint32), None, 1.0
    if name == "holed_square":
        # one outer border and one hole border
        m = np.zeros((40, 40), np.uint32)
        m[5:29, 5:29] = 1
        m[12:22, 12:22] = 0
        return m, None, 1.0
    if name == "two_fragments":
        # one label in two separate pieces, each with an outer border of its own
        m = np.zeros((80, 80), np.uint32)
        m[5:30, 5:30] = 1
        m[40:65, 45:70] = 1
        return m, None, 1.0
    if name == "small_irregular":
        # the worked example of the fractal dimension page: padded side 16, so box counting takes
        # the minimum count over shifted grids
        M = np.array([
            [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 0, 0, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 1, 1, 0, 0, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 1, 1, 0]], np.uint32)
        m = np.zeros((20, 20), np.uint32)
        m[5:14, 3:13] = M
        return m, None, 1.0
    if name == "two_rois":
        # two labels in one image: the out-of-core path measures them one after the other with
        # the same feature instances
        m = np.zeros((128, 128), np.uint32)
        m[0:21, 0:21] = 1
        yy, xx = np.mgrid[0:128, 0:128]
        m[((xx - 100) ** 2 + (yy - 90) ** 2) <= 15 ** 2] = 2
        return m, None, 1.0
    if name == "multi_tile":
        # a holed ROI away from the origin across 32 x 32 tiles of a 160 x 96 image, plus a second
        # piece of the same label in another tile
        m = np.zeros((96, 160), np.uint32)
        m[20:80, 40:130] = 1
        m[40:60, 70:100] = 0
        m[84:92, 140:156] = 1
        return m, 32, 1.0
    if name == "anisotropic_disk":
        # a disk of radius 8, resampled twice along x
        yy, xx = np.mgrid[0:40, 0:40]
        return (((xx - 20) ** 2 + (yy - 20) ** 2) <= 64).astype(np.uint32), None, 2.0
    raise ValueError(name)


CONTOUR_FIXTURES = ("offset_rectangle", "diagonal_disk", "holed_square", "two_fragments",
                    "small_irregular", "two_rois", "multi_tile", "anisotropic_disk")


def _featurize_both_paths(tmp_path, mask, tile=None, aniso=1.0, feats=("*ALL*",)):
    """(in-RAM frame, out-of-core frame) of one pair, rows sorted by ROI_label."""
    intdir, segdir = _write_pair(tmp_path, mask, tile=tile)
    frames = []
    for ram_mb in (RAM_LIMIT_LARGE_MB, 0):    # 0 forces the oversized branch for any ROI
        kw = {"anisotropy_x": aniso} if aniso != 1.0 else {}
        nyx = nyxus.Nyxus(list(feats), **kw)
        _set_ram_limit_mb(nyx, ram_mb)
        frames.append(nyx.featurize_directory(intdir, segdir).sort_values("ROI_label").reset_index(drop=True))
    return frames


@pytest.mark.parametrize("fixture", CONTOUR_FIXTURES)
def test_2d_ooc_2d_contour_features_match_in_ram_invariant(tmp_path, fixture):
    """Every contour-based feature of a ROI forced out of core equals its in-memory value.

    Each fixture is a shape on which a contour that differs from the in-memory one changes values:
    a ROI away from the image origin (a contour relative to the bounding box), a diagonal boundary
    (PERIMETER as a pixel count), a hole and a second fragment (a contour of one border), a small
    irregular ROI (box counting on a single grid), two ROIs in one image (state carried from one
    ROI to the next), a ROI across several tiles, and an anisotropic ROI (a contour of the resampled
    ROI). Every column but the neighbour features is compared, and the comparison is required to
    include every contour-based feature code."""
    mask, tile, aniso = _contour_mask(fixture)
    df_ram, df_ooc = _featurize_both_paths(tmp_path, mask, tile=tile, aniso=aniso)

    assert list(df_ram["ROI_label"]) == list(df_ooc["ROI_label"])
    cols = [c for c in _feature_cols(df_ram)[0] if not c.startswith(NEIGHBOR_FEATURE_PREFIXES)]
    contour_cols = [c for c in cols if _is_contour_feature(c)]
    missing = [c for c in CONTOUR_FEATURE_CODES if c not in contour_cols]
    assert not missing, "contour features missing from the frame: %r" % missing
    for p in CONTOUR_FEATURE_PREFIXES:
        assert any(c.startswith(p) for c in contour_cols), "no %s* column in the frame" % p

    # Tolerance: same build, same contour, same formulas; only the order the pixels are visited in
    # differs, and so the float summation order.
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
    assert not bad, "%s: out-of-core features diverge from in-RAM: %r" % (fixture, bad[:10])


def test_2d_ooc_2d_offset_rectangle_contour_features_are_analytic_invariant(tmp_path):
    """Both paths give the analytic values on the 3 x 5 rectangle away from the image origin.

    The centroid is (6, 8) and the farthest contour pixels are the corners, 1 down and 2 across, so
    the circumscribing circle's diameter is 2 sqrt(5). The three middle-row interior pixels lie 1
    from the contour and the other twelve on it, so ROI_RADIUS_MEAN is 3/15, ROI_RADIUS_MAX 1 and
    ROI_RADIUS_MEDIAN 0. The in-memory value is part of the check: agreeing on a value that moved
    would not satisfy it."""
    mask, _, _ = _contour_mask("offset_rectangle")
    for df in _featurize_both_paths(tmp_path, mask, feats=("*ALL_MORPHOLOGY*",)):
        assert float(df["DIAMETER_CIRCUMSCRIBING_CIRCLE"].iloc[0]) == pytest.approx(2 * np.sqrt(5), rel=1e-12)
        assert float(df["ROI_RADIUS_MEAN"].iloc[0]) == pytest.approx(0.2, rel=1e-12)
        assert float(df["ROI_RADIUS_MAX"].iloc[0]) == pytest.approx(1.0, rel=1e-12)
        assert float(df["ROI_RADIUS_MEDIAN"].iloc[0]) == pytest.approx(0.0, abs=1e-12)
        assert float(df["PERIMETER"].iloc[0]) == pytest.approx(12.0, rel=1e-12)


def test_2d_ooc_2d_boxcount_of_two_box_sizes_is_positive_zero_invariant(tmp_path):
    """A 2 x 2 square leaves too few box sizes to fit a slope, so FRACT_DIM_BOXCOUNT is 0 on both
    paths, with a positive sign."""
    mask = np.zeros((8, 8), np.uint32)
    mask[3:5, 2:4] = 1
    for df in _featurize_both_paths(tmp_path, mask, feats=("*ALL_MORPHOLOGY*",)):
        v = float(df["FRACT_DIM_BOXCOUNT"].iloc[0])
        assert v == 0.0 and not np.signbit(v), "FRACT_DIM_BOXCOUNT is %r" % v


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


def test_2d_ooc_2d_one_pixel_roi_after_ordinary_roi_matches_in_ram_invariant(tmp_path):
    """A one-pixel ROI computed after an ordinary ROI in the same out-of-core run. It has no chords
    and an empty convex hull, so the chord features must read 0 rather than the previous ROI's values
    (one feature instance serves every out-of-core ROI) and the Martin and Nassenstein diameters must
    take the guard for an empty hull rather than read past it. Those diameters are requested without
    any convex hull feature, so the ordinary ROI's nonzero values also require their dependency on
    the hull."""
    inten = np.arange(1, 64 * 64 + 1, dtype=np.uint32).reshape(64, 64)
    mask = np.zeros((64, 64), np.uint32)
    mask[5:35, 5:35] = 1
    mask[50, 50] = 2
    intdir = tmp_path / "int"
    segdir = tmp_path / "seg"
    intdir.mkdir()
    segdir.mkdir()
    tifffile.imwrite(str(intdir / "img.tif"), inten)
    tifffile.imwrite(str(segdir / "img.tif"), mask)

    feats = ["MAXCHORDS_MAX", "MAXCHORDS_MEAN", "ALLCHORDS_MAX", "ALLCHORDS_MEAN",
             "STAT_MARTIN_DIAM_MAX", "STAT_MARTIN_DIAM_MEAN",
             "STAT_NASSENSTEIN_DIAM_MAX", "STAT_NASSENSTEIN_DIAM_MEAN"]
    frames = []
    for limit in (RAM_LIMIT_LARGE_MB, 0):
        n = nyxus.Nyxus(feats)
        _set_ram_limit_mb(n, limit)
        df = n.featurize_directory(str(intdir) + os.sep, str(segdir) + os.sep)
        frames.append(df.sort_values("ROI_label").reset_index(drop=True))
    df_ram, df_ooc = frames

    assert df_ooc["MAXCHORDS_MAX"].iloc[0] > 0, "the ordinary ROI must have chords to leak"
    assert df_ooc["STAT_MARTIN_DIAM_MAX"].iloc[0] > 0
    assert df_ooc["STAT_NASSENSTEIN_DIAM_MAX"].iloc[0] > 0
    for c in feats:
        assert df_ooc[c].to_numpy() == pytest.approx(df_ram[c].to_numpy(), rel=1e-6, abs=1e-9), c
        assert df_ooc[c].iloc[1] == 0, c


def test_2d_ooc_2d_two_rois_no_state_leak_invariant(tmp_path):
    """A non-degenerate ROI and a degenerate (constant) ROI in the same image, all feature groups.
    The out-of-core feature loop reuses one persistent feature-method instance across ROIs (unlike
    the in-RAM path, which uses a fresh instance per ROI), so a feature that only pads-rather-than-
    resets its output buffer on the degenerate branch would report the previous ROI's values for the
    constant ROI. This caught ZERNIKE2D emitting the gradient ROI's moments for the constant ROI
    because its blank-ROI guard used vector::resize() (a no-op at unchanged length) instead of
    assign(). Also guards the ellipse-fitting family, whose out-of-core body had drifted from the
    in-RAM formulas for eccentricity/elongation and disagreed on any non-circular ROI.

    Scoped to the families whose state a degenerate ROI must reset. The contour-based features of two
    ROIs in one image are compared by test_2d_ooc_2d_contour_features_match_in_ram_invariant; the
    neighbour features are not compared out of core at all (see NEIGHBOR_FEATURE_PREFIXES)."""
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


# A 160 x 96 image at 32 x 32 tiles: 5 tiles across, 3 down. The grid is not square, so a scan that
# swaps the horizontal and vertical tile counts, or derives a tile's column from the wrong one, puts
# pixels at different coordinates than the untiled copy does.
TILED_W, TILED_H, TILE = 160, 96, 32


def _make_tiled_untiled_pairs(tmp_path):
    """The same intensity and mask arrays written twice: as tiled TIFFs and as strip TIFFs.

    Label 1 is an ellipse inside the single tile at grid row 0, column 3, off the grid's diagonal.
    Label 2 spans tiles: a column at x 36..50 reaching down to y 60, and an arm to its right at
    y 40..50 that runs to x 120. Tile by tile, the arm's last pixel is the last one streamed, and the
    column's rows below the arm come before it, which is the reverse of their raster order."""
    yy, xx = np.mgrid[0:TILED_H, 0:TILED_W]
    inten = (1 + xx + 7 * yy).astype(np.uint32)
    mask = np.zeros((TILED_H, TILED_W), np.uint32)
    mask[((xx - 112) / 12.0) ** 2 + ((yy - 15) / 9.0) ** 2 <= 1.0] = 1
    mask[4:61, 36:51] = 2
    mask[40:51, 51:121] = 2
    assert set(np.unique(mask[0:TILE, 3 * TILE:4 * TILE])) == {0, 1}
    assert not (mask == 1)[:, :3 * TILE].any() and not (mask == 1)[TILE:, :].any()

    dirs = {}
    for kind, kw in (("tiled", {"tile": (TILE, TILE)}), ("untiled", {})):
        intdir = tmp_path / kind / "int"
        segdir = tmp_path / kind / "seg"
        intdir.mkdir(parents=True)
        segdir.mkdir(parents=True)
        tifffile.imwrite(str(intdir / "img.tif"), inten, **kw)
        tifffile.imwrite(str(segdir / "img.tif"), mask, **kw)
        with tifffile.TiffFile(str(intdir / "img.tif")) as tf:
            assert tf.pages[0].is_tiled == (kind == "tiled")
        dirs[kind] = (str(intdir) + os.sep, str(segdir) + os.sep)
    return dirs


def test_2d_ooc_2d_tiled_matches_untiled_invariant(tmp_path):
    """Out-of-core featurization of a tiled TIFF must equal that of a strip TIFF holding the same
    arrays, every column of both ROIs.

    Both sides run out-of-core, so the pixels and the code path are the same and only the order in
    which the pixel cloud is filled differs: tile by tile against row by row. That isolates the
    out-of-core tile scan from any out-of-core vs in-RAM difference. What it discriminates: a scan
    that places a tile's pixels at the wrong column fails on label 1, and a contour that depends on
    the order the cloud was filled in fails on label 2."""
    dirs = _make_tiled_untiled_pairs(tmp_path)

    frames = {}
    for kind, (intdir, segdir) in dirs.items():
        nyx = nyxus.Nyxus(ALL_2D_FEATURE_GROUPS)
        _set_ram_limit_mb(nyx, 0)  # 0 -> every ROI oversized -> out-of-core
        frames[kind] = nyx.featurize_directory(intdir, segdir).sort_values("ROI_label").reset_index(drop=True)

    df_t, df_u = frames["tiled"], frames["untiled"]
    assert list(df_t["ROI_label"]) == [1, 2] and list(df_u["ROI_label"]) == [1, 2]
    cols, _ = _feature_cols(df_u)
    assert cols and _feature_cols(df_t)[0] == cols

    # Tolerance: same build, same pixels, same formulas; only the summation order differs. The
    # absolute floor covers moments that are zero analytically and come out as float noise.
    bad = []
    for row in range(df_u.shape[0]):
        lab = int(df_u["ROI_label"].iloc[row])
        for c in cols:
            p = float(df_t[c].iloc[row])
            q = float(df_u[c].iloc[row])
            if np.isnan(p) and np.isnan(q):
                continue
            if np.isnan(p) != np.isnan(q) or abs(p - q) > 1e-6 * max(abs(p), abs(q), 1.0) + 1e-9:
                bad.append((lab, c, p, q))
    assert not bad, "out-of-core features of a tiled TIFF diverge from its untiled copy: %r" % (bad[:8],)
