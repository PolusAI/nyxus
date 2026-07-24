"""Mechanics tests for the out-of-core (oversized-ROI) path.

An ROI whose memory footprint reaches ram_limit is streamed through the disk-backed
OutOfRamPixelCloud instead of an in-memory pixel vector, and its features are computed by
the osized_calculate path. These tests exercise that plumbing (not an external oracle):

- test_ooc_2d_matches_in_ram: the out-of-core path must produce the same intensity features
  as the in-RAM path on the same file pair.
- test_ooc_montage_oversized_fails_loudly: the in-memory montage path has no out-of-core
  support and must raise on an oversized ROI rather than emit a silent all-zero row.

ram_limit is a process-global in Nyxus, so each test sets it explicitly on both sides to stay
order-independent.
"""
import os
import pathlib
import numpy as np
import pytest

import nyxus

tifffile = pytest.importorskip("tifffile")

DATA_NIFTI = pathlib.Path(__file__).resolve().parent.parent / "data" / "nifti"

# The "large" ram_limit for the in-RAM side of these comparisons. It has to clear the biggest
# fixture footprint here (a few MB) while staying under the RAM the host actually has free:
# Nyxus rejects a limit above available RAM and keeps the previous value, and since ram_limit is
# a process-global, a rejected setting silently leaves an earlier test's 0/1 MB limit in place
# and the in-RAM side then comes back oversized. A limit near a machine's total RAM therefore
# passes on a workstation and fails on a CI runner.
RAM_LIMIT_LARGE_MB = 1000


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


def test_ooc_2d_matches_in_ram(tmp_path):
    intdir, segdir = _make_pair(tmp_path)
    feats = ["*ALL_INTENSITY*"]

    n_ram = nyxus.Nyxus(feats)
    n_ram.set_params(ram_limit=RAM_LIMIT_LARGE_MB)  # large -> in-RAM (trivial); explicit so test is order-independent
    df_ram = n_ram.featurize_directory(intdir, segdir)

    n_ooc = nyxus.Nyxus(feats)
    n_ooc.set_params(ram_limit=1)  # 1 MB -> forces the oversized / out-of-core path
    df_ooc = n_ooc.featurize_directory(intdir, segdir)

    cols, a = _feature_cols(df_ram)
    _, b = _feature_cols(df_ooc)
    assert a.size > 0 and a.shape == b.shape

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
    """The 2D counterpart of _ooc_vs_ram_3d: featurize the same file pair twice, in-RAM and
    out-of-core, and require every feature column to agree. ram_limit=0 forces the oversized branch
    for any ROI (roiFootprint >= 0 is always true), which is what routes the ROI through the
    disk-backed OutOfRamPixelCloud and each feature's osized_calculate."""
    intdir, segdir = _make_pair_2d(tmp_path, size=size, const=const)

    n_ram = nyxus.Nyxus(feats)
    n_ram.set_params(ram_limit=RAM_LIMIT_LARGE_MB)  # large -> in-RAM (trivial); explicit so test is order-independent
    df_ram = n_ram.featurize_directory(intdir, segdir)

    n_ooc = nyxus.Nyxus(feats)
    n_ooc.set_params(ram_limit=0)  # 0 -> every ROI oversized -> out-of-core
    df_ooc = n_ooc.featurize_directory(intdir, segdir)

    cols, a = _feature_cols(df_ram)
    _, b = _feature_cols(df_ooc)
    assert a.size > 0 and a.shape == b.shape

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


def test_ooc_2d_texture_matches_in_ram(tmp_path):
    """All seven 2D texture families out-of-core must match the in-RAM path. Each of these used to
    bin intensities with to_grayscale() instead of TextureFeature::bin_intensities(), so GLCM threw,
    GLRLM/NGTDM indexed out of bounds and crashed, and GLSZM/GLDM/GLDZM returned wrong values."""
    _ooc_vs_ram_2d(
        tmp_path,
        ["*ALL_GLCM*", "*ALL_GLRLM*", "*ALL_GLSZM*", "*ALL_GLDZM*", "*ALL_GLDM*", "*ALL_NGLDM*", "*ALL_NGTDM*"],
    )


def test_ooc_2d_moments_matches_in_ram(tmp_path):
    """2D geometric moments out-of-core must match in-RAM. Intensity- and shape-moments used to
    share one osized_calculate that dropped the intenfunction, so shape moments summed raw
    intensities (SPAT_MOMENT_00 returned the intensity sum rather than the ROI area)."""
    _ooc_vs_ram_2d(tmp_path, ["*GEOMOMS*"])


def test_ooc_2d_gabor_matches_in_ram(tmp_path):
    """Gabor out-of-core must match in-RAM. Its streaming variant never assigned the
    'originalScore' baseline, so each frequency was divided by the tiny-number floor and returned an
    astronomically large value instead of a ratio in [0,1]."""
    _ooc_vs_ram_2d(tmp_path, ["GABOR"])


def test_ooc_2d_morphology_matches_in_ram(tmp_path):
    """2D morphology out-of-core must match in-RAM. Chords used to step over columns instead of
    scanning every one, reporting shorter max/min/median chord lengths than the in-RAM path."""
    _ooc_vs_ram_2d(tmp_path, ["*ALL_MORPHOLOGY*"])


def test_ooc_2d_zernike_matches_in_ram(tmp_path):
    """Zernike out-of-core must match in-RAM on an ordinary ROI as well as on the degenerate one
    covered below (its streaming variant lacked calculate()'s constant-ROI guard)."""
    _ooc_vs_ram_2d(tmp_path, ["ZERNIKE2D"])


ALL_2D_FEATURE_GROUPS = [
    "*ALL_INTENSITY*", "*ALL_IH*", "*BASIC_MORPHOLOGY*", "*ALL_MORPHOLOGY*", "*ALL_GLCM*",
    "*ALL_GLRLM*", "*ALL_GLSZM*", "*ALL_GLDZM*", "*ALL_GLDM*", "*ALL_NGLDM*", "*ALL_NGTDM*",
    "*GEOMOMS*", "GABOR", "ZERNIKE2D",
]


def test_ooc_2d_blank_matches_in_ram(tmp_path):
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


def test_ooc_2d_all_groups_match_in_ram(tmp_path):
    """The whole 2D feature surface at once on an ordinary ROI -- the umbrella guard for the
    trivial == out-of-core invariant."""
    _ooc_vs_ram_2d(tmp_path, ALL_2D_FEATURE_GROUPS)


def _make_volume_pair(tmp_path):
    # A 3D OME-TIFF volume (one IFD per Z) + a whole-volume mask. Z*Y*X = 8*90*90 = 64800
    # voxels; the in-memory 3D footprint is well over 1 MB, so ram_limit=1 forces the oversized
    # (out-of-core) volumetric path. Intensity is a deterministic, non-degenerate function of
    # (x,y,z) so every 3D intensity feature is meaningful.
    Z, Y, X = 8, 90, 90
    z = np.arange(Z)[:, None, None]
    y = np.arange(Y)[None, :, None]
    x = np.arange(X)[None, None, :]
    inten = (1 + (x % 256) + (y % 200) * 256 + z * 10000).astype(np.uint32)
    inten = np.broadcast_to(inten, (Z, Y, X)).astype(np.uint32)
    mask = np.ones((Z, Y, X), np.uint32)  # a single ROI covering the whole volume
    intp = tmp_path / "vol_int.ome.tif"
    segp = tmp_path / "vol_seg.ome.tif"
    tifffile.imwrite(str(intp), inten, metadata={"axes": "ZYX"})
    tifffile.imwrite(str(segp), mask, metadata={"axes": "ZYX"})
    return str(intp), str(segp)


def test_ooc_3d_matches_in_ram(tmp_path):
    """The 3D out-of-core path (voxel cloud streamed to disk, keeping z) must produce the same
    3D intensity AND surface/morphology features as the in-RAM path on the same volume pair."""
    intp, segp = _make_volume_pair(tmp_path)
    feats = ["*3D_ALL_INTENSITY*", "*3D_ALL_MORPHOLOGY*"]

    # Nyxus3D takes ram_limit in the constructor (its set_params does not expose it)
    n_ram = nyxus.Nyxus3D(feats, ram_limit=RAM_LIMIT_LARGE_MB)  # large -> in-RAM (trivial)
    df_ram = n_ram.featurize_files([intp], [segp], False)

    n_ooc = nyxus.Nyxus3D(feats, ram_limit=1)  # 1 MB -> forces the oversized / out-of-core volumetric path
    df_ooc = n_ooc.featurize_files([intp], [segp], False)

    cols, a = _feature_cols(df_ram)
    _, b = _feature_cols(df_ooc)
    assert a.size > 0 and a.shape == b.shape

    bad = [
        (c, p, q)
        for c, p, q in zip(cols, a, b)
        if abs(p - q) > 1e-6 * max(abs(p), abs(q), 1.0) + 1e-9
    ]
    assert not bad, "3D out-of-core intensity features diverge from in-RAM: %r" % (bad[:8],)


def _make_volume_pair_partial(tmp_path):
    # Same intensity field as _make_volume_pair, but a non-cuboid (ellipsoid) mask that leaves
    # background voxels INSIDE the ROI's bounding box. Out-of-core paths that build a grey-level
    # LUT from the whole binned cube (mask + background) -- e.g. matlab binning maps the raw-0
    # background to a nonzero bin that must still appear in the LUT -- are only exercised when the
    # bbox actually contains background; a whole-volume mask (hasBackground == false) never does.
    Z, Y, X = 8, 90, 90
    z = np.arange(Z)[:, None, None]
    y = np.arange(Y)[None, :, None]
    x = np.arange(X)[None, None, :]
    inten = (1 + (x % 256) + (y % 200) * 256 + z * 10000).astype(np.uint32)
    inten = np.broadcast_to(inten, (Z, Y, X)).astype(np.uint32)
    zz, yy, xx = np.meshgrid(np.arange(Z), np.arange(Y), np.arange(X), indexing="ij")
    cz, cy, cx = (Z - 1) / 2.0, (Y - 1) / 2.0, (X - 1) / 2.0
    rz, ry, rx = Z / 2.0, Y / 3.0, X / 3.0
    inside = (((zz - cz) / rz) ** 2 + ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2) <= 1.0
    mask = inside.astype(np.uint32)
    intp = tmp_path / "vol_int_partial.ome.tif"
    segp = tmp_path / "vol_seg_partial.ome.tif"
    tifffile.imwrite(str(intp), inten, metadata={"axes": "ZYX"})
    tifffile.imwrite(str(segp), mask, metadata={"axes": "ZYX"})
    return str(intp), str(segp)


def _ooc_vs_ram_3d(tmp_path, feats, pair_fn=_make_volume_pair):
    intp, segp = pair_fn(tmp_path)
    n_ram = nyxus.Nyxus3D(feats, ram_limit=RAM_LIMIT_LARGE_MB)
    df_ram = n_ram.featurize_files([intp], [segp], False)
    n_ooc = nyxus.Nyxus3D(feats, ram_limit=1)
    df_ooc = n_ooc.featurize_files([intp], [segp], False)
    cols, a = _feature_cols(df_ram)
    _, b = _feature_cols(df_ooc)
    assert a.size > 0 and a.shape == b.shape
    bad = [
        (c, p, q)
        for c, p, q in zip(cols, a, b)
        if abs(p - q) > 1e-6 * max(abs(p), abs(q), 1.0) + 1e-9
    ]
    assert not bad, "3D out-of-core features diverge from in-RAM: %r" % (bad[:8],)


def test_ooc_3d_glcm_matches_in_ram(tmp_path):
    """3D GLCM out-of-core (13 co-occurrence matrices built over a streaming 2-plane window)
    must match the in-RAM path."""
    _ooc_vs_ram_3d(tmp_path, ["*3D_GLCM*"])


def test_ooc_3d_gldm_matches_in_ram(tmp_path):
    """3D GLDM out-of-core (dependence matrix built over a streaming 3-plane window) must match
    the in-RAM path."""
    _ooc_vs_ram_3d(tmp_path, ["*3D_GLDM*"])


def test_ooc_3d_ngldm_matches_in_ram(tmp_path):
    """3D NGLDM out-of-core (3-plane window, interior scan) must match the in-RAM path."""
    _ooc_vs_ram_3d(tmp_path, ["*3D_NGLDM*"])


def test_ooc_3d_ngtdm_matches_in_ram(tmp_path):
    """3D NGTDM out-of-core (radius-window neighbourhood averages) must match the in-RAM path."""
    _ooc_vs_ram_3d(tmp_path, ["*3D_NGTDM*"])


def test_ooc_3d_glrlm_matches_in_ram(tmp_path):
    """3D GLRLM out-of-core (in-plane runs reuse gather_rl_zones on a per-plane depth-1 cube;
    cross-plane runs use a 2-plane carry) must match the in-RAM path."""
    _ooc_vs_ram_3d(tmp_path, ["*3D_GLRLM*"])


def test_ooc_3d_glszm_matches_in_ram(tmp_path):
    """3D GLSZM out-of-core (streaming 26-connectivity connected-component labeling via a
    growable union-find over a 2-plane window) must match the in-RAM path."""
    _ooc_vs_ram_3d(tmp_path, ["*3D_GLSZM*"])


def test_ooc_3d_gldzm_matches_in_ram(tmp_path):
    """3D GLDZM out-of-core (streaming 6-connectivity connected-component labeling with a
    min-distance-to-border metric per zone) must match the in-RAM path."""
    _ooc_vs_ram_3d(tmp_path, ["*3D_GLDZM*"])


def test_ooc_3d_wholevolume_streams_oob(tmp_path):
    """The WHOLE-VOLUME (single_roi) 3D path -- Nyxus3D.featurize_directory(dir, dir, ...), i.e.
    label_dir==intensity_dir -- used to have NO out-of-core support at all: an oversized whole
    volume failed loudly (workflow_3d_whole.cpp). It now streams via the same
    populate_3d_voxel_cloud/run_3d_ooc_features primitives as the segmented path. This is the
    Python-facing equivalent of the gtest TEST_3D_WHOLEVOLUME_OVERSIZED_STREAMS_OOC: force
    oversized with ram_limit=0 (every footprint >= 0 is always true) and compare against an
    in-RAM run of the identical volume."""
    Z, Y, X = 8, 90, 90
    z = np.arange(Z)[:, None, None]
    y = np.arange(Y)[None, :, None]
    x = np.arange(X)[None, None, :]
    inten = (1 + (x % 256) + (y % 200) * 256 + z * 10000).astype(np.uint32)
    inten = np.broadcast_to(inten, (Z, Y, X)).astype(np.uint32)
    voldir = tmp_path / "wv"
    voldir.mkdir()
    tifffile.imwrite(str(voldir / "vol.ome.tif"), inten, metadata={"axes": "ZYX"})

    feats = ["*3D_ALL_INTENSITY*", "*3D_ALL_MORPHOLOGY*", "*3D_GLCM*"]

    n_ram = nyxus.Nyxus3D(feats, ram_limit=RAM_LIMIT_LARGE_MB)
    df_ram = n_ram.featurize_directory(str(voldir), str(voldir), ".*")

    n_ooc = nyxus.Nyxus3D(feats, ram_limit=0)  # every footprint >= 0 -> always oversized
    df_ooc = n_ooc.featurize_directory(str(voldir), str(voldir), ".*")

    cols, a = _feature_cols(df_ram)
    _, b = _feature_cols(df_ooc)
    assert a.size > 0 and a.shape == b.shape

    bad = [
        (c, p, q)
        for c, p, q in zip(cols, a, b)
        if abs(p - q) > 1e-6 * max(abs(p), abs(q), 1.0) + 1e-9
    ]
    assert not bad, "whole-volume out-of-core features diverge from in-RAM: %r" % (bad[:8],)


def _make_volume_pair_blank(tmp_path):
    # A degenerate ROI (constant intensity everywhere) at the same size as _make_volume_pair, so it
    # is still classified oversized at ram_limit=1 -- exercises each osized_calculate's early-return
    # "blank ROI" guard (aux_min==aux_max) via the out-of-core path, which no other fixture reaches
    # (every other fixture here varies intensity). Must not crash or hang, and must match the
    # in-RAM path's degenerate-ROI output (STNGS_NAN defaults to plain 0.0 in this build, not IEEE
    # NaN, so the existing tolerance-based comparison in _ooc_vs_ram_3d applies unchanged).
    Z, Y, X = 8, 90, 90
    inten = np.full((Z, Y, X), 42, dtype=np.uint32)
    mask = np.ones((Z, Y, X), np.uint32)
    intp = tmp_path / "vol_int_blank.ome.tif"
    segp = tmp_path / "vol_seg_blank.ome.tif"
    tifffile.imwrite(str(intp), inten, metadata={"axes": "ZYX"})
    tifffile.imwrite(str(segp), mask, metadata={"axes": "ZYX"})
    return str(intp), str(segp)


def test_ooc_3d_blank_matches_in_ram(tmp_path):
    """A degenerate (constant-intensity) oversized 3D ROI must not crash and must match the in-RAM
    path's degenerate-ROI output for intensity, surface, and all seven texture families."""
    _ooc_vs_ram_3d(
        tmp_path,
        ["*3D_ALL_INTENSITY*", "*3D_ALL_MORPHOLOGY*", "*3D_GLCM*", "*3D_GLDM*", "*3D_NGLDM*", "*3D_NGTDM*", "*3D_GLRLM*", "*3D_GLSZM*", "*3D_GLDZM*"],
        pair_fn=_make_volume_pair_blank,
    )


@pytest.mark.skipif(
    not (DATA_NIFTI / "compat_int" / "compat_int_mri.nii").exists(),
    reason="NIfTI compat fixtures not present in tests/data/nifti",
)
def test_ooc_3d_nifti_unsupported_format_fails_loudly():
    """NIfTI delivers the whole X*Y*Z*T volume in one read (ImageLoader::stream_volume_planes
    declines it, since it isn't plane-by-plane), so an oversized ROI through this loader cannot
    stream out-of-core. Must fail loudly with an actionable message rather than crash, hang, or
    silently emit a wrong/zero row. ram_limit=0 forces every ROI oversized regardless of its actual
    footprint (roiFootprint >= 0 is always true), so this doesn't depend on the fixture's ROI size."""
    intp = str(DATA_NIFTI / "compat_int" / "compat_int_mri.nii")
    segp = str(DATA_NIFTI / "compat_seg" / "compat_seg_liver.nii")
    n = nyxus.Nyxus3D(["*3D_ALL_INTENSITY*"], ram_limit=0)
    with pytest.raises(Exception, match="not supported for this input format"):
        n.featurize_files([intp], [segp], False)


def test_ooc_3d_partial_mask_matches_in_ram(tmp_path):
    """Same equivalence check as the whole-volume tests above, but with a non-cuboid mask so the
    ROI bbox contains background voxels -- exercises grey-level LUT construction (must include the
    background bin, matching each feature's whole-cube-based calculate()) that a whole-volume mask
    never triggers. Covers intensity/surface plus all four texture families in one pass."""
    _ooc_vs_ram_3d(
        tmp_path,
        ["*3D_ALL_INTENSITY*", "*3D_ALL_MORPHOLOGY*", "*3D_GLCM*", "*3D_GLDM*", "*3D_NGLDM*", "*3D_NGTDM*", "*3D_GLRLM*", "*3D_GLSZM*", "*3D_GLDZM*"],
        pair_fn=_make_volume_pair_partial,
    )


def test_ooc_montage_oversized_fails_loudly():
    """The in-memory (montage) path has no out-of-core support, so an ROI whose footprint
    reaches ram_limit must fail loudly rather than emit a silent all-zero feature row."""
    Y, X = 300, 300
    xg = (np.arange(X) % 256).astype(np.uint32)
    yg = ((np.arange(Y) % 200) * 256).astype(np.uint32)
    inten = (1 + xg[None, :] + yg[:, None]).astype(np.uint32)
    mask = np.ones((Y, X), np.uint32)

    # sanity: with a large ram_limit the montage path succeeds and is non-zero. Set it
    # explicitly (nyxus ram_limit is process-global) so this does not depend on test order.
    ok = nyxus.Nyxus(["*ALL_INTENSITY*"])
    ok.set_params(ram_limit=RAM_LIMIT_LARGE_MB)
    df_ok = ok.featurize(inten, mask, intensity_names=["I"], label_names=["M"])
    assert df_ok["MEAN"].iloc[0] > 0

    # ram_limit=1 makes the single ROI oversized -> must raise, not return zeros
    n = nyxus.Nyxus(["*ALL_INTENSITY*"])
    n.set_params(ram_limit=1)
    with pytest.raises(Exception, match="oversized"):
        n.featurize(inten, mask, intensity_names=["I"], label_names=["M"])
