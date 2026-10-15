"""Mechanics of coarse_gray_depth: which values the API takes, and where the value reaches.

The sign of the coarse grey depth selects the texture binning -- a positive value bins into that many
MATLAB-style levels, 0 bins nothing, and a negative value bins into abs(value) PyRadiomics-style bins
over each ROI's range -- and every 2D and 3D texture family starts from it. The intensity histograms
take only its magnitude as a bin count, and their default of 24 bins at 0.

Kind: *mechanics* per tests/vetting/SPEC.md 2. Each case compares two routes to one configuration, or
one configuration to its definition, and claims no oracle of its own; the oracle rows for the
binned values themselves are the families' MIRP and PyRadiomics tests.
"""
import math
import pathlib

import numpy as np
import pytest
import nyxus

DATA = pathlib.Path(__file__).resolve().parent.parent / "data" / "nifti"
COMPAT_INT = str(DATA / "compat_int" / "compat_int_mri.nii")
COMPAT_SEG = str(DATA / "compat_seg" / "compat_seg_liver.nii")
UT_INT = str(DATA / "phantoms" / "ut_inten.nii")
UT_SEG = str(DATA / "phantoms" / "ut_mask57.nii")

# the 3D families that read a grey depth of their own, with the metaparameter that sets it
FAMILIES_3D = {"3GLRLM_": "3glrlm", "3GLSZM_": "3glszm", "3GLDM_": "3gldm", "3NGTDM_": "3ngtdm"}

FIRST_ORDER = ["ENTROPY", "UNIFORMITY", "P01", "P10", "P25", "P75", "P90", "P99", "MEDIAN",
               "INTERQUARTILE_RANGE", "ROBUST_MEAN_ABSOLUTE_DEVIATION"]


def _gradient_disk():
    """A 64x64 disk over a gradient with fixed noise: many grey levels, so every binning moves it."""
    rng = np.random.default_rng(7)
    yy, xx = np.mgrid[0:64, 0:64]
    img = (500 + 10 * xx + 5 * yy + rng.integers(0, 200, size=(64, 64))).astype(np.uint16)
    mask = np.zeros((64, 64), dtype=np.uint16)
    mask[(yy - 32) ** 2 + (xx - 32) ** 2 < 28 ** 2] = 1
    return img, mask


def _row_3d(nyx, intensity, segmentation, label):
    df = nyx.featurize_files([intensity], [segmentation], False)
    col = "ROI_label" if "ROI_label" in df.columns else "label"
    return df[df[col] == label].iloc[0]


@pytest.mark.parametrize("depth", [-20, 0, 20])
def test_coarse_gray_depth_every_sign_accepted_mechanics(depth):
    for cls in (nyxus.Nyxus, nyxus.Nyxus3D, nyxus.ImageQuality):
        nyx = cls(["*ALL*"] if cls is nyxus.Nyxus else
                  ["*3D_ALL*"] if cls is nyxus.Nyxus3D else ["*ALL_IMQ*"],
                  coarse_gray_depth=depth)
        assert nyx.get_params("coarse_gray_depth")["coarse_gray_depth"] == depth


# The refusal path: a value that is no 32-bit integer, and the one 32-bit integer whose magnitude
# is no 32-bit integer.
@pytest.mark.parametrize("bad", [2.5, True, "64", None, -2**31, 2**31])
def test_coarse_gray_depth_non_integer_refused_mechanics(bad):
    for cls in (nyxus.Nyxus, nyxus.Nyxus3D, nyxus.ImageQuality):
        with pytest.raises(ValueError, match="coarse_gray_depth must be a 32-bit integer"):
            cls(["*ALL*"] if cls is nyxus.Nyxus else
                ["*3D_ALL*"] if cls is nyxus.Nyxus3D else ["*ALL_IMQ*"],
                coarse_gray_depth=bad)


# set_params() refuses a coarse grey depth other than the object's, whatever its sign, and a refused
# call applies none of its other parameters either. The depth is copied into every family's settings
# at construction and set_params() does not rebuild them, so an accepted new value would change the
# HISTOGRAM column count and nothing that is binned.
@pytest.mark.parametrize("depth", [-20, 0, 32])
def test_coarse_gray_depth_set_params_refused_mechanics(depth):
    for cls in (nyxus.Nyxus, nyxus.Nyxus3D, nyxus.ImageQuality):
        nyx = cls(["*ALL*"] if cls is nyxus.Nyxus else
                  ["*3D_ALL*"] if cls is nyxus.Nyxus3D else ["*ALL_IMQ*"])
        before = nyx.get_params("coarse_gray_depth", "neighbor_distance")
        with pytest.raises(ValueError, match="pass it to the constructor"):
            nyx.set_params(coarse_gray_depth=depth, neighbor_distance=before["neighbor_distance"] + 3)
        with pytest.raises(ValueError, match="pass it to the constructor"):
            nyx.set_environment_params(coarse_gray_depth=depth)
        assert nyx.get_params("coarse_gray_depth", "neighbor_distance") == before


# Negative case: the object's own depth changes nothing, so it is accepted beside the parameters
# set_params() applies, and those still reach the object.
def test_coarse_gray_depth_set_params_current_value_accepted_mechanics():
    for cls in (nyxus.Nyxus, nyxus.Nyxus3D, nyxus.ImageQuality):
        nyx = cls(["*ALL*"] if cls is nyxus.Nyxus else
                  ["*3D_ALL*"] if cls is nyxus.Nyxus3D else ["*ALL_IMQ*"])
        p = nyx.get_params("coarse_gray_depth", "neighbor_distance")
        nyx.set_params(coarse_gray_depth=p["coarse_gray_depth"], neighbor_distance=p["neighbor_distance"] + 3)
        nyx.set_environment_params(coarse_gray_depth=p["coarse_gray_depth"])
        after = nyx.get_params("coarse_gray_depth", "neighbor_distance")
        assert after == {"coarse_gray_depth": p["coarse_gray_depth"],
                         "neighbor_distance": p["neighbor_distance"] + 3}


# and so a Nyxus object's own get_params() goes back through set_params() unchanged
def test_coarse_gray_depth_get_params_round_trip_mechanics():
    nyx = nyxus.Nyxus(["*ALL*"], coarse_gray_depth=-20)
    before = nyx.get_params()
    nyx.set_params(**before)
    assert nyx.get_params() == before


# Through the constructor, the coarse grey depth bins the four 3D families with a grey depth of their
# own exactly as each family's own metaparameter does. -20 is the configuration their PyRadiomics
# goldens are pinned at (binCount 20), so the two routes have to agree to the last bit, and on
# every feature of all four families.
@pytest.mark.skipif(not pathlib.Path(COMPAT_INT).exists(), reason="compat phantom not present")
def test_coarse_gray_depth_reaches_3d_texture_families_mechanics():
    groups = ["*3D_GLRLM*", "*3D_GLSZM*", "*3D_GLDM*", "*3D_NGTDM*"]

    by_flag = _row_3d(nyxus.Nyxus3D(groups, coarse_gray_depth=-20), COMPAT_INT, COMPAT_SEG, 1)

    per_family = nyxus.Nyxus3D(groups)
    for meta in FAMILIES_3D.values():
        per_family.set_metaparam(f"{meta}/greydepth=-20")
    by_metaparam = _row_3d(per_family, COMPAT_INT, COMPAT_SEG, 1)

    # and the default 64 is a different configuration, so the comparison above is not vacuous
    by_default = _row_3d(nyxus.Nyxus3D(groups), COMPAT_INT, COMPAT_SEG, 1)

    for prefix in FAMILIES_3D:
        cols = [c for c in by_flag.index if c.startswith(prefix)]
        assert cols, prefix
        for c in cols:
            assert by_flag[c] == by_metaparam[c], c
        assert any(by_flag[c] != by_default[c] for c in cols), prefix


# At 0 the intensity histograms keep their default bin count, so every statistic read off them equals
# the run at 24 bins. A 0 used as a bin count would leave the histogram one slot long and write
# before it.
def test_coarse_gray_depth_zero_first_order_2d_mechanics():
    img, mask = _gradient_disk()
    at0 = nyxus.Nyxus(["*ALL_INTENSITY*"], coarse_gray_depth=0).featurize(img, mask).iloc[0]
    at24 = nyxus.Nyxus(["*ALL_INTENSITY*"], coarse_gray_depth=24).featurize(img, mask).iloc[0]
    for f in FIRST_ORDER:
        assert math.isfinite(at0[f]), f
        assert at0[f] == at24[f], f


@pytest.mark.skipif(not pathlib.Path(UT_INT).exists(), reason="ut_ phantom not present")
def test_coarse_gray_depth_zero_first_order_3d_mechanics():
    at0 = _row_3d(nyxus.Nyxus3D(["*3D_ALL_INTENSITY*"], coarse_gray_depth=0), UT_INT, UT_SEG, 57)
    at24 = _row_3d(nyxus.Nyxus3D(["*3D_ALL_INTENSITY*"], coarse_gray_depth=24), UT_INT, UT_SEG, 57)
    for f in FIRST_ORDER:
        assert math.isfinite(at0["3" + f]), f
        assert at0["3" + f] == at24["3" + f], f


# A negative depth gives the HISTOGRAM feature abs(depth) bins, and the output has one column per bin
# holding them in order: the same columns and values as the positive depth, summing to the ROI's
# pixel count. MEAN is requested beside it because HISTOGRAM on its own comes back as zeros: it belongs to
# no featureset that triggers the intensity family's calculation.
def test_coarse_gray_depth_negative_histogram_columns_mechanics():
    img, mask = _gradient_disk()
    neg = nyxus.Nyxus(["MEAN", "HISTOGRAM"], coarse_gray_depth=-20).featurize(img, mask)
    pos = nyxus.Nyxus(["MEAN", "HISTOGRAM"], coarse_gray_depth=20).featurize(img, mask)
    bins = [f"HISTOGRAM_BIN_{i}" for i in range(20)]
    assert [c for c in neg.columns if c.startswith("HISTOGRAM")] == bins
    for c in bins:
        assert neg[c].iloc[0] == pos[c].iloc[0], c
    assert sum(neg[c].iloc[0] for c in bins) == int(mask.sum())
