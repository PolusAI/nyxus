"""Invariant test for the image-quality focus scores on the out-of-core path: in-RAM == out-of-core.

An ROI past ram_limit is scored by FocusScoreFeature::osized_calculate() from a disk-backed copy of
its pixels instead of by calculate(). Both hand the same bounding-box image to the same scoring code,
so FOCUS_SCORE and LOCAL_FOCUS_SCORE must come out equal - exactly, not within a band.

Kind: *invariant* per tests/vetting/SPEC.md 2. It relates two Nyxus code paths and vets nothing; the
in-RAM scores are vetted in tests/test_imq_opencv.h and tests/test_imq_analytic.h. The C++ twin,
tests/test_imq_invariant.h, calls osized_calculate() directly; this one goes through
ImageQuality.featurize_directory, so the real oversized-ROI loop, its settings lookup and its single
long-lived feature instance are what run.

ram_limit is a process-global in Nyxus, so each side sets it explicitly to stay order-independent.
"""
import os

import numpy as np
import pytest

import nyxus

tifffile = pytest.importorskip("tifffile")

# The in-RAM side's ram_limit. The image here is a few kB, so this only has to clear that; Nyxus
# refuses a limit above the RAM currently available, which is why it is not sized generously.
RAM_LIMIT_LARGE_MB = 64

# Only the two features under test are requested. The ROIs here clear POWER_SPECTRUM_SLOPE's
# 24 px guard, and the algorithm behind that guard indexes past its radius array (matrix/imq.md).
FOCUS_FEATURES = ["FOCUS_SCORE", "LOCAL_FOCUS_SCORE"]


def _image_quality(ram_limit_mb):
    """An ImageQuality instance with the process-global ram_limit set, verified to have taken.

    ImageQuality takes ram_limit at construction only - its set_params() does not accept it - and
    Nyxus keeps the previous value on a refusal without raising, so the value is read back."""
    nyx = nyxus.ImageQuality(FOCUS_FEATURES, ram_limit=ram_limit_mb)
    got = nyx.get_params("ram_limit")["ram_limit"]
    assert got == ram_limit_mb, "ram_limit=%d MB was not accepted (still %d MB)" % (ram_limit_mb, got)
    return nyx


def _make_pair(tmp_path):
    """One 80 x 60 slide carrying three ROIs, in this label order:

    1. a 70 x 45 block of pseudo-random 16-bit intensities - textured, and wider and taller than
       30 px, so every tile of the local grid carries signal;
    2. a 6 x 6 constant block - an ROI whose min equals its max;
    3. a 5 x 1 row with one bright pixel - too thin for any tile, so LOCAL_FOCUS_SCORE is the
       soft-NaN on both paths while FOCUS_SCORE is still a number.

    The textured ROI comes first, so a score left over from it on the shared feature instance cannot
    pass for the constant or thin ROI's own.
    """
    rng = np.random.default_rng(479)
    inten = np.zeros((60, 80), np.uint16)
    mask = np.zeros((60, 80), np.uint16)

    inten[2:47, 2:72] = rng.integers(0, 65536, size=(45, 70), dtype=np.uint16)
    mask[2:47, 2:72] = 1

    inten[50:56, 2:8] = 7
    mask[50:56, 2:8] = 2

    inten[58, 20:25] = 0
    inten[58, 22] = 6
    mask[58, 20:25] = 3

    intdir = tmp_path / "int"
    segdir = tmp_path / "seg"
    intdir.mkdir()
    segdir.mkdir()
    tifffile.imwrite(str(intdir / "img.tif"), inten)
    tifffile.imwrite(str(segdir / "img.tif"), mask)
    return str(intdir) + os.sep, str(segdir) + os.sep


def _focus_scores(df):
    return df.sort_values("ROI_label")[["ROI_label"] + FOCUS_FEATURES].reset_index(drop=True)


def test_imq_focus_score_out_of_core_invariant(tmp_path):
    intdir, segdir = _make_pair(tmp_path)

    n_ram = _image_quality(RAM_LIMIT_LARGE_MB)     # large -> in-RAM calculate()
    ram = _focus_scores(n_ram.featurize_directory(intdir, segdir))

    n_ooc = _image_quality(0)                      # 0 -> every ROI through osized_calculate()
    ooc = _focus_scores(n_ooc.featurize_directory(intdir, segdir))

    assert ram["ROI_label"].tolist() == [1, 2, 3]
    assert ooc["ROI_label"].tolist() == [1, 2, 3]

    # The fixture has to reach the cells it is built for, or equality proves nothing about them
    assert ram.loc[0, "LOCAL_FOCUS_SCORE"] > 0, "the textured ROI's tiles carry no signal"
    assert ram.loc[2, "FOCUS_SCORE"] > 0, "the thin ROI's FOCUS_SCORE is not computed"
    assert ram.loc[2, "LOCAL_FOCUS_SCORE"] != ram.loc[0, "LOCAL_FOCUS_SCORE"]

    for feat in FOCUS_FEATURES:
        a = ram[feat].to_numpy(dtype=float)
        b = ooc[feat].to_numpy(dtype=float)
        # exact; the thin ROI's LOCAL_FOCUS_SCORE is the soft-NaN on both paths (--noval, 0 by
        # default), and assert_array_equal would also take NaN == NaN
        np.testing.assert_array_equal(b, a, err_msg=feat + " differs between out-of-core and in-RAM")
