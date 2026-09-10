"""Python regression tests for CT-Hounsfield handling on 3D NIfTI volumes.

Uses the committed synthetic NIfTI fixtures in tests/data/hounsfield:
ct3d_int16.nii is an 8x8x8 signed int16 volume with stored pixel(idx) = idx - 200 (idx 0..511),
so stored values run -200..311 and cross zero. Its header sets scl_slope=2, scl_inter=-1024, i.e.
true HU = 2*stored - 1024 (range -1424..-402). mask3d.nii is an all-ones ROI over the whole volume.

The NIfTI loader always rescales to those true Hounsfield values, then offsets the volume by its
own floored minimum so the negatives survive the unsigned cast; the intensity family adds the
offset back. Reported features are therefore absolute HU: MIN -1424, MAX -402, MEAN -913. The
non-unit scl_slope is what makes the rescale observable, and it no longer depends on
--preserve-hu: a volume in physical units is read in physical units either way.
"""
import os
import pathlib
import pytest
import nyxus

DATA = pathlib.Path(__file__).resolve().parent.parent / "data" / "hounsfield"
INTEN = str(DATA / "ct3d_int16.nii")
MASK = str(DATA / "mask3d.nii")

pytestmark = pytest.mark.skipif(
    not (os.path.exists(INTEN) and os.path.exists(MASK)),
    reason="NIfTI Hounsfield fixtures not present in tests/data/hounsfield",
)

FEATS = ["3MIN", "3MAX", "3MEAN", "3INTEGRATED_INTENSITY"]


def _featurize(preserve_hu):
    nyx = nyxus.Nyxus3D(FEATS, preserve_hu=preserve_hu)
    df = nyx.featurize_files([INTEN], [MASK], False)  # explicit whole-volume mask
    return {c: float(df[c].iloc[0]) for c in FEATS}


def test_3d_hu_nifti_absolute_hounsfield_values_regression():
    # HU = 2*stored - 1024 over stored -200..311 -> -1424..-402, mean -913 over 512 voxels.
    f = _featurize(True)
    assert f["3MIN"] == pytest.approx(-1424.0)
    assert f["3MAX"] == pytest.approx(-402.0)
    assert f["3MEAN"] == pytest.approx(-913.0)
    assert f["3INTEGRATED_INTENSITY"] == pytest.approx(-913.0 * 512)


def test_3d_hu_nifti_preserve_no_wraparound_regression():
    # The negative signed int16 values must not wrap into billions on the unsigned cast.
    f = _featurize(True)
    assert -1e6 < f["3MIN"] < f["3MAX"] < 0.0
    assert abs(f["3MEAN"]) < 1e6


def test_3d_hu_nifti_rescale_needs_no_flag_regression():
    # scl_slope/scl_inter are part of what the file means, so they are applied whether or not
    # --preserve-hu is given. The flag used to gate them, which left the default path reporting
    # raw stored values shifted by an offset nothing undid.
    on = _featurize(True)
    off = _featurize(False)
    assert off["3MAX"] == pytest.approx(on["3MAX"])
    assert off["3MEAN"] == pytest.approx(on["3MEAN"])
    assert off["3MAX"] == pytest.approx(-402.0)
