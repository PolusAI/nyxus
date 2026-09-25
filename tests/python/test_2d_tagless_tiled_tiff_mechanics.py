"""Mechanics: a tiled TIFF that omits SampleFormat (tag 339) loads in both request modes.

TIFF 6.0 makes tag 339 optional and defines its absence as 1, unsigned integer, so a file
without it is conformant -- and that is what writers emit for unsigned data: `tifffile`
declines to write the tag at all in that case, which is why the checked-in tiled masks
tests/data/hounsfield/ct_small_mask.tif and mask.tif carry no 339.

Which loader a run reaches is decided by the file's LAYOUT and by whether the request is
whole-image (`single_roi=True`, image_loader1x -> the strip loader whatever the layout) or
segmented (`single_roi=False`, ImageLoader -> the tile loader for a tiled file). Neither of
those says anything about the pixel format, so the absent tag has to mean the same thing on
every one of those routes.

ct_small_mask.tif is tiled (64x64 tiles), tagless, and a single all-ones label over the whole
128x128 frame, so the segmented request produces exactly one ROI covering the same pixels as
the whole-image request. That makes the two modes directly comparable, and the intensity
goldens are the pydicom-pinned HU values of test_2d_hu_ct_small_pydicom.py.

Fixtures: tests/data/hounsfield/{ct_small_hu.tif, ct_small_mask.tif}.
"""
import os
import pathlib
import pytest
import nyxus

DATA = pathlib.Path(__file__).resolve().parent.parent / "data" / "hounsfield"
INTEN = str(DATA / "ct_small_hu.tif")
MASK = str(DATA / "ct_small_mask.tif")

pytestmark = pytest.mark.skipif(
    not (os.path.exists(INTEN) and os.path.exists(MASK)),
    reason="CT_small HU fixtures not present in tests/data/hounsfield",
)

FEATS = ["MIN", "MAX", "MEAN", "INTEGRATED_INTENSITY"]

# pydicom-pinned absolute HU over the whole 128x128 frame (see test_2d_hu_ct_small_pydicom.py)
HU_MIN = -896
HU_MAX = 1167
HU_MEAN = -119.0738525390625
HU_INTEGRATED = -1950906


def _featurize(single_roi):
    nyx = nyxus.Nyxus(FEATS, preserve_hu=True)
    return nyx.featurize_files([INTEN], [MASK], single_roi)


def test_2d_tagless_tiled_tiff_segmented_mode_loads_mechanics():
    """The segmented route reaches the tile loader, which reads the absent tag as unsigned."""
    df = _featurize(False)
    assert len(df) == 1, "the all-ones mask is one ROI covering the frame"


def test_2d_tagless_tiled_tiff_segmented_mode_values_mechanics():
    """That ROI is the whole frame, so its intensities are the pydicom HU goldens."""
    df = _featurize(False)
    o = {c: float(df[c].iloc[0]) for c in FEATS}

    assert o["MIN"] == pytest.approx(HU_MIN)
    assert o["MAX"] == pytest.approx(HU_MAX)
    assert o["MEAN"] == pytest.approx(HU_MEAN, rel=1e-6)
    assert o["INTEGRATED_INTENSITY"] == pytest.approx(HU_INTEGRATED)


def test_2d_tagless_tiled_tiff_modes_agree_mechanics():
    """The request mode picks a loader, not a pixel format: same file, same numbers."""
    whole = _featurize(True)
    segmented = _featurize(False)

    for c in FEATS:
        assert float(segmented[c].iloc[0]) == pytest.approx(float(whole[c].iloc[0])), c
