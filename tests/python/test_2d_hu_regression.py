"""Python regression tests for CT-Hounsfield handling on 2D TIFF slides.

Uses the committed TIFF fixtures in tests/data/hounsfield:
ct_int16.tif is a signed int16 CT image with pixel(r,c) = -1024 + idx*8,
idx = r*16 + c (values -1024..1016, crossing 0). The load-time map has to be exercised
through the FILE loader (numpy input bypasses it), so these use featurize_files.

The loader offsets the image by its own floored minimum, u = value + 1024 = idx*8, and the
intensity family adds that offset back, so the reported features are in Hounsfield units:
MIN -1024, MAX 1016, MEAN -4. Nothing here depends on --preserve-hu any more: an image
holding negative values is offset-preserved either way, and the flag now only decides whether
a REAL-VALUED image is min-max rescaled or carried on that same offset map.
"""
import os
import pathlib
import pytest
import nyxus

DATA = pathlib.Path(__file__).resolve().parent.parent / "data" / "hounsfield"
INTEN = str(DATA / "ct_int16.tif")
FLOAT = str(DATA / "ct_float.tif")
MASK = str(DATA / "mask.tif")

pytestmark = pytest.mark.skipif(
    not (os.path.exists(INTEN) and os.path.exists(FLOAT) and os.path.exists(MASK)),
    reason="Hounsfield TIFF fixtures not present in tests/data/hounsfield",
)

FEATS = ["INTEGRATED_INTENSITY", "MEAN", "MAX", "MIN"]


def _featurize(preserve_hu, inten=INTEN):
    nyx = nyxus.Nyxus(FEATS, preserve_hu=preserve_hu)
    df = nyx.featurize_files([inten], [MASK], True)
    return {c: float(df[c].iloc[0]) for c in FEATS}


def test_2d_hu_absolute_hounsfield_values_regression():
    # pixel(idx) = -1024 + idx*8, idx 0..255 -> MIN -1024, MAX 1016, MEAN -4.
    f = _featurize(True)
    assert f["MIN"] == pytest.approx(-1024.0)
    assert f["MAX"] == pytest.approx(1016.0)
    assert f["MEAN"] == pytest.approx(-4.0)
    assert f["INTEGRATED_INTENSITY"] == pytest.approx(-4.0 * 256)


def test_2d_hu_preserve_no_wraparound_regression():
    # The whole point: negative CT values must not wrap into billions.
    f = _featurize(True)
    assert -2000.0 < f["MIN"] < f["MAX"] < 1e6
    assert abs(f["MEAN"]) < 1e6


def test_2d_hu_signed_int_needs_no_flag_regression():
    # A signed image is offset-preserved whether or not --preserve-hu is given, so both
    # invocations report the same absolute Hounsfield values. Before the load-time map was
    # recorded and inverted, the no-flag path clamped every negative pixel to 0 instead.
    on = _featurize(True)
    off = _featurize(False)
    assert off["MIN"] == pytest.approx(on["MIN"])
    assert off["MEAN"] == pytest.approx(on["MEAN"])
    assert off["MIN"] == pytest.approx(-1024.0)


def test_2d_hu_float_mappings_agree_after_inversion_regression():
    # On a FLOAT CT image the two modes still take DIFFERENT load-time maps --
    #   preserve_hu=True  -> slope-1 offset (1 grey level == 1 HU)
    #   preserve_hu=False -> min-max rescale into [0, DR]
    # -- but both are now inverted on the way out, so the reported values agree to within one
    # quantization step of the coarser map ((1016+1024)/10000 = 0.204 HU).
    on = _featurize(True, FLOAT)
    off = _featurize(False, FLOAT)
    assert off["MEAN"] == pytest.approx(on["MEAN"], abs=1.0)
    assert off["MIN"] == pytest.approx(on["MIN"], abs=1.0)
    assert on["MIN"] == pytest.approx(-1024.0)
