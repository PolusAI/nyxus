"""Python regression tests for the --preserve-hu / preserve_hu CT-Hounsfield mode.

Uses the committed TIFF fixtures in tests/data/hounsfield:
ct_int16.tif is a signed int16 CT image with pixel(r,c) = -1024 + idx*8,
idx = r*16 + c (values -1024..1016, crossing 0). preserve_hu must be exercised
through the FILE loader (numpy input bypasses the load-time quantization), so
these use featurize_files.

In HU mode the loader maps value -> value - floor(min) = value + 1024 = idx*8, so
the offset-domain feature pixels are 0..2040 (MEAN 1020). Without it, the negative
int16 values wrap on the unsigned cast into billions.
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


def test_preserve_hu_offset_domain_values():
    # Offset-domain: pixel(idx) = idx*8, idx 0..255 -> MIN 0, MAX 2040, MEAN 1020.
    f = _featurize(True)
    assert f["MIN"] == pytest.approx(0.0)
    assert f["MAX"] == pytest.approx(2040.0)
    assert f["MEAN"] == pytest.approx(1020.0)
    assert f["INTEGRATED_INTENSITY"] == pytest.approx(1020.0 * 256)


def test_preserve_hu_no_wraparound():
    # The whole point: negative CT values must not wrap into billions.
    f = _featurize(True)
    assert f["MIN"] < f["MAX"] < 1e6
    assert f["MEAN"] < 1e6


def test_preserve_hu_differs_from_default_mapping():
    # On a FLOAT CT image both mappings are well-defined and must DIFFER:
    #   preserve_hu=True  -> slope-1 offset domain          (MEAN 1020)
    #   preserve_hu=False -> min-max rescale into [0, DR]    (MEAN ~5000)
    # We compare on the float fixture on purpose: the signed-int16 no-flag path
    # instead casts -1024 to unsigned and wraps to ~4.29e9, and pushing those absurd
    # intensities through the pipeline can SEGFAULT on some platforms (macOS) -- which
    # is exactly the breakage --preserve-hu exists to prevent. So we do NOT featurize
    # that path here; the correct-HU values are pinned in test_preserve_hu_offset_domain_values.
    on = _featurize(True, FLOAT)
    off = _featurize(False, FLOAT)
    assert on["MEAN"] != pytest.approx(off["MEAN"])
    assert on["MEAN"] < 1e6 and off["MEAN"] < 1e6   # both stay sane, no wraparound
