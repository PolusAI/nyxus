"""Negative-input mechanics: a real-valued slide holding NaN or an infinity.

A float TIFF is free to contain non-finite samples, and one of them used to take the whole
slide with it. The prescan folded every sample into the slide's min/max, so a single +Inf or
-Inf gave the recorded quantized map an infinite span; every finite pixel then mapped to NaN,
the unsigned conversion of which is undefined, and the entire ROI came back as zeros -- MIN,
MAX and MEAN all 0 on a slide whose real values ran from 0.5 to 10.5. Silently: nothing in
the output said the data had been lost.

The scan now measures the extrema over the finite samples only, and every load-time map
stores a non-finite sample as grey level 0 rather than converting it. So the finite pixels
keep their own values and a stray infinity costs one pixel instead of the slide.

Kind: *mechanics* per tests/vetting/SPEC.md 2 -- these pin loader behaviour on malformed
input, not feature values against a reference, and establish no vetting.
"""
import math

import numpy as np
import pytest
import nyxus

tifffile = pytest.importorskip("tifffile")

H, W = 32, 32

# The finite content of every slide below: a ramp whose ends are exactly known.
LO, HI = 0.5, 10.5


def _ramp():
    return np.linspace(LO, HI, H * W).astype(np.float32).reshape(H, W)


def _write(tmp_path, tag, inten):
    lab = np.ones((H, W), np.uint16)
    ip = tmp_path / (tag + ".tif")
    sp = tmp_path / (tag + "_seg.tif")
    tifffile.imwrite(str(ip), inten)
    tifffile.imwrite(str(sp), lab)
    return [str(ip)], [str(sp)]


def _featurize(tmp_path, tag, inten, single_roi=False):
    ifiles, sfiles = _write(tmp_path, tag, inten)
    nyx = nyxus.Nyxus(features=["MIN", "MAX", "MEAN", "RANGE"], n_feature_calc_threads=1)
    return nyx.featurize_files(ifiles, sfiles, single_roi).iloc[0]


def test_2d_nonfinite_clean_float_baseline_mechanics(tmp_path):
    """The control: without a non-finite sample the ramp is reported as it is."""
    row = _featurize(tmp_path, "clean", _ramp())
    assert row["MIN"] == pytest.approx(LO, rel=1e-3)
    assert row["MAX"] == pytest.approx(HI, rel=1e-3)
    assert row["MEAN"] == pytest.approx((LO + HI) / 2.0, rel=1e-3)


@pytest.mark.parametrize("tag,value", [
    ("pos_inf", np.inf),
    ("neg_inf", -np.inf),
    ("nan", np.nan),
])
def test_2d_nonfinite_pixel_does_not_destroy_the_slide_mechanics(tmp_path, tag, value):
    """One non-finite sample must cost one pixel, not the whole slide.

    Before the scan skipped them, +Inf and -Inf each collapsed MIN/MAX/MEAN to 0.
    """
    inten = _ramp()
    inten[3, 3] = value
    row = _featurize(tmp_path, tag, inten)

    assert math.isfinite(row["MIN"]) and math.isfinite(row["MAX"]) and math.isfinite(row["MEAN"])

    # the finite extrema of the ramp survive, rather than collapsing to 0
    assert row["MAX"] == pytest.approx(HI, rel=1e-3), \
        "a single %s took the slide's maximum with it" % tag
    assert row["MEAN"] == pytest.approx((LO + HI) / 2.0, rel=2e-2)
    assert row["RANGE"] > 0.0


def test_2d_nonfinite_all_pixels_mechanics(tmp_path):
    """A slide with nothing finite in it: no range to measure, and no crash either.

    The exact values matter here, not just their finiteness. Neither extremum is ever
    assigned on such a slide, so both keep the sentinel the scan starts from; left there,
    the recorded map offsets by DBL_MAX and the intensity family adds it back, which put
    1.797e308 in MIN, MAX and MEAN. That is a finite number, so the output sanitizer let it
    through and an ``isfinite`` assertion passed on it. The scan settles a slide that
    measured nothing to a flat zero range instead, which leaves the map at the identity --
    so what comes out is the grey level every non-finite sample was stored as.
    """
    inten = np.full((H, W), np.nan, np.float32)
    row = _featurize(tmp_path, "allnan", inten)

    for f in ("MIN", "MAX", "MEAN", "RANGE"):
        assert row[f] == 0.0, "%s is %r, not 0" % (f, row[f])


def test_2d_nonfinite_all_pixels_whole_slide_mechanics(tmp_path):
    """The same slide with no mask, which reaches the scanned extrema a second way.

    Whole-slide mode synthesises its one ROI and takes that ROI's grey-level range straight
    from the scanned extrema through ``SlideProps::to_grey_level()``, so the sentinels get a
    second consumer here -- and it is the range every texture family then bins against, not
    only what first-order reports. On this configuration the recorded offset is itself the
    sentinel and cancels, so the grey levels come out 0 either way and it is the source-domain
    report that carried the 1.797e308. On an offset map whose offset does not cancel -- the same
    slide under ``--preserve-hu``, where the recorder leaves the offset at 0 because the
    all-pixel minimum is a sentinel and so not negative -- the same call converts DBL_MAX to an
    unsigned grey level, which is undefined. Recording a zero range settles both.
    """
    inten = np.full((H, W), np.nan, np.float32)
    row = _featurize(tmp_path, "allnan_wsi", inten, single_roi=True)

    for f in ("MIN", "MAX", "MEAN", "RANGE"):
        assert row[f] == 0.0, "%s is %r, not 0" % (f, row[f])
