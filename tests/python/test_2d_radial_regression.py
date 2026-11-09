"""Whole-slide (single-ROI) radial distribution through the file pipeline, pinned.

Kind: *regression* per tests/vetting/SPEC.md 2 -- this pins CURRENT behaviour and establishes no
vetting. `SINGLEROI=true` is a reachable production configuration for the radial features (the
Nyxus whole-slide mode), and this is the one test that runs it end to end, through
`featurize_files(..., single_roi=True)`.

What the cell computes. At `SINGLEROI=true` the contour is the four corners of the slide's box
(`ContourFeature::buildWholeSlideContour()`). `Pixel2::find_center()` takes as the centre the pixel
whose distances to those corners spread least, which is the middle of the slide, and the pixels are
binned in rings around it, so on a uniform 20 x 20 slide the counts rise and then fall again towards
the corners. The distance searches scan a walk of up to 10 points exhaustively; they used to compare
corners 0 and 2 only, which put the centre on the slide's top edge and made the fractions grow
monotonically outwards (https://github.com/PolusAI/nyxus/issues/516).

docs/examples/radial_worked_example.py checks the same mode against rings computed from the centre,
on this slide and on a 64 x 48 one; this pin keeps the 20 x 20 counts in the test tree.
"""
import os

import numpy as np
import pytest

import nyxus

tifffile = pytest.importorskip("tifffile")

SIDE = 20
# FRAC_AT_D_0 .. FRAC_AT_D_7 on the uniform 20 x 20 slide: pixels per ring over the 400 of the slide
WHOLE_SLIDE_COUNTS = [13, 36, 64, 100, 106, 64, 16, 1]


def test_2d_radial_wholeslide_uniform_regression(tmp_path):
    """FRAC_AT_D of a uniform whole slide, pinned as pixel counts out of the slide's 400."""
    image = np.full((SIDE, SIDE), 100, np.uint32)
    ip, sp = os.path.join(tmp_path, "int.tif"), os.path.join(tmp_path, "seg.tif")
    tifffile.imwrite(ip, image)
    tifffile.imwrite(sp, np.ones((SIDE, SIDE), np.uint32))

    df = nyxus.Nyxus(["FRAC_AT_D"]).featurize_files([ip], [sp], True)
    assert len(df) == 1
    got = [float(df[f"FRAC_AT_D_{i}"].iloc[0]) for i in range(len(WHOLE_SLIDE_COUNTS))]

    assert sum(WHOLE_SLIDE_COUNTS) == SIDE * SIDE
    for i, (g, n) in enumerate(zip(got, WHOLE_SLIDE_COUNTS)):
        assert g == pytest.approx(n / (SIDE * SIDE), rel=1e-9), f"FRAC_AT_D_{i}"

    # rings around the middle: the fractions rise to a peak and fall again, which a centre off the
    # middle (fractions growing towards the far side) does not give
    peak = got.index(max(got))
    assert 0 < peak < len(got) - 1
    assert all(got[i] < got[i + 1] for i in range(peak))
    assert all(got[i] > got[i + 1] for i in range(peak, len(got) - 1))
