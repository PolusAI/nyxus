"""Mechanics tests for anisotropy through the Python bindings.

The bindings pass the anisotropy factors to the C++ core in double precision, as the command line
does. The 2D resampling maps a virtual column vc back to the pixel column int(vc / ax), so a factor
that is not exact in single precision, such as 1.3, selects other pixels at single precision than
at double; and the as-acquired features weigh each pixel by ax * ay, which single precision moves by
about 1e-8.

- test_2d_anisotropy_python_factors_double_mechanics: at (1.3, 0.8), AREA_PIXELS_COUNT is the
  resampled pixel count at double precision and IMOM_RM_00 is the intensity sum times ax * ay to
  1e-12. The fixture checks that the single-precision count differs, so the test discriminates.
"""
import math
import os

import numpy as np
import pytest

import nyxus

tifffile = pytest.importorskip("tifffile")

W, H = 64, 64
AX, AY = 1.3, 0.8


def _fixture():
    # a tilted ellipse with an intensity gradient
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    c, s = math.cos(0.5), math.sin(0.5)
    u = (xx - 30) * c + (yy - 33) * s
    v = -(xx - 30) * s + (yy - 33) * c
    mask = ((u / 22) ** 2 + (v / 11) ** 2 <= 1).astype(np.uint16)
    inten = (10 + 3 * xx + 5 * yy + (xx * yy) % 7).astype(np.uint16) * mask
    return inten, mask


def _resampled_count(mask, ax, ay):
    # every virtual pixel below int(extent * factor) carries the pixel its coordinate truncates to
    vc = np.arange(int(W * ax))
    vr = np.arange(int(H * ay))
    cols = (vc / ax).astype(int)
    rows = (vr / ay).astype(int)
    return int(mask[np.ix_(rows, cols)].sum())


def test_2d_anisotropy_python_factors_double_mechanics(tmp_path):
    inten, mask = _fixture()
    single = _resampled_count(mask, float(np.float32(AX)), float(np.float32(AY)))
    double = _resampled_count(mask, AX, AY)
    assert single != double, "the fixture does not tell single from double precision apart"

    intdir, segdir = tmp_path / "int", tmp_path / "seg"
    intdir.mkdir()
    segdir.mkdir()
    tifffile.imwrite(str(intdir / "a.tif"), inten)
    tifffile.imwrite(str(segdir / "a.tif"), mask)

    nyx = nyxus.Nyxus(["AREA_PIXELS_COUNT", "IMOM_RM_00"], anisotropy_x=AX, anisotropy_y=AY)
    df = nyx.featurize_directory(str(intdir) + os.sep, str(segdir) + os.sep)

    assert df["AREA_PIXELS_COUNT"].iloc[0] == double
    want = float(inten[mask > 0].astype(float).sum()) * AX * AY
    assert df["IMOM_RM_00"].iloc[0] == pytest.approx(want, rel=1e-12)
