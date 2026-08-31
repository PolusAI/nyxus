"""Known-defect characterization for the 2D out-of-core contour path.

Kind: *regression* per tests/vetting/SPEC.md 2 -- these pin CURRENT behaviour, not correct
behaviour, and establish no vetting. The values below are what Nyxus returns today; a fix to the
defect they describe MUST change them, which is the point of pinning them.

The path-equality assertions live in test_2d_ooc_invariant.py. This file exists because one
contour feature does NOT satisfy that equality, and a divergence a test suite cannot see is worse
than one it records.

PERIMETER is computed by two different definitions on the two paths (contour.cpp):

    calculate()        fval_PERIMETER = sum of sqrt(p1.sqdist(p2)) around the contour  (Euclidean)
    osized_calculate() fval_PERIMETER = (StatsInt) K.size()                            (pixel count)

CLAUDE.md states the requirement the two paths are under: "the in-RAM path and the out-of-core path
must produce identical values." They do not, so this is a defect rather than a documented
convention -- see tests/vetting/matrix/morphology.md and not_covered.md.

Why it went unnoticed: test_2d_ooc_invariant.py already asserts *ALL_MORPHOLOGY* agrees across the
two paths and passes. Its fixture is a full-image RECTANGLE, and around a rectangle every contour
step is an axis-aligned unit step, so the pixel count and the Euclidean sum are the same number. The
existing invariant is real, it simply cannot discriminate on that shape. A disk separates them.
"""
import os

import numpy as np
import pytest

import nyxus

tifffile = pytest.importorskip("tifffile")

RAM_LIMIT_LARGE_MB = 64

# Measured on the disk fixture below. Not a correctness claim on either side: the in-RAM value is the
# Euclidean contour length and the out-of-core value is the number of contour pixels (112 of them).
PERIMETER_IN_RAM = 131.88225099390849
PERIMETER_OUT_OF_CORE = 112.0


def _disk_pair(tmp_path):
    """Same 64x64 single-disk fixture as test_2d_ooc_invariant.py's diagonal-boundary test."""
    Y = X = 64
    yy, xx = np.mgrid[0:Y, 0:X]
    mask = (((yy - 32) ** 2 + (xx - 32) ** 2) <= 20 * 20).astype(np.uint32)
    inten = ((1 + xx + yy * 7) * mask).astype(np.uint32)
    intdir = tmp_path / "int"
    segdir = tmp_path / "seg"
    intdir.mkdir()
    segdir.mkdir()
    tifffile.imwrite(str(intdir / "img.tif"), inten)
    tifffile.imwrite(str(segdir / "img.tif"), mask)
    return str(intdir) + os.sep, str(segdir) + os.sep


def _perimeter(intdir, segdir, ram_limit_mb):
    n = nyxus.Nyxus(["*ALL_MORPHOLOGY*"])
    n.set_params(ram_limit=ram_limit_mb)
    got = n.get_params("ram_limit")["ram_limit"]
    assert got == ram_limit_mb, (
        "ram_limit=%d MB was not accepted (still %d MB); Nyxus refuses a limit above available RAM"
        % (ram_limit_mb, got)
    )
    return float(n.featurize_directory(intdir, segdir)["PERIMETER"].iloc[0])


def test_2d_ooc_perimeter_diverges_from_in_ram_regression(tmp_path):
    """PERIMETER does not survive the out-of-core round trip, and the gap is the whole definition.

    Pinned at rel=1e-9 on both sides because each is a deterministic function of the same contour --
    one a sum of unit and sqrt(2) steps, the other an integer count -- so any movement is a change of
    behaviour rather than float wobble. Fixing osized_calculate() to sum step lengths makes the two
    equal and fails this test, which is the intended signal: delete this file's assertion and fold
    PERIMETER back into OOC_AGREEING_CONTOUR_FEATURES in test_2d_ooc_invariant.py.
    """
    intdir, segdir = _disk_pair(tmp_path)

    p_ram = _perimeter(intdir, segdir, RAM_LIMIT_LARGE_MB)
    p_ooc = _perimeter(intdir, segdir, 0)          # 0 forces the oversized branch for any ROI

    assert p_ram == pytest.approx(PERIMETER_IN_RAM, rel=1e-9)
    assert p_ooc == pytest.approx(PERIMETER_OUT_OF_CORE, rel=1e-9)

    # The characterization proper: the two paths disagree, by ~15% here. Asserting the inequality
    # rather than only the two numbers means a fix cannot half-land -- a change that moves one side
    # toward the other without meeting it still fails.
    assert p_ram != pytest.approx(p_ooc, rel=1e-6), (
        "in-RAM and out-of-core PERIMETER now agree (%r vs %r). If osized_calculate() was fixed to "
        "sum Euclidean step lengths, this file has served its purpose: remove it and add PERIMETER "
        "to OOC_AGREEING_CONTOUR_FEATURES in test_2d_ooc_invariant.py." % (p_ram, p_ooc)
    )

    # and the out-of-core value is exactly the contour pixel count, i.e. an integer, which is what
    # identifies the defect as the definition rather than an accumulation error
    assert p_ooc == float(int(p_ooc))
