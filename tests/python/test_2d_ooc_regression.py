"""Known-defect characterization for the 2D out-of-core contour path.

Kind: *regression* per tests/vetting/SPEC.md 2 -- these pin CURRENT behaviour, not correct
behaviour, and establish no vetting. The values below are what Nyxus returns today; a fix to the
defect they describe MUST change them, which is the point of pinning them.

The path-equality assertions live in test_2d_ooc_invariant.py. This file exists because two contour
features do NOT satisfy that equality, and a divergence a test suite cannot see is worse than one it
records.

PERIMETER is computed by two different definitions on the two paths (contour.cpp):

    calculate()        fval_PERIMETER = sum of sqrt(p1.sqdist(p2)) around the contour  (Euclidean)
    osized_calculate() fval_PERIMETER = (StatsInt) K.size()                            (pixel count)

DIAMETER_EQUAL_PERIMETER inherits it exactly. Both paths compute it as `fval_PERIMETER / M_PI`
(contour.cpp lines 976 and 1000), so the ratio between the two paths is identical for the two
features and the derived one cannot be correct wherever the base one is not. Pinning only PERIMETER
would leave a second public value diverging with nothing to say so.

CLAUDE.md states the requirement the two paths are under: "the in-RAM path and the out-of-core path
must produce identical values." They do not, so this is a defect rather than a documented
convention -- see tests/vetting/matrix/morphology.md and not_covered.md section G.

Why it went unnoticed: test_2d_ooc_invariant.py already asserts *ALL_MORPHOLOGY* agrees across the
two paths and passes. Its fixture is a full-image RECTANGLE, and around a rectangle every contour
step is an axis-aligned unit step, so the pixel count and the Euclidean sum are the same number. The
existing invariant is real, it simply cannot discriminate on that shape. A disk separates them.
"""
import numpy as np
import pytest

import nyxus

import test_data

tifffile = pytest.importorskip("tifffile")

RAM_LIMIT_LARGE_MB = 64

# Measured on bench_disk64_diagonal_boundary. Not a correctness claim on either side: the in-RAM
# value is the Euclidean contour length and the out-of-core value is the number of contour pixels
# (112 of them, which is the fixture's edge-pixel count in benchmarks.md).
PERIMETER_IN_RAM = 131.88225099390849
PERIMETER_OUT_OF_CORE = 112.0
# = the two above divided by pi, to the last digit -- the derived feature carries the whole gap.
DIAMETER_EQUAL_PERIMETER_IN_RAM = 41.97942430353313
DIAMETER_EQUAL_PERIMETER_OUT_OF_CORE = 35.65070725258456


def _disk_pair(tmp_path):
    """bench_disk64_diagonal_boundary, built by test_data.disk64_arrays() so the three modules that
    read this fixture cannot drift apart."""
    return test_data.write_disk64_pair(tmp_path, tifffile, as_dirs=True)


def _contour_values(intdir, segdir, ram_limit_mb):
    n = nyxus.Nyxus(["*ALL_MORPHOLOGY*"])
    n.set_params(ram_limit=ram_limit_mb)
    got = n.get_params("ram_limit")["ram_limit"]
    assert got == ram_limit_mb, (
        "ram_limit=%d MB was not accepted (still %d MB); Nyxus refuses a limit above available RAM"
        % (ram_limit_mb, got)
    )
    df = n.featurize_directory(intdir, segdir)
    return (float(df["PERIMETER"].iloc[0]), float(df["DIAMETER_EQUAL_PERIMETER"].iloc[0]))


def test_2d_ooc_perimeter_diverges_from_in_ram_regression(tmp_path):
    """PERIMETER does not survive the out-of-core round trip, and the gap is the whole definition.

    Pinned at rel=1e-9 on both sides because each is a deterministic function of the same contour --
    one a sum of unit and sqrt(2) steps, the other an integer count -- so any movement is a change of
    behaviour rather than float wobble. Fixing osized_calculate() to sum step lengths makes the two
    equal and fails this test, which is the intended signal: delete this file's assertions and fold
    PERIMETER and DIAMETER_EQUAL_PERIMETER back into test_2d_ooc_invariant.py.
    """
    intdir, segdir = _disk_pair(tmp_path)

    p_ram, _ = _contour_values(intdir, segdir, RAM_LIMIT_LARGE_MB)
    p_ooc, _ = _contour_values(intdir, segdir, 0)      # 0 forces the oversized branch for any ROI

    assert p_ram == pytest.approx(PERIMETER_IN_RAM, rel=1e-9)
    assert p_ooc == pytest.approx(PERIMETER_OUT_OF_CORE, rel=1e-9)

    # The characterization proper: the two paths disagree, by ~15% here. Asserting the inequality
    # rather than only the two numbers means a fix cannot half-land -- a change that moves one side
    # toward the other without meeting it still fails.
    assert p_ram != pytest.approx(p_ooc, rel=1e-6), (
        "in-RAM and out-of-core PERIMETER now agree (%r vs %r). If osized_calculate() was changed to "
        "sum Euclidean step lengths, this file has served its purpose: remove it and add PERIMETER "
        "back to test_2d_ooc_invariant.py." % (p_ram, p_ooc)
    )

    # the out-of-core value is exactly the contour pixel count, i.e. an integer, which is what
    # identifies the defect as the definition rather than an accumulation error
    assert p_ooc == float(int(p_ooc))


def test_2d_ooc_diameter_equal_perimeter_inherits_the_divergence_regression(tmp_path):
    """DIAMETER_EQUAL_PERIMETER carries the same gap, because it is PERIMETER/pi on both paths.

    Asserted as the identity rather than only as two literals: what makes this feature's divergence
    a consequence rather than a second defect is that the ratio between the paths is the SAME for
    both features. A fix to PERIMETER alone must therefore fix this one, and if the two ratios ever
    stop matching that is a new finding, not this one.
    """
    intdir, segdir = _disk_pair(tmp_path)

    p_ram, d_ram = _contour_values(intdir, segdir, RAM_LIMIT_LARGE_MB)
    p_ooc, d_ooc = _contour_values(intdir, segdir, 0)

    assert d_ram == pytest.approx(DIAMETER_EQUAL_PERIMETER_IN_RAM, rel=1e-9)
    assert d_ooc == pytest.approx(DIAMETER_EQUAL_PERIMETER_OUT_OF_CORE, rel=1e-9)
    assert d_ram != pytest.approx(d_ooc, rel=1e-6)

    # derived exactly, on both paths
    assert d_ram == pytest.approx(p_ram / np.pi, rel=1e-12)
    assert d_ooc == pytest.approx(p_ooc / np.pi, rel=1e-12)
    # and therefore the same ratio between the paths as PERIMETER has
    assert (d_ram / d_ooc) == pytest.approx(p_ram / p_ooc, rel=1e-12)
