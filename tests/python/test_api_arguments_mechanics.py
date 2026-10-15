"""Mechanics: the Python constructors refuse a coarse_gray_depth that is not positive, and say so.

Kind: *mechanics* per tests/vetting/SPEC.md 2 -- this asserts argument checking in the Python
wrappers, not a feature value.

Nyxus, Nyxus3D and ImageQuality each refuse coarse_gray_depth <= 0 with a ValueError. The
message names the rule the check applies: 0 is refused, so the value must be positive. The
command line accepts 0 and negative values (no binning, PyRadiomics-style binning); the Python
constructors do not reach those modes.
"""
import pytest

import nyxus

CONSTRUCTORS = [
    pytest.param(lambda d: nyxus.Nyxus(["MEAN"], coarse_gray_depth=d), id="Nyxus"),
    pytest.param(lambda d: nyxus.Nyxus3D(["3MEAN"], coarse_gray_depth=d), id="Nyxus3D"),
    pytest.param(lambda d: nyxus.ImageQuality(["FOCUS_SCORE"], coarse_gray_depth=d), id="ImageQuality"),
]


@pytest.mark.parametrize("make", CONSTRUCTORS)
@pytest.mark.parametrize("depth", [0, -8])
def test_coarse_gray_depth_not_positive_refused_mechanics(make, depth):
    with pytest.raises(ValueError, match="coarse_gray_depth.*must be positive"):
        make(depth)


@pytest.mark.parametrize("make", CONSTRUCTORS)
def test_coarse_gray_depth_positive_accepted_mechanics(make):
    make(8)
