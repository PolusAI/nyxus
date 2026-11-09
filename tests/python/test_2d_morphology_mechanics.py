"""The circle diameters requested by name, without the features they are computed from.

Kind: *mechanics* per tests/vetting/SPEC.md 2 -- this asserts dispatch, not a value against an
oracle. `DIAMETER_MIN_ENCLOSING_CIRCLE`, `DIAMETER_INSCRIBING_CIRCLE` and
`DIAMETER_CIRCUMSCRIBING_CIRCLE` are measured to the ROI's contour, the last two from its centroid.
The trivial-ROI reduction runs the contour and the basic morphology passes only when
`ContourFeature::required()` and `BasicMorphologyFeatures::required()` hold, so both list the
diameters: a diameter requested on its own gets the same value as with every morphology feature.
"""
import os

import numpy as np
import pytest

import nyxus

tifffile = pytest.importorskip("tifffile")

DIAMETERS = ["DIAMETER_MIN_ENCLOSING_CIRCLE", "DIAMETER_INSCRIBING_CIRCLE", "DIAMETER_CIRCUMSCRIBING_CIRCLE"]


def _disk_pair(tmp_path):
    # a disk of radius 30 away from the image origin, so a centroid left at 0 shows
    yy, xx = np.mgrid[:100, :100]
    mask = ((yy - 50) ** 2 + (xx - 50) ** 2 <= 30 * 30).astype(np.uint32)
    ip, sp = os.path.join(tmp_path, "int.tif"), os.path.join(tmp_path, "seg.tif")
    tifffile.imwrite(ip, mask * 100)
    tifffile.imwrite(sp, mask)
    return ip, sp


@pytest.mark.parametrize("ram_limit", [None, 0], ids=["in_ram", "out_of_core"])
def test_2d_morphology_circle_diameters_by_name_mechanics(tmp_path, ram_limit):
    ip, sp = _disk_pair(tmp_path)
    kw = {} if ram_limit is None else {"ram_limit": ram_limit}

    every = nyxus.Nyxus(["*ALL_MORPHOLOGY*"], **kw).featurize_files([ip], [sp], False)
    # the circumscribing circle of a disk of radius 30 measured from its centre, so the reference
    # itself is right before the by-name requests are compared with it
    if ram_limit is None:
        assert float(every["DIAMETER_CIRCUMSCRIBING_CIRCLE"].iloc[0]) == pytest.approx(60.0, rel=1e-12)

    for code in DIAMETERS:
        alone = nyxus.Nyxus([code], **kw).featurize_files([ip], [sp], False)
        assert float(alone[code].iloc[0]) == pytest.approx(float(every[code].iloc[0]), rel=1e-12), code
        assert float(alone[code].iloc[0]) > 0.0, code
