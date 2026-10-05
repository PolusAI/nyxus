"""3D NGTDM against PyRadiomics 3.0.1 on a ROI that does not fill its bounding box, through both the
in-RAM and the out-of-core path.

The gtest oracle assertions (tests/test_3d_ngtdm_pyradiomics.h) cover calculate() on small NIfTI
phantoms. The out-of-core osized_calculate() is reached only by a ROI whose footprint passes
ram_limit, which takes a volume too large to commit as a fixture, so it is built here instead.

test_ooc_mechanics.py::test_ooc_3d_ngtdm_matches_in_ram_mechanics holds the two paths to each other on
a whole-volume mask, which has no background in its bounding box; two paths that both took background
into the neighbourhood would agree there. This holds each of them to the oracle instead.

Goldens: tests/vetting/oracles/gen_ngtdm3d_pyradiomics.py (make_ooc_ellipsoid), which re-verifies
NGTDM_3D_NONBOX_PYRADIOMICS below on every run.
"""
import numpy as np
import pytest

import nyxus

tifffile = pytest.importorskip("tifffile")

# The in-RAM side's ram_limit, as in test_ooc_mechanics.py: large enough for this volume, small enough
# that a CI runner accepts it (a limit above free RAM is rejected and leaves the previous one in place).
RAM_LIMIT_LARGE_MB = 1000

# PyRadiomics 3.0.1 NGTDM, binWidth=1, distances=[1], on _make_nonbox_volume_pair's ellipsoid.
NGTDM_3D_NONBOX_PYRADIOMICS = {
    "3NGTDM_BUSYNESS": 190.1765607383966,
    "3NGTDM_COARSENESS": 9.511548544293416e-05,
    "3NGTDM_COMPLEXITY": 170.27612063669142,
    "3NGTDM_CONTRAST": 0.5461918932271368,
    "3NGTDM_STRENGTH": 0.004876244827762912,
}


def _make_nonbox_volume_pair(tmp_path):
    # An ellipsoid in a 24x60x60 volume: 37,792 ROI voxels in an 86,400-voxel bounding box, so more
    # than half the box is background, and a footprint over 1 MB, so ram_limit=1 forces the oversized
    # path. Levels 0..11 cover the ROI too, so a ROI voxel at 0 and a background cell -- equal in the
    # binned planes -- are both in reach of most surface voxels. Every ROI voxel has a neighbour.
    Z, Y, X = 24, 60, 60
    z, y, x = np.indices((Z, Y, X))
    inten = ((3 * x + 5 * y + 7 * z + x * y * z) % 12).astype(np.uint16)
    mask = ((z - 11.5) / 11.5) ** 2 + ((y - 29.5) / 28.0) ** 2 + ((x - 29.5) / 28.0) ** 2 <= 1.0
    intp = tmp_path / "nonbox_int.ome.tif"
    segp = tmp_path / "nonbox_seg.ome.tif"
    tifffile.imwrite(str(intp), inten, metadata={"axes": "ZYX"})
    tifffile.imwrite(str(segp), mask.astype(np.uint32), metadata={"axes": "ZYX"})
    return str(intp), str(segp)


@pytest.mark.parametrize("ram_limit", [RAM_LIMIT_LARGE_MB, 1], ids=["in_ram", "out_of_core"])
def test_3d_ngtdm_nonbox_pyradiomics(tmp_path, ram_limit):
    """The five 3D NGTDM features of the ellipsoid, at ram_limit 1000 MB (in-RAM) and 1 MB
    (out-of-core), each against PyRadiomics at rel=1e-9.

    Raw levels (3ngtdm/greydepth=0) with Nyxus' zero-min correction match binWidth=1 on a ROI whose
    lowest level is 0, so no binning convention separates the two.
    """
    intp, segp = _make_nonbox_volume_pair(tmp_path)
    nyx = nyxus.Nyxus3D(["*3D_NGTDM*"], ram_limit=ram_limit)
    nyx.set_metaparam("3ngtdm/greydepth=0")
    nyx.set_metaparam("3ngtdm/radius=1")
    df = nyx.featurize_files([intp], [segp], False)
    assert len(df) == 1

    row = df.iloc[0]
    bad = [
        (f, float(row[f]), want)
        for f, want in NGTDM_3D_NONBOX_PYRADIOMICS.items()
        if abs(float(row[f]) - want) > 1e-9 * abs(want)
    ]
    assert not bad, "3D NGTDM (ram_limit=%d) diverges from PyRadiomics: %r" % (ram_limit, bad)
