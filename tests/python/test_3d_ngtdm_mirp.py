"""3D NGTDM against MIRP 2.6.0 on a ROI that mixes voxels with a ROI neighbour and one without, through
both the in-RAM and the out-of-core path.

IBSI leaves a voxel with no neighbour out of the NGTDM -- it is in no row and counts towards neither
n_i, Nvp nor Ngp -- and MIRP and Nyxus both follow it, so this is held to MIRP. PyRadiomics keeps such
a voxel as a row with s_i = 0 and parts from both on any ROI holding one.

The volume is the ellipsoid of test_3d_ngtdm_pyradiomics.py, every voxel of which has a neighbour,
plus one voxel at the corner (0,0,0), outside the ellipsoid's reach, at level 12, which the ellipsoid
(levels 0..11) does not carry. Under IBSI that voxel changes nothing: these goldens equal the
ellipsoid's PyRadiomics ones to 1e-15. Its level is still one of the ROI's levels, so its matrix row is
empty, and Ngp -- the number of non-empty rows, which Contrast divides by -- is 12, not 13.

Goldens: tests/vetting/oracles/gen_ngtdm3d_mirp.py (make_ooc_mixed), which re-verifies
NGTDM_3D_MIXED_MIRP below against MIRP and against an independent numpy NGTDM on every run.
"""
import numpy as np
import pytest

from test_3d_ngtdm_common import featurize_3d_ngtdm

tifffile = pytest.importorskip("tifffile")

# The in-RAM side's ram_limit, as in test_ooc_mechanics.py.
RAM_LIMIT_LARGE_MB = 1000

# MIRP 2.6.0 NGTDM, by_slice=False, distance 1, fixed_bin_number n=13 (levels 0..12 -> 1..13), on
# _make_mixed_volume_pair's volume.
NGTDM_3D_MIXED_MIRP = {
    "3NGTDM_BUSYNESS": 190.1765607383964,
    "3NGTDM_COARSENESS": 9.511548544293426e-05,
    "3NGTDM_COMPLEXITY": 170.2761206366912,
    "3NGTDM_CONTRAST": 0.546191893227136,
    "3NGTDM_STRENGTH": 0.004876244827762919,
}


def _make_mixed_volume_pair(tmp_path):
    # 24x60x60, so the ROI's footprint is over 1 MB and ram_limit=1 forces the oversized path.
    Z, Y, X = 24, 60, 60
    z, y, x = np.indices((Z, Y, X))
    inten = ((3 * x + 5 * y + 7 * z + x * y * z) % 12).astype(np.uint16)
    mask = ((z - 11.5) / 11.5) ** 2 + ((y - 29.5) / 28.0) ** 2 + ((x - 29.5) / 28.0) ** 2 <= 1.0
    inten[0, 0, 0] = 12
    mask[0, 0, 0] = True
    intp = tmp_path / "mixed_int.ome.tif"
    segp = tmp_path / "mixed_seg.ome.tif"
    tifffile.imwrite(str(intp), inten, metadata={"axes": "ZYX"})
    tifffile.imwrite(str(segp), mask.astype(np.uint32), metadata={"axes": "ZYX"})
    return str(intp), str(segp)


@pytest.mark.parametrize("ram_limit", [RAM_LIMIT_LARGE_MB, 1], ids=["in_ram", "out_of_core"])
def test_3d_ngtdm_mixed_mirp(tmp_path, ram_limit):
    """The five 3D NGTDM features of the mixed volume, at ram_limit 1000 MB (in-RAM) and 1 MB
    (out-of-core), each against MIRP at rel=1e-9. The run's own log says which path it took."""
    intp, segp = _make_mixed_volume_pair(tmp_path)
    got, oversized = featurize_3d_ngtdm(intp, segp, ram_limit, 1)
    assert oversized == (ram_limit == 1), "ram_limit=%d took the %s path" % (
        ram_limit, "oversized" if oversized else "in-RAM")

    bad = [(f, got[f], want) for f, want in NGTDM_3D_MIXED_MIRP.items()
           if abs(got[f] - want) > 1e-9 * abs(want)]
    assert not bad, "3D NGTDM (ram_limit=%d) diverges from MIRP: %r" % (ram_limit, bad)
