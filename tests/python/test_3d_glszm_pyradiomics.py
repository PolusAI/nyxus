"""3D GLSZM against PyRadiomics 3.0.1 on bench_volmin_3d, whose ROI holds the volume's lowest
intensity, and on its control bench_volmin_ctrl_3d, in RAM and out of core.

Nyxus stores a volume offset by its lowest intensity, so the darkest voxels of this ROI are stored as
0, the value the background cells of its bounding box hold. Under PyRadiomics-style binning they are
the ROI's bin 1 like any other voxel at its lowest level. The control moves the volume's minimum to a
voxel outside the ROI's bounding box, so the same ROI is stored without a 0. PyRadiomics reads the ROI's
own intensities and gives both phantoms the same values, so one table pins both. What this
discriminates: a family that bins a stored 0 to level 0 reads those voxels as background, or shifts
every level by one, and parts from the table on bench_volmin_3d while it still holds on the control.

The out-of-core side streams the same stored intensities from an OME-TIFF Z-stack at ram_limit=0; the
NIfTI loader delivers the whole volume in one read and so cannot stream (see test_data.featurize_volmin).

The logarithmic features (3GLSZM_ZE) carry Nyxus' fast_log10 approximation, a residual
of the order of 1e-3 that the family's other PyRadiomics tests band; they are left out here so
every pin holds at rel 1e-9.

Goldens: binCount=8, no resampling, weightingNorm=None, from tests/vetting/oracles/gen_glszm3d_pyradiomics.py,
which re-verifies GLSZM_3D_VOLMIN_PYRADIOMICS below on every run.

Kind: *pyradiomics* per tests/vetting/SPEC.md 4.
"""

import pytest

from test_data import VOLMIN_NBINS, assert_volmin_matches, featurize_volmin

# PyRadiomics 3.0.1 on bench_volmin_3d at 8 bins: the same values on bench_volmin_ctrl_3d, whose ROI is the same.
GLSZM_3D_VOLMIN_PYRADIOMICS = {
    "3GLSZM_GLN": 13.685714285714285,
    "3GLSZM_GLNN": 0.13034013605442177,
    "3GLSZM_GLV": 5.067210884353742,
    "3GLSZM_HGLZE": 23.19047619047619,
    "3GLSZM_LAE": 169.56190476190477,
    "3GLSZM_LAHGLE": 7802.114285714286,
    "3GLSZM_LALGLE": 11.393171836194796,
    "3GLSZM_LGLZE": 0.2178823021272001,
    "3GLSZM_SAE": 0.598210506718065,
    "3GLSZM_SAHGLE": 14.642090673477966,
    "3GLSZM_SALGLE": 0.1060690942220315,
    "3GLSZM_SZN": 35.095238095238095,
    "3GLSZM_SZNN": 0.33424036281179137,
    "3GLSZM_ZP": 0.1875,
    "3GLSZM_ZV": 141.1174603174603,
}


@pytest.mark.parametrize("side", ["in_ram", "out_of_core"])
@pytest.mark.parametrize("phantom", ["volmin", "ctrl"])
def test_3d_glszm_volmin_pyradiomics(tmp_path, phantom, side):
    assert VOLMIN_NBINS == 8
    row = featurize_volmin(["*3D_GLSZM*"], phantom, side, tmp_path)
    assert_volmin_matches(row, GLSZM_3D_VOLMIN_PYRADIOMICS, 1e-9, "3D GLSZM on %s, %s" % (phantom, side))
