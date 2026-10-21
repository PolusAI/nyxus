"""3D GLDM against PyRadiomics 3.0.1 on bench_volmin_3d, whose ROI holds the volume's lowest
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

The logarithmic features (3GLDM_DE) carry Nyxus' fast_log10 approximation, a residual
of the order of 1e-3 that the family's other PyRadiomics tests band; they are left out here so
every pin holds at rel 1e-9.

Goldens: binCount=8, no resampling, weightingNorm=None, from tests/vetting/oracles/gen_gldm3d_pyradiomics.py,
which re-verifies GLDM_3D_VOLMIN_PYRADIOMICS below on every run.

Kind: *pyradiomics* per tests/vetting/SPEC.md 4.
"""

import pytest

from test_data import VOLMIN_NBINS, assert_volmin_matches, featurize_volmin

# PyRadiomics 3.0.1 on bench_volmin_3d at 8 bins: the same values on bench_volmin_ctrl_3d, whose ROI is the same.
GLDM_3D_VOLMIN_PYRADIOMICS = {
    "3GLDM_DN": 95.43214285714286,
    "3GLDM_DNN": 0.17041454081632654,
    "3GLDM_DV": 3.170599489795918,
    "3GLDM_GLN": 73.86785714285715,
    "3GLDM_GLV": 5.647755102040816,
    "3GLDM_HGLE": 28.414285714285715,
    "3GLDM_LDE": 16.053571428571427,
    "3GLDM_LDHGLE": 571.8785714285714,
    "3GLDM_LDLGLE": 1.8839352147108843,
    "3GLDM_LGLE": 0.16519536311548427,
    "3GLDM_SDE": 0.19885432544311835,
    "3GLDM_SDHGLE": 4.976939775800455,
    "3GLDM_SDLGLE": 0.03671006037664267,
}


@pytest.mark.parametrize("side", ["in_ram", "out_of_core"])
@pytest.mark.parametrize("phantom", ["volmin", "ctrl"])
def test_3d_gldm_volmin_pyradiomics(tmp_path, phantom, side):
    assert VOLMIN_NBINS == 8
    row = featurize_volmin(["*3D_GLDM*"], phantom, side, tmp_path)
    assert_volmin_matches(row, GLDM_3D_VOLMIN_PYRADIOMICS, 1e-9, "3D GLDM on %s, %s" % (phantom, side))
