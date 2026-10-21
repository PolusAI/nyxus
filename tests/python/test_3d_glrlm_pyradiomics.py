"""3D GLRLM against PyRadiomics 3.0.1 on bench_volmin_3d, whose ROI holds the volume's lowest
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

The logarithmic features (3GLRLM_RE_AVE) carry Nyxus' fast_log10 approximation, a residual
of the order of 1e-3 that the family's other PyRadiomics tests band; they are left out here so
every pin holds at rel 1e-9.

Goldens: binCount=8, no resampling, weightingNorm=None, from tests/vetting/oracles/gen_glrlm3d_pyradiomics.py,
which re-verifies GLRLM_3D_VOLMIN_PYRADIOMICS below on every run.

Kind: *pyradiomics* per tests/vetting/SPEC.md 4.
"""

import pytest

from test_data import VOLMIN_NBINS, assert_volmin_matches, featurize_volmin

# PyRadiomics 3.0.1 on bench_volmin_3d at 8 bins: the same values on bench_volmin_ctrl_3d, whose ROI is the same.
GLRLM_3D_VOLMIN_PYRADIOMICS = {
    "3GLRLM_GLN_AVE": 65.60159853259819,
    "3GLRLM_GLNN_AVE": 0.13010124589226885,
    "3GLRLM_GLV_AVE": 5.583584348280135,
    "3GLRLM_HGLRE_AVE": 27.916001064711843,
    "3GLRLM_LGLRE_AVE": 0.16877635951279563,
    "3GLRLM_LRE_AVE": 1.4286252340245666,
    "3GLRLM_LRHGLE_AVE": 43.39820759601726,
    "3GLRLM_LRLGLE_AVE": 0.2199225750395456,
    "3GLRLM_RLN_AVE": 418.1133441240835,
    "3GLRLM_RLNN_AVE": 0.8288371363404127,
    "3GLRLM_RP_AVE": 0.9004120879120878,
    "3GLRLM_RV_AVE": 0.194291794390713,
    "3GLRLM_SRE_AVE": 0.927980602428168,
    "3GLRLM_SRHGLE_AVE": 25.652855539715063,
    "3GLRLM_SRLGLE_AVE": 0.1586678736288215,
}


@pytest.mark.parametrize("side", ["in_ram", "out_of_core"])
@pytest.mark.parametrize("phantom", ["volmin", "ctrl"])
def test_3d_glrlm_volmin_pyradiomics(tmp_path, phantom, side):
    assert VOLMIN_NBINS == 8
    row = featurize_volmin(["*3D_GLRLM*"], phantom, side, tmp_path)
    assert_volmin_matches(row, GLRLM_3D_VOLMIN_PYRADIOMICS, 1e-9, "3D GLRLM on %s, %s" % (phantom, side))
