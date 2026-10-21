"""3D GLCM against PyRadiomics 3.0.1 on bench_volmin_3d, whose ROI holds the volume's lowest
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

The logarithmic features (3GLCM_DIFENTRO_AVE, 3GLCM_INFOMEAS1_AVE, 3GLCM_INFOMEAS2_AVE, 3GLCM_JE_AVE, 3GLCM_SUMENTROPY_AVE) carry Nyxus' fast_log10 approximation, a residual
of the order of 1e-3 that the family's other PyRadiomics tests band; they are left out here so
every pin holds at rel 1e-9.

Goldens: binCount=8, no resampling, weightingNorm=None, from tests/vetting/oracles/gen_glcm3d_pyradiomics.py,
which re-verifies GLCM_3D_VOLMIN_PYRADIOMICS below on every run.

Kind: *pyradiomics* per tests/vetting/SPEC.md 4.
"""

import pytest

from test_data import VOLMIN_NBINS, assert_volmin_matches, featurize_volmin

# PyRadiomics 3.0.1 on bench_volmin_3d at 8 bins: the same values on bench_volmin_ctrl_3d, whose ROI is the same.
GLCM_3D_VOLMIN_PYRADIOMICS = {
    "3GLCM_ACOR_AVE": 22.881028955947695,
    "3GLCM_ASM_AVE": 0.018426882097042536,
    "3GLCM_CLUPROM_AVE": 296.1317852575256,
    "3GLCM_CLUSHADE_AVE": -2.6224476716790366,
    "3GLCM_CLUTEND_AVE": 11.26684347438195,
    "3GLCM_CONTRAST_AVE": 11.30257334990674,
    "3GLCM_CORRELATION_AVE": -0.0015127185740918584,
    "3GLCM_DIFAVE_AVE": 2.719947376686564,
    "3GLCM_DIFVAR_AVE": 3.894859125384144,
    "3GLCM_ID_AVE": 0.38579892103265034,
    "3GLCM_IDM_AVE": 0.2967164396584654,
    "3GLCM_IDMN_AVE": 0.8718309682097499,
    "3GLCM_IDN_AVE": 0.7712322067003375,
    "3GLCM_IV_AVE": 0.2839973830578059,
    "3GLCM_JAVE_AVE": 4.784279883150361,
    "3GLCM_JMAX_AVE": 0.040581520437353,
    "3GLCM_JVAR_AVE": 5.642354206072174,
    "3GLCM_SUMAVERAGE_AVE": 9.56855976630072,
}


@pytest.mark.parametrize("side", ["in_ram", "out_of_core"])
@pytest.mark.parametrize("phantom", ["volmin", "ctrl"])
def test_3d_glcm_volmin_pyradiomics(tmp_path, phantom, side):
    assert VOLMIN_NBINS == 8
    row = featurize_volmin(["*3D_GLCM*"], phantom, side, tmp_path)
    assert_volmin_matches(row, GLCM_3D_VOLMIN_PYRADIOMICS, 1e-9, "3D GLCM on %s, %s" % (phantom, side))
