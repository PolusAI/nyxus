"""3D NGLDM against MIRP fixed_bin_number 8 on bench_volmin_3d, whose ROI holds the volume's lowest
intensity, and on its control bench_volmin_ctrl_3d, in RAM and out of core.

Nyxus stores a volume offset by its lowest intensity, so the darkest voxels of this ROI are stored as
0, the value the background cells of its bounding box hold. Under PyRadiomics-style binning -- IBSI's
fixed bin number, which MIRP computes itself -- they are the ROI's bin 1 like any other voxel at its
lowest level. The control moves the volume's minimum to a voxel outside the ROI's bounding box, so the
same ROI is stored without a 0. MIRP reads the ROI's own intensities and gives both phantoms the same
values, so one table pins both. What this discriminates: a family that bins a stored 0 to level 0 and
then lifts every level by one, or drops those voxels, parts from the table on bench_volmin_3d while it
still holds on the control.

The out-of-core side streams the same stored intensities from an OME-TIFF Z-stack at ram_limit=0; the
NIfTI loader delivers the whole volume in one read and so cannot stream (see test_data.featurize_volmin).

3NGLDM_DCP is left out as in the family's other MIRP tables: both tools return 1 whenever every
voxel has a same-level neighbour, so it would claim more than it shows.

Goldens: by_slice=False, base_discretisation_method="fixed_bin_number", n_bins=8, from
tests/vetting/oracles/gen_ngldm3d_mirp.py, which re-verifies NGLDM_3D_VOLMIN_MIRP below on every run.

Kind: *mirp* per tests/vetting/SPEC.md 4.
"""

import pytest

from test_data import VOLMIN_NBINS, assert_volmin_matches, featurize_volmin

# MIRP fixed_bin_number on bench_volmin_3d at 8 bins: the same values on bench_volmin_ctrl_3d, whose ROI is the same.
NGLDM_3D_VOLMIN_MIRP = {
    "3NGLDM_DCENE": 0.02431122448979592,
    "3NGLDM_DCENT": 5.529960784380476,
    "3NGLDM_DCNU": 95.43214285714286,
    "3NGLDM_DCNUN": 0.17041454081632654,
    "3NGLDM_DCV": 3.170599489795918,
    "3NGLDM_GLNU": 73.86785714285715,
    "3NGLDM_GLNUN": 0.13190688775510204,
    "3NGLDM_GLV": 5.647755102040818,
    "3NGLDM_HDE": 16.053571428571427,
    "3NGLDM_HDHGLE": 571.8785714285714,
    "3NGLDM_HDLGLE": 1.8839352147108843,
    "3NGLDM_HGLCE": 28.414285714285715,
    "3NGLDM_LDE": 0.19885432544311835,
    "3NGLDM_LDHGLE": 4.976939775800456,
    "3NGLDM_LDLGLE": 0.03671006037664266,
    "3NGLDM_LGLCE": 0.16519536311548427,
}


@pytest.mark.parametrize("side", ["in_ram", "out_of_core"])
@pytest.mark.parametrize("phantom", ["volmin", "ctrl"])
def test_3d_ngldm_volmin_mirp(tmp_path, phantom, side):
    assert VOLMIN_NBINS == 8
    row = featurize_volmin(["*3D_NGLDM*"], phantom, side, tmp_path)
    assert_volmin_matches(row, NGLDM_3D_VOLMIN_MIRP, 1e-9, "3D NGLDM on %s, %s" % (phantom, side))
