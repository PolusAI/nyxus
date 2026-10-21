"""3D GLDZM against MIRP fixed_bin_number 8 on bench_volmin_3d, whose ROI holds the volume's lowest
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

Goldens: by_slice=False, base_discretisation_method="fixed_bin_number", n_bins=8, from
tests/vetting/oracles/gen_gldzm3d_mirp.py, which re-verifies GLDZM_3D_VOLMIN_MIRP below on every run.

Kind: *mirp* per tests/vetting/SPEC.md 4.
"""

import pytest

from test_data import VOLMIN_NBINS, assert_volmin_matches, featurize_volmin

# MIRP fixed_bin_number on bench_volmin_3d at 8 bins: the same values on bench_volmin_ctrl_3d, whose ROI is the same.
GLDZM_3D_VOLMIN_MIRP = {
    "3GLDZM_GLNU": 13.685714285714285,
    "3GLDZM_GLNUN": 0.13034013605442177,
    "3GLDZM_GLV": 5.067210884353742,
    "3GLDZM_HGLZE": 23.19047619047619,
    "3GLDZM_LDE": 1.561904761904762,
    "3GLDZM_LDHGLE": 31.485714285714284,
    "3GLDZM_LDLGLE": 0.3926128792786956,
    "3GLDZM_LGLZE": 0.2178823021272001,
    "3GLDZM_SDE": 0.9018518518518518,
    "3GLDZM_SDHGLE": 21.58227513227513,
    "3GLDZM_SDLGLE": 0.1875565067517307,
    "3GLDZM_ZDE": 3.5395774379847715,
    "3GLDZM_ZDNU": 81.53333333333333,
    "3GLDZM_ZDNUN": 0.7765079365079365,
    "3GLDZM_ZDV": 0.21188208616780047,
    "3GLDZM_ZP": 0.1875,
}


@pytest.mark.parametrize("side", ["in_ram", "out_of_core"])
@pytest.mark.parametrize("phantom", ["volmin", "ctrl"])
def test_3d_gldzm_volmin_mirp(tmp_path, phantom, side):
    assert VOLMIN_NBINS == 8
    row = featurize_volmin(["*3D_GLDZM*"], phantom, side, tmp_path)
    assert_volmin_matches(row, GLDZM_3D_VOLMIN_MIRP, 1e-9, "3D GLDZM on %s, %s" % (phantom, side))
