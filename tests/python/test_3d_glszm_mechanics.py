"""Mechanics of the 3D GLSZM family through the Python API: which settings each workflow hands it.

Kind: *mechanics* per tests/vetting/SPEC.md 2 -- two workflows over one volume are compared with each
other, and nothing is asserted against a reference.
"""
import os
import pathlib
import shutil

import pytest
import nyxus

DATA = pathlib.Path(__file__).resolve().parent.parent / "data" / "hounsfield"
INTENSITY = DATA / "ct3d_int16.nii"
WHOLE_MASK = DATA / "mask3d.nii"     # label 1 in every voxel of the 8x8x8 volume

pytestmark = pytest.mark.skipif(
    not (INTENSITY.exists() and WHOLE_MASK.exists()),
    reason="ct3d_int16.nii / mask3d.nii not present in tests/data/hounsfield",
)


def _glszm(df):
    return {c: float(df[c].iloc[0]) for c in df.columns if c.startswith("3GLSZM_")}


# The whole-volume workflow and the segmented one hand 3D GLSZM the same settings vector, so the
# family's own grey depth reaches it on both. A segmentation covering every voxel is the same ROI as
# the whole volume, which makes the two runs comparable value for value. -8 PyRadiomics bins over the
# volume's 512 distinct levels moves every feature, so a workflow that read another family's vector
# would differ on all sixteen; the default run is the control that the two agree anyway.
@pytest.mark.parametrize("metaparam", [None, "3glszm/greydepth=-8"])
def test_3d_glszm_whole_volume_settings_match_segmented_mechanics(tmp_path, metaparam):
    shutil.copy(INTENSITY, tmp_path)

    def nyx():
        n = nyxus.Nyxus3D(["*3D_GLSZM*"])
        if metaparam:
            assert n.set_metaparam(metaparam) is None
        return n

    segmented = nyx().featurize_files([str(INTENSITY)], [str(WHOLE_MASK)], False)
    whole = nyx().featurize_directory(str(tmp_path), None, ".*")

    seg, wv = _glszm(segmented), _glszm(whole)
    assert len(seg) == 16 and seg.keys() == wv.keys()
    for c in seg:
        assert wv[c] == seg[c], c
