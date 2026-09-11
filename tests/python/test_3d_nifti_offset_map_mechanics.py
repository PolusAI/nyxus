"""Mechanics tests for the NIfTI load-time offset map: fractional rescale, and non-finite voxels.

The NIfTI loader rescales each voxel to physical units (``scl_slope * stored + scl_inter``) and then
shifts it by the offset the scan recorded, so the volume reaches the features in its own domain. Two
properties of that narrowing are pinned here, neither of which the CT fixtures can show:

*Which way it narrows, and when.* ``--preserve-hu`` exists to carry absolute intensities, and
``inten_scale`` is 1 on the offset branch, so the inverse map cannot recover a fraction a truncating
cast drops -- there the narrowing rounds to nearest. Without the flag it truncates, which is what a
real-valued NIfTI has always done and what the 3D texture goldens encode. ``ct3d_int16.nii`` cannot
show the difference: its ``scl_slope`` is 2, so every rescaled value is integral and the two agree.
``ct3d_frac.nii`` uses 0.5, where they do not, and the two cases below pin both directions.

*Non-finite voxels.* A float volume is free to hold NaN -- a masked-out background, most often -- and
converting one to an unsigned grey level is undefined. The loader stores grey level 0 for it, the
convention every load-time map uses.

Kind: *mechanics* per tests/vetting/SPEC.md 2 -- these pin loader plumbing rather than feature values
against a reference, and establish no vetting.

A caveat on what the non-finite case can and cannot prove. The undefined conversion it guards against
is undefined, not wrong: on x86-64 ``cvttsd2si`` happens to yield 0 for a NaN, which is the same grey
level the guard stores deliberately. So those assertions pin the intended values and would catch a
platform that behaves differently, but the assertion that actually discriminates is the UBSan run
over this fixture. The rounding cases have no such caveat -- they fail on arithmetic alone.
"""
import math
import os
import pathlib

import pytest
import nyxus

DATA = pathlib.Path(__file__).resolve().parent.parent / "data" / "hounsfield"
FRAC = str(DATA / "ct3d_frac.nii")
NONFINITE = str(DATA / "ct3d_nan.nii")
MASK = str(DATA / "mask3d.nii")

pytestmark = pytest.mark.skipif(
    not (os.path.exists(FRAC) and os.path.exists(NONFINITE) and os.path.exists(MASK)),
    reason="NIfTI offset-map fixtures not present in tests/data/hounsfield",
)

FEATS = ["3MIN", "3MAX", "3MEAN"]


def _featurize(intensity, preserve_hu=False):
    nyx = nyxus.Nyxus3D(FEATS, preserve_hu=preserve_hu)
    df = nyx.featurize_files([intensity], [MASK], False)  # explicit whole-volume mask
    return {c: float(df[c].iloc[0]) for c in FEATS}


def test_3d_nifti_offset_map_fractional_slope_rounds_under_preserve_hu_mechanics():
    """With the flag, a fractional scl_slope rounds to nearest.

    ct3d_frac.nii stores idx (0..511) with scl_slope=0.5, scl_inter=-1024, so the physical value is
    0.5*idx - 1024 over [-1024.0, -768.5] and every odd voxel is a half. The offset is
    floor(min) = -1024, so the stored grey level is round(0.5*idx): MAX -768, MEAN -896 exactly.
    Truncating would give -769 and -896.5, which is what the case below asserts.
    """
    f = _featurize(FRAC, preserve_hu=True)

    assert f["3MIN"] == pytest.approx(-1024.0)
    assert f["3MAX"] == pytest.approx(-768.0), "truncating under --preserve-hu would report -769"
    assert f["3MEAN"] == pytest.approx(-896.0), "truncating under --preserve-hu would report -896.5"


def test_3d_nifti_offset_map_fractional_slope_truncates_by_default_mechanics():
    """Without the flag it truncates, which is the pre-existing behaviour the 3D goldens encode.

    Rounding a real-valued volume read without --preserve-hu would move every 3D texture golden in
    the tree, since the phantoms are float32 and every voxel is fractional. That is a decision about
    quantization, not a defect to fix in passing, so the default is left as it has always been.
    """
    f = _featurize(FRAC, preserve_hu=False)

    assert f["3MIN"] == pytest.approx(-1024.0)
    assert f["3MAX"] == pytest.approx(-769.0), "rounding by default would report -768"
    assert f["3MEAN"] == pytest.approx(-896.5), "rounding by default would report -896"


@pytest.mark.parametrize("preserve_hu", [False, True])
def test_3d_nifti_offset_map_nonfinite_voxels_mechanics(preserve_hu):
    """NaN and either infinity take grey level 0, whichever way the map narrows.

    ct3d_nan.nii is float32 v(idx) = idx with a NaN at 3, +Inf at 5 and -Inf at 7, and no header
    rescale. The finite range is [0, 511] and the offset is 0 -- the non-finite voxels are left out
    of the scanned extrema -- so the map is the identity and what is reported is what was stored.
    Every value here is integral, so the narrowing does not enter into it.
    """
    f = _featurize(NONFINITE, preserve_hu=preserve_hu)

    for k, v in f.items():
        assert math.isfinite(v), "%s is not finite: %r" % (k, v)

    # the three non-finite voxels neither drag the extrema nor wrap
    assert f["3MIN"] == pytest.approx(0.0)
    assert f["3MAX"] == pytest.approx(511.0)

    # sum(0..511) minus the three voxels that became 0, over 512
    assert f["3MEAN"] == pytest.approx((130816 - 3 - 5 - 7) / 512.0)
