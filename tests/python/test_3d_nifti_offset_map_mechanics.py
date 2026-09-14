"""Mechanics tests for the NIfTI load-time maps: fractional rescale, and non-finite voxels.

An integer NIfTI with a header rescale reaches the features in its own domain by one of two maps.
Without ``--preserve-hu`` it takes the stored map: the stored integers are its grey levels and
``scl_slope`` / ``scl_inter`` go into the recorded inverse, so nothing is narrowed. Under the flag it
takes the offset map: each voxel is rescaled to physical units (``scl_slope * stored + scl_inter``),
shifted by the offset the scan recorded, and narrowed to 1 grey level == 1 intensity unit. Two
properties are pinned here, neither of which the CT fixtures can show:

*What a fractional value reads back as.* ``inten_scale`` is 1 on the offset map, so the inverse cannot
recover a fraction the narrowing drops -- there it rounds to nearest. The stored map drops nothing and
reports the fraction exactly. ``ct3d_int16.nii`` cannot show the difference: its ``scl_slope`` is 2,
so every rescaled value is integral. ``ct3d_frac.nii`` uses 0.5, and the two cases below pin both maps.

*Non-finite voxels.* A float volume is free to hold NaN -- a masked-out background, most often -- and
converting one to an unsigned grey level is undefined. The loader stores grey level 0 for it, the
convention every load-time map uses. ``ct3d_nan.nii`` is float32, which has no stored integers to
keep, so it takes the offset map with or without the flag.

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
    Truncating would give -769 and -896.5; the stored map, below, gives -768.5 and -896.25.
    """
    f = _featurize(FRAC, preserve_hu=True)

    assert f["3MIN"] == pytest.approx(-1024.0)
    assert f["3MAX"] == pytest.approx(-768.0), "truncating under --preserve-hu would report -769"
    assert f["3MEAN"] == pytest.approx(-896.0), "truncating under --preserve-hu would report -896.5"


def test_3d_nifti_stored_map_fractional_slope_is_exact_by_default_mechanics():
    """Without the flag the stored map keeps every stored level, so the fractions come back exactly.

    The grey levels are idx itself and the inverse is -1024 + 0.5*u: MAX is 0.5*511 - 1024 = -768.5
    and MEAN is 0.5*255.5 - 1024 = -896.25. Rescaling before narrowing would lose the halves -- to
    -769 / -896.5 by truncation, or -768 / -896 by rounding -- and a slope well below 1 would lose
    far more than halves.
    """
    f = _featurize(FRAC, preserve_hu=False)

    assert f["3MIN"] == pytest.approx(-1024.0)
    assert f["3MAX"] == pytest.approx(-768.5), "a narrowed map would report -769 or -768"
    assert f["3MEAN"] == pytest.approx(-896.25), "a narrowed map would report -896.5 or -896"


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
