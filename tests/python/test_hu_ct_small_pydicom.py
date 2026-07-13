"""Oracle test: Nyxus `preserve_hu` HU feature values vs an INDEPENDENT pydicom reference.

Unlike the synthetic Hounsfield fixtures (whose expected values are recomputed from the
same offset formula the loader uses — self-consistency, SPEC.md §5.2), this test vets the
actual first-order HU features against numbers derived by pydicom from a REAL scanner slice:

  fixture : ct_small_hu.tif — the true-HU int16 field of pydicom's CT_small.dcm
            (128x128, HU range -896..1167), written by make_ct_small_hu_fixture.py.
  oracle  : pydicom 3.0.2 — HU = RescaleSlope*stored + RescaleIntercept, then numpy stats.

Nyxus preserve_hu emits the OFFSET domain u = HU - floor(HU_min); floor(HU_min) == HU_MIN
because HU is integer-valued. Reconstructing absolute HU = offset + HU_MIN must equal the
pinned pydicom goldens. This is the first `vetted-by-oracle` assertion on HU *feature*
outputs (docs/vetting SPEC.md §3.1); the other HU tests are analytic / mechanics / invariant.

Provenance (record at the golden site, SPEC.md §6.4):
  tool    : pydicom 3.0.2, get_testdata_file('CT_small.dcm')
  config  : RescaleSlope=1.0, RescaleIntercept=-1024.0; whole-image all-ones ROI
  gen     : tests/data/hounsfield/make_ct_small_hu_fixture.py
"""
import os
import pathlib
import pytest
import nyxus

DATA = pathlib.Path(__file__).resolve().parent.parent / "data" / "hounsfield"
INTEN = str(DATA / "ct_small_hu.tif")
MASK = str(DATA / "ct_small_mask.tif")

pytestmark = pytest.mark.skipif(
    not (os.path.exists(INTEN) and os.path.exists(MASK)),
    reason="CT_small HU fixtures not present (run tests/data/hounsfield/make_ct_small_hu_fixture.py)",
)

# --- pydicom-pinned goldens (absolute HU, whole-image ROI) -------------------
HU_MIN = -896
HU_MAX = 1167
HU_MEAN = -119.0738525390625
HU_INTEGRATED = -1950906
OFFSET_BASE = HU_MIN          # floor(HU_min); HU is integer-valued so floor(min) == min
N = 128 * 128                 # ROI voxel count (whole 128x128 image)

FEATS = ["MIN", "MAX", "MEAN", "INTEGRATED_INTENSITY"]


def _featurize():
    nyx = nyxus.Nyxus(FEATS, preserve_hu=True)
    df = nyx.featurize_files([INTEN], [MASK], True)
    return {c: float(df[c].iloc[0]) for c in FEATS}


def test_hu_ct_small_pydicom():
    """Reconstructed absolute HU (offset + floor(HU_min)) == pydicom HU stats."""
    o = _featurize()

    # Nyxus reports the offset domain; the minimum HU maps to 0 by construction.
    assert o["MIN"] == pytest.approx(0.0)

    # Reconstruct absolute HU and compare to the independent pydicom goldens.
    assert o["MIN"] + OFFSET_BASE == pytest.approx(HU_MIN)
    assert o["MAX"] + OFFSET_BASE == pytest.approx(HU_MAX)
    assert o["MEAN"] + OFFSET_BASE == pytest.approx(HU_MEAN, rel=1e-6)
    assert o["INTEGRATED_INTENSITY"] + N * OFFSET_BASE == pytest.approx(HU_INTEGRATED)


def test_hu_ct_small_pydicom_no_wraparound():
    """Real signed CT (negative HU) must not wrap into billions on the unsigned cast."""
    o = _featurize()
    assert 0.0 <= o["MIN"] < o["MAX"] < 1e6
    assert o["MEAN"] < 1e6
