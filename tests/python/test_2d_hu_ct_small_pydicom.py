"""Oracle test: Nyxus first-order HU feature values vs an INDEPENDENT pydicom reference.

Unlike the synthetic Hounsfield fixtures (whose expected values are recomputed from the
same offset formula the loader uses — self-consistency, SPEC.md §5.2), this test vets the
actual first-order HU features against numbers derived by pydicom from a REAL scanner slice:

  fixture : ct_small_hu.tif — the true-HU int16 field of pydicom's CT_small.dcm
            (128x128, HU range -896..1167).
  oracle  : pydicom 3.0.2 — HU = RescaleSlope*stored + RescaleIntercept, then numpy stats.

The loader carries the slice on the offset map u = HU - floor(HU_min) and the intensity family
adds that offset back, so Nyxus reports absolute Hounsfield units directly and they must equal
the pinned pydicom goldens. This is the `vetted-by-oracle` assertion on HU *feature* outputs
(docs/vetting SPEC.md §3.1); the other HU tests are analytic / mechanics / invariant.

Provenance (record at the golden site, SPEC.md §6.4):
  tool    : pydicom 3.0.2, get_testdata_file('CT_small.dcm')
  config  : RescaleSlope=1.0, RescaleIntercept=-1024.0; whole-image all-ones ROI
  fixture : tests/data/hounsfield/ct_small_hu.tif
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
    reason="CT_small HU fixtures not present in tests/data/hounsfield",
)

# --- pydicom-pinned goldens (absolute HU, whole-image ROI) -------------------
HU_MIN = -896
HU_MAX = 1167
HU_MEAN = -119.0738525390625
HU_INTEGRATED = -1950906
N = 128 * 128                 # ROI voxel count (whole 128x128 image)

FEATS = ["MIN", "MAX", "MEAN", "INTEGRATED_INTENSITY"]


def _featurize():
    nyx = nyxus.Nyxus(FEATS, preserve_hu=True)
    df = nyx.featurize_files([INTEN], [MASK], True)
    return {c: float(df[c].iloc[0]) for c in FEATS}


def test_2d_hu_ct_small_values_pydicom():
    """Reported absolute HU == pydicom HU stats, with no reconstruction by the caller."""
    o = _featurize()

    assert o["MIN"] == pytest.approx(HU_MIN)
    assert o["MAX"] == pytest.approx(HU_MAX)
    assert o["MEAN"] == pytest.approx(HU_MEAN, rel=1e-6)
    assert o["INTEGRATED_INTENSITY"] == pytest.approx(HU_INTEGRATED)


def test_2d_hu_ct_small_no_wraparound_pydicom():
    """Real signed CT (negative HU) must not wrap into billions on the unsigned cast."""
    o = _featurize()
    assert -1e6 < o["MIN"] < o["MAX"] < 1e6
    assert HU_MIN <= o["MEAN"] <= HU_MAX


def test_2d_hu_ct_small_min_is_the_true_hounsfield_minimum_pydicom():
    """The invariant the reported defect would have failed: for a CT read in Hounsfield units
    MIN is the ROI's true minimum in Hounsfield units, not that minimum displaced by whatever
    offset the loader needed to keep the buffer unsigned. Needs no oracle beyond the range."""
    o = _featurize()
    assert o["MIN"] == pytest.approx(HU_MIN)
    assert o["MIN"] <= o["MEAN"] <= o["MAX"]
