"""Generate the real-scanner HU TIFF fixture + pinned goldens for the pydicom
oracle test (test_hu_ct_small_pydicom.py).

Decodes pydicom's real CT_small.dcm (128x128 signed int16 CT slice) and applies
the DICOM rescale HU = RescaleSlope*stored + RescaleIntercept using pydicom as an
INDEPENDENT reference, then writes:

  ct_small_hu.tif   : true-HU int16 field (HU range -896..1167, crosses 0)
  ct_small_mask.tif : all-ones uint16 ROI (whole image)

...and prints the first-order HU statistics that the test pins as goldens.

Why this breaks the synthetic-fixture tautology (SPEC.md §5.2): the pixel values
come from a real scanner file decoded by an independent library, and the expected
MIN/MAX/MEAN/INTEGRATED come from pydicom+numpy — NOT from the same offset formula
the Nyxus loader applies. So it is a genuine oracle assertion (§3.1), the first one
covering HU *feature* values (not just loader pixels).

Run:  python make_ct_small_hu_fixture.py    (needs numpy + pydicom + tifffile)
"""
import pathlib
import numpy as np
import pydicom
from pydicom.data import get_testdata_file
import tifffile

HERE = pathlib.Path(__file__).parent

ds = pydicom.dcmread(get_testdata_file("CT_small.dcm"))
slope, intercept = float(ds.RescaleSlope), float(ds.RescaleIntercept)
stored = ds.pixel_array.astype(np.int64)
hu = (slope * stored + intercept).astype(np.int64)   # true HU, independent of Nyxus
rows, cols = hu.shape

tifffile.imwrite(HERE / "ct_small_hu.tif", hu.astype(np.int16),
                 tile=(64, 64), photometric="minisblack")
tifffile.imwrite(HERE / "ct_small_mask.tif", np.ones((rows, cols), np.uint16),
                 tile=(64, 64), photometric="minisblack")

print(f"pydicom {pydicom.__version__}  shape {rows}x{cols}  slope {slope} intercept {intercept}")
print("PINNED GOLDENS (absolute HU, whole-image ROI):")
print(f"  HU_MIN={int(hu.min())} HU_MAX={int(hu.max())} "
      f"HU_MEAN={hu.mean()!r} HU_INTEGRATED={int(hu.sum())}")
