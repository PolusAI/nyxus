# Hounsfield-Unit (HU) loader test fixtures

Tiny tiled TIFFs used by the HU loader mechanics tests in `tests/test_hu_mechanics.h`
to verify the `--preserve-hu` offset mapping through the real TIFF decode path.

| File | Type | Notes |
|---|---|---|
| `ct_int16.tif` | signed int16 (SampleFormat=2) | exercises `loadTile<int16_t>` (signed CT wraparound fix) |
| `ct_float.tif` | float32 (SampleFormat=3) | exercises `loadTile_real_intens<float>` |
| `ct_u16.dcm` | DICOM uint16, RescaleIntercept=−1024 | textbook CT: rescale then offset |
| `ct_i16.dcm` | DICOM int16 signed, intercept 0 | signed stored values, no wraparound |

All are 16×16, a single tile, encoding the SAME logical HU field
`HU(r,c) = -1024 + idx*8`, `idx = r*16 + c` (0..255) — a CT/HU-like range
−1024..1016 crossing 0 (water). In HU mode the loader maps HU → HU + 1024 = idx*8.

These fixtures are committed to the repository and consumed as-is. The construction
of each file is documented in this README so the ground truth the tests assert
against stays auditable.

## 3D NIfTI fixtures

Used by `tests/python/test_hounsfield_nifti.py` to verify `preserve_hu` through the 3D NIfTI
loader path (`RawNiftiLoader`/`NiftiLoader`).

| File | Type | Notes |
|---|---|---|
| `ct3d_int16.nii` | 8×8×8 signed int16, `scl_slope=2`, `scl_inter=-1024` | stored `idx-200` (−200..311, crosses 0); true HU = 2·stored−1024 |
| `mask3d.nii` | 8×8×8 uint8, all ones | single ROI over the whole volume |

The non-unit `scl_slope` makes the HU rescale observable: in HU mode the loader maps each voxel to
`u = round((2·stored − 1024) − floor(HU min)) = 2·stored + 400` (offset domain 0..1022, mean 511),
whereas with the flag off it keeps the raw-stored (shifted) values (MAX 511). The 348-byte
NIfTI-1 header was written by hand, so the fixtures carry no `nibabel` dependency.

## Real-scanner fixture

`ct_small.dcm` is pydicom's `CT_small.dcm` test file (128x128 signed int16,
RescaleSlope=1, RescaleIntercept=-1024, HU range -896..1167), vendored from the
pydicom distribution (MIT-licensed test data). It gives the HU tests a genuine
scanner slice alongside the synthetic ones. The `TEST_HU_LOADER_DICOM_CT_SMALL_*`
gtests assert pixel values computed independently with pydicom
(`RescaleSlope*stored + RescaleIntercept`, then offset by floor(HU min) = -896).
Obtain the original via: `python -c "from pydicom.data import get_testdata_file; print(get_testdata_file('CT_small.dcm'))"`.

## pydicom oracle fixture (feature-level)

| File | Type | Notes |
|---|---|---|
| `ct_small_hu.tif` | 128×128 signed int16 | the **true-HU field** of `CT_small.dcm` (HU = slope·stored + intercept), HU range −896..1167 |
| `ct_small_mask.tif` | 128×128 uint16, all ones | whole-image ROI |

Used by `tests/python/test_hu_ct_small_pydicom.py` — the oracle test that vets HU *feature*
values (MIN/MAX/MEAN/INTEGRATED) against pydicom (docs/vetting SPEC.md §4 token `pydicom`). The
pixels come from pydicom's decode of a real scanner slice and the goldens are pinned from
pydicom+numpy, so it is an independent oracle, not a self-consistency snapshot (§5.2). CI
consumes only the committed TIFFs.

