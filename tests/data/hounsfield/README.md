# Hounsfield-Unit (HU) loader test fixtures

Tiny tiled TIFFs used by `tests/test_hounsfield_loader.h` to verify the
`--preserve-hu` offset mapping through the real TIFF decode path.

| File | Type | Notes |
|---|---|---|
| `ct_int16.tif` | signed int16 (SampleFormat=2) | exercises `loadTile<int16_t>` (signed CT wraparound fix) |
| `ct_float.tif` | float32 (SampleFormat=3) | exercises `loadTile_real_intens<float>` |
| `ct_u16.dcm` | DICOM uint16, RescaleIntercept=−1024 | textbook CT: rescale then offset |
| `ct_i16.dcm` | DICOM int16 signed, intercept 0 | signed stored values, no wraparound |

All are 16×16, a single tile, encoding the SAME logical HU field
`HU(r,c) = -1024 + idx*8`, `idx = r*16 + c` (0..255) — a CT/HU-like range
−1024..1016 crossing 0 (water). In HU mode the loader maps HU → HU + 1024 = idx*8.

Regenerate with:

```
python make_fixtures.py         # TIFF fixtures  (needs numpy + tifffile)
python make_dicom_fixtures.py   # DICOM fixtures (needs numpy + pydicom)
```

## Real-scanner fixture

`ct_small.dcm` is pydicom's `CT_small.dcm` test file (128x128 signed int16,
RescaleSlope=1, RescaleIntercept=-1024, HU range -896..1167), vendored from the
pydicom distribution (MIT-licensed test data). It gives the HU tests a genuine
scanner slice alongside the synthetic ones. The `TEST_HU_LOADER_DICOM_CT_SMALL_*`
gtests assert pixel values computed independently with pydicom
(`RescaleSlope*stored + RescaleIntercept`, then offset by floor(HU min) = -896).
Obtain the original via: `python -c "from pydicom.data import get_testdata_file; print(get_testdata_file('CT_small.dcm'))"`.
