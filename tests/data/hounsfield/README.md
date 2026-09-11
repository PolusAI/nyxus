# Hounsfield-Unit (HU) loader test fixtures

Tiny tiled TIFFs used by the HU loader mechanics tests in `tests/test_2d_hu_mechanics.h`
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

Used by `tests/python/test_3d_hu_nifti_regression.py` to verify `preserve_hu` through the 3D NIfTI
loader path (`RawNiftiLoader`/`NiftiLoader`).

| File | Type | Notes |
|---|---|---|
| `ct3d_int16.nii` | 8×8×8 signed int16, `scl_slope=2`, `scl_inter=-1024` | stored `idx-200` (−200..311, crosses 0); true HU = 2·stored−1024 |
| `ct3d_frac.nii` | 8×8×8 signed int16, `scl_slope=0.5`, `scl_inter=-1024` | stored `idx` (0..511); physical = 0.5·idx−1024 over −1024..−768.5, half the voxels fractional |
| `ct3d_nan.nii` | 8×8×8 float32, no rescale | `v(idx) = idx`, with NaN at 3, +Inf at 5, −Inf at 7 |
| `mask3d.nii` | 8×8×8 uint8, all ones | single ROI over the whole volume, shared by all three |

The non-unit `scl_slope` makes the HU rescale observable: in HU mode the loader maps each voxel to
`u = round((2·stored − 1024) − floor(HU min)) = 2·stored + 400` (offset domain 0..1022, mean 511),
whereas with the flag off it keeps the raw-stored (shifted) values (MAX 511). The 348-byte
NIfTI-1 header was written by hand, so the fixtures carry no `nibabel` dependency.

`ct3d_frac.nii` and `ct3d_nan.nii` are consumed by
`tests/python/test_3d_nifti_offset_map_mechanics.py`, which pins the two properties of the offset
map that `ct3d_int16.nii` cannot show:

- **Which way the map narrows, and when.** A `scl_slope` of 2 makes every rescaled value integral,
  so rounding and truncation agree on it. At 0.5 they do not: the offset is `floor(−1024) = −1024`
  and the stored grey level is `round(0.5·idx)` under `--preserve-hu` (MAX −768, MEAN −896) against
  `trunc(0.5·idx)` without it (MAX −769, MEAN −896.5). Both directions are pinned. The flag is what
  selects the narrowing because `inten_scale` is 1 on the offset branch — a fraction dropped there
  is unrecoverable — while a real-valued volume read *without* the flag has always truncated, which
  is what the 3D texture goldens encode.
- **Non-finite voxels.** They are left out of the scanned extrema and stored as grey level 0, so the
  finite range stays `[0, 511]` and the mean is `(130816 − 3 − 5 − 7)/512 = 255.470703125`.

Both were produced by patching `ct3d_int16.nii`'s own 352-byte header block — its `datatype`
(offset 70), `bitpix` (72), `scl_slope` (112), `scl_inter` (116), `cal_max` (124) and `cal_min`
(128) — and appending new little-endian payload, so every field the loader does not read stays
byte-identical to the fixture already in the tree:

```python
import struct, math, pathlib
hdr = bytearray(pathlib.Path("ct3d_int16.nii").read_bytes()[:352])   # 348 header + 4 extender

def write(name, datatype, bitpix, slope, inter, payload, cal_min, cal_max):
    h = bytearray(hdr)
    struct.pack_into("<h", h, 70, datatype); struct.pack_into("<h", h, 72, bitpix)
    struct.pack_into("<f", h, 112, slope);   struct.pack_into("<f", h, 116, inter)
    struct.pack_into("<f", h, 124, cal_max); struct.pack_into("<f", h, 128, cal_min)
    pathlib.Path(name).write_bytes(bytes(h) + payload)

write("ct3d_frac.nii", 4, 16, 0.5, -1024.0,
      struct.pack("<512h", *range(512)), -1024.0, -768.5)

v = [float(i) for i in range(512)]
v[3], v[5], v[7] = math.nan, math.inf, -math.inf
write("ct3d_nan.nii", 16, 32, 1.0, 0.0, struct.pack("<512f", *v), 0.0, 511.0)
```

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

Used by `tests/python/test_2d_hu_ct_small_pydicom.py` — the oracle test that vets HU *feature*
values (MIN/MAX/MEAN/INTEGRATED) against pydicom (docs/vetting SPEC.md §4 token `pydicom`). The
pixels come from pydicom's decode of a real scanner slice and the goldens are pinned from
pydicom+numpy, so it is an independent oracle, not a self-consistency snapshot (§5.2). CI
consumes only the committed TIFFs.

