# OME-Zarr test datasets

These datasets are consumed by `tests/test_omezarr.h`, which exercises the two
z5-based OME-Zarr readers:

- `NyxusOmeZarrLoader` (`src/nyx/omezarr.h`)  — the Hedgehog tile-loader
- `RawOmezarrLoader`   (`src/nyx/raw_omezarr.h`) — the raw-format loader

Both readers expect a **zarr v2** store laid out as OME-NGFF: a root group whose
`.zattrs` carries `multiscales[0].datasets[0].path`, pointing at a 5D
`(T, C, Z, Y, X)` array. The loaders read `shape[2..4]` as depth/height/width
and `chunks[2..4]` as the tile sizes.

## Datasets

| Store              | Size (HxW) | dtype   | Chunk     | Tile grid | Pixel value                 |
|--------------------|-----------:|---------|-----------|-----------|-----------------------------|
| `test.ome.zarr`    |   512x512  | uint16  | 1024x1024 | 1x1       | `(row + col) % 65536`       |
| `multi.ome.zarr`   | 1500x1200  | uint16  | 1024x1024 | 2x2*      | `(row*7 + col*3) % 65536`   |
| `signed.ome.zarr`  |     32x32  | int16   | 32x32     | 1x1       | `-1013 + (row + col)`       |
| `float.ome.zarr`   |     32x32  | float32 | 32x32     | 1x1       | `-10.5 + 0.25*(row + col)`  |

\* `multi.ome.zarr` has partial edge tiles (1500 = 1024 + 476, 1200 = 1024 + 176),
so it also exercises the loaders' partial-tile clipping path.

`signed.ome.zarr` and `float.ome.zarr` cover the load-time intensity map: every sample of
the signed store is negative (range `[-1013, -951]`, a CT-like air value at the minimum) and
every sample of the real-valued store is fractional and straddles zero (range `[-10.5, 5.0]`),
so a loader that narrows either into the unsigned pipeline type is caught rather than
silently returning wrapped or truncated grey levels. They are uncompressed and 32x32, which
keeps each store a few kB.

Deterministic checksums asserted by the tests:

- `test.ome.zarr`   sum of all pixels = `133955584`
- `multi.ome.zarr`  sum of all pixels = `12681000000`
- `signed.ome.zarr` sum of all pixels = `-1005568`

## Regeneration

The datasets were generated with [bfio](https://pypi.org/project/bfio/) (writes
zarr v2 OME-Zarr by default):

```python
import numpy as np
from bfio import BioWriter

# test.ome.zarr
H, W = 512, 512
Y, X = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
img = ((Y.astype(np.uint32) + X.astype(np.uint32)) % 65536).astype(np.uint16)
with BioWriter("test.ome.zarr", X=W, Y=H, Z=1, C=1, T=1, dtype=np.uint16) as bw:
    bw[:] = img[..., np.newaxis, np.newaxis, np.newaxis]

# multi.ome.zarr
H, W = 1500, 1200
Y, X = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
img = ((Y.astype(np.uint32) * 7 + X.astype(np.uint32) * 3) % 65536).astype(np.uint16)
with BioWriter("multi.ome.zarr", X=W, Y=H, Z=1, C=1, T=1, dtype=np.uint16) as bw:
    bw[:] = img[..., np.newaxis, np.newaxis, np.newaxis]
```

bfio's default tile size (1024) determines the chunk size; the image dimensions
above were chosen so `multi.ome.zarr` produces a 2x2 grid with partial edges.

`signed.ome.zarr` and `float.ome.zarr` are written directly rather than through bfio, so
they stay uncompressed and depend on no codec. Each is a root group whose `.zattrs` names
one 5D `(T, C, Z, Y, X)` dataset, plus a single full-size chunk `0/0.0.0.0.0` holding the
raw little-endian samples:

```python
import json, os
import numpy as np

def write_store(name, arr, dtype_str):
    h, w = arr.shape
    os.makedirs(os.path.join(name, "0"), exist_ok=True)
    json.dump({"zarr_format": 2}, open(os.path.join(name, ".zgroup"), "w"), indent=4)
    json.dump({"multiscales": [{"version": "0.1", "name": name,
                                "datasets": [{"path": "0"}],
                                "metadata": {"method": "mean"}}]},
              open(os.path.join(name, ".zattrs"), "w"), indent=2)
    json.dump({"shape": [1, 1, 1, h, w], "chunks": [1, 1, 1, h, w], "dtype": dtype_str,
               "fill_value": 0, "order": "C", "filters": None,
               "dimension_separator": ".", "compressor": None, "zarr_format": 2},
              open(os.path.join(name, "0", ".zarray"), "w"), indent=4)
    open(os.path.join(name, "0", "0.0.0.0.0"), "wb").write(arr.tobytes(order="C"))

H = W = 32
r, c = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
write_store("signed.ome.zarr", (-1013 + (r + c)).astype("<i2"), "<i2")
write_store("float.ome.zarr", (-10.5 + 0.25 * (r + c)).astype("<f4"), "<f4")
```
