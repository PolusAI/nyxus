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
| `nonfinite.ome.zarr` |   32x32  | float32 | 32x32     | 1x1       | see below                   |
| `dim5.ome.zarr`    |     6x8    | uint16  | 2x3x1x6x8 | 1x1       | 5D coordinate encoding (below) |

\* `multi.ome.zarr` has partial edge tiles (1500 = 1024 + 476, 1200 = 1024 + 176),
so it also exercises the loaders' partial-tile clipping path.

`signed.ome.zarr` and `float.ome.zarr` cover the load-time intensity map: every sample of
the signed store is negative (range `[-1013, -951]`, a CT-like air value at the minimum) and
every sample of the real-valued store is fractional and straddles zero (range `[-10.5, 5.0]`),
so a loader that narrows either into the unsigned pipeline type is caught rather than
silently returning wrapped or truncated grey levels. They are uncompressed and 32x32, which
keeps each store a few kB.

`nonfinite.ome.zarr` covers the classes the unsigned accessor cannot convert at all. Row 0
opens with `NaN`, `+Inf`, `-Inf`, `5e9` (exact in float32, and above `UINT32_MAX`), `-7.5`
and `3.75`; every other sample is `1.0`. A real-valued store is free to hold any of them, and
`RawOmezarrLoader::get_uint32_pixel` is what a float32 *mask* is read through -- converting a
non-finite or an out-of-range double to an unsigned integer is undefined, not merely lossy.

The table above covers the stores written by hand or by bfio. Everything else in this
directory is written by `gen_dim5.py`, whose module docstring and per-writer docstrings are the
inventory for those; the four described under "Converter-shaped and diagnostic stores" below are
called out here because what they cover is a property of the *container*, not of the pixels.

### Converter-shaped and diagnostic stores

These exist because the rest of the 3D/5D fixtures here are little-endian, flat-separator and
uncompressed — i.e. none of them resembles what `bioformats2raw` actually writes. They were
added after a cross-container run over real converter output.

| Store | What is unusual about it | Expected |
|-------|--------------------------|----------|
| `dim3_nested.ome.zarr` | `dimension_separator: "/"` (chunk `(1,0,0)` is the file `0/1/0/0`) plus a blosc/lz4 codec. Nesting is the NGFF 0.4 default and what the converter emits; `--no-nested` downgrades the declared version to 0.1. | reads identically to `dim3_zyx.ome.zarr` |
| `bigendian.ome.zarr` | `dtype: ">u2"`, payload genuinely byte-swapped. `bioformats2raw` through 0.9.x writes big-endian and has no switch to change it. z5 maps only the `<` and `|` spellings, so the array cannot be opened. | refused, and the message names big-endianness and the remedy |
| `b2r_layout.ome.zarr` | `bioformats2raw` layout: the root carries only `{"bioformats2raw.layout": 3}` and the image is the child group `0`. The data is fine; the path is one level too high. | refused, and the message names the layout and the series group |
| `b2r_layout_v3.ome.zarr` | the same, in zarr v3 / NGFF 0.5, where the key sits under `"ome"`. | as above |

`tests/test_omezarr_mechanics.h` asserts the first through
`test_omezarr_nested_chunk_keys_mechanics()` and the other three through
`test_omezarr_diagnosed_refusals_mechanics()`, which checks the message text rather than only
that a throw happened — the throw alone was already there.

### `dim5.ome.zarr` — 5D channel/timeframe addressability

A genuinely 5D store, shape `(T=2, C=3, Z=4, Y=6, X=8)`, chunked `(T,C,1,Y,X)` —
one chunk per z-slice (4 chunk files) — and written **uncompressed** (so it reads
with header-only z5). Z gets its own chunk because the loader maps the layer index
as `layer*tileDepth` (`tileDepth = chunks[2]`); C and T are addressed directly by
the read offset, so they don't need separate chunks. Every voxel encodes its own
coordinate:

```
value(x,y,z,c,t) = 1 + ((((t*C + c)*Z + z)*Y + y)*X + x)    # C=3, Z=4, Y=6, X=8
```

`test_omezarr_5d_channel_time_addressing` / `test_raw_omezarr_5d_channel_time_addressing`
read plane `(z,c,t)` via `loadTileFromFile(..., layer=z, channel=c, timeframe=t, ...)`
and assert every value matches the encoding — so a loader that ignored C/T (offset
pinned to `{0,0,...}`) returns the wrong plane and fails. Generated by `gen_dim5.py`
(zarr 3.x). Regenerate: `python gen_dim5.py`.

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

`signed.ome.zarr`, `float.ome.zarr` and `nonfinite.ome.zarr` are written directly rather than
through bfio, so they stay uncompressed and depend on no codec. Each is a root group whose `.zattrs` names
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

nonfinite = np.full((H, W), 1.0, dtype="<f4")
nonfinite[0, :6] = [np.nan, np.inf, -np.inf, 5.0e9, -7.5, 3.75]
write_store("nonfinite.ome.zarr", nonfinite, "<f4")
```
