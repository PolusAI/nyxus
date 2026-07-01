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

| Store            | Size (HxW) | dtype  | Chunk     | Tile grid | Pixel value            |
|------------------|-----------:|--------|-----------|-----------|------------------------|
| `test.ome.zarr`  |   512x512  | uint16 | 1024x1024 | 1x1       | `(row + col) % 65536`  |
| `multi.ome.zarr` | 1500x1200  | uint16 | 1024x1024 | 2x2*      | `(row*7 + col*3) % 65536` |

\* `multi.ome.zarr` has partial edge tiles (1500 = 1024 + 476, 1200 = 1024 + 176),
so it also exercises the loaders' partial-tile clipping path.

Deterministic checksums asserted by the tests:

- `test.ome.zarr`  sum of all pixels = `133955584`
- `multi.ome.zarr` sum of all pixels = `12681000000`

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
