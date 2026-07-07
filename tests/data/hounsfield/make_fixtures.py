"""Generate tiny tiled TIFF fixtures for the Hounsfield-Unit (HU) loader tests.

Both fixtures are 16x16, single 16x16 tile, with pixel(r,c) = BASE + idx*STEP
where idx = r*16 + c, BASE = -1024 (air), STEP = 8. So values span
-1024 .. -1024 + 255*8 = 1016 — i.e. a CT/HU-like signed range crossing 0 (water).

  ct_int16.tif  : signed int16   (SampleFormat=2) -> exercises loadTile<int16_t>
  ct_float.tif  : float32        (SampleFormat=3) -> exercises loadTile_real_intens<float>

In HU mode with fpmin=-1024 the loader must map value -> value - floor(-1024) = value+1024,
so the stored feature-domain pixel equals idx*8 (0..2040). The C++ test recomputes this.

Run:  python make_fixtures.py   (needs numpy + tifffile)
"""
import numpy as np, tifffile, pathlib

HERE = pathlib.Path(__file__).parent
BASE, STEP, N = -1024, 8, 16

idx = np.arange(N * N).reshape(N, N)
vals = (BASE + idx * STEP)

tifffile.imwrite(HERE / "ct_int16.tif", vals.astype(np.int16),
                 tile=(16, 16), photometric="minisblack")
tifffile.imwrite(HERE / "ct_float.tif", vals.astype(np.float32),
                 tile=(16, 16), photometric="minisblack")
# all-ones ROI mask (single whole-image ROI) for the Python featurize test
tifffile.imwrite(HERE / "mask.tif", np.ones((N, N), np.uint16),
                 tile=(16, 16), photometric="minisblack")

# Report back for sanity
for name in ("ct_int16.tif", "ct_float.tif"):
    with tifffile.TiffFile(HERE / name) as t:
        p = t.pages[0]
        print(name, "tiled" if p.is_tiled else "STRIP",
              "sampleformat", p.tags.get("SampleFormat").value if p.tags.get("SampleFormat") else "?",
              "dtype", p.dtype, "shape", p.shape)
print("min", int(vals.min()), "max", int(vals.max()))
