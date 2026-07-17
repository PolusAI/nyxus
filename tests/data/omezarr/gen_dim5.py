"""Generate dim5.ome.zarr: a small 5D (T,C,Z,Y,X) OME-Zarr v2 fixture whose
every voxel encodes its own (x,y,z,c,t) coordinate, so a reader that mis-orders
or fails to address the C/T/Z axes reads a provably wrong value.

    value(x,y,z,c,t) = 1 + ((((t*C + c)*Z + z)*Y + y)*X + x)

Layout: shape (T=2, C=3, Z=4, Y=6, X=8), chunked (T,C,1,Y,X) = one chunk per
z-slice (4 chunk files total). Z needs its own chunk because the loader maps the
layer index as layer*tileDepth (tileDepth = chunks[2]); C and T are addressed
directly by the read offset, so they don't need separate chunks. Written
UNCOMPRESSED (compressor=None) so the store reads with header-only z5 (no
blosc/zlib link needed).

Run with the nyxus_build env python (zarr 3.x):
    C:\\Users\\dvladi\\miniforge3\\envs\\nyxus_build\\python.exe gen_dim5.py
"""
import os
import shutil
import numpy as np
import zarr

T, C, Z, Y, X = 2, 3, 4, 6, 8
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "dim5.ome.zarr")


def encoded():
    t, c, z, y, x = np.meshgrid(
        np.arange(T), np.arange(C), np.arange(Z), np.arange(Y), np.arange(X), indexing="ij")
    return (1 + ((((t * C + c) * Z + z) * Y + y) * X + x)).astype("uint16")


def main():
    shutil.rmtree(OUT, ignore_errors=True)
    data = encoded()

    g = zarr.open_group(OUT, mode="w", zarr_format=2)
    a = g.create_array("0", shape=data.shape, dtype=data.dtype,
                       chunks=(T, C, 1, Y, X), compressors=None)
    a[:] = data

    g.attrs.put({"multiscales": [{
        "version": "0.4",
        "axes": [{"name": "t", "type": "time"}, {"name": "c", "type": "channel"},
                 {"name": "z", "type": "space"}, {"name": "y", "type": "space"},
                 {"name": "x", "type": "space"}],
        "datasets": [{"path": "0",
                      "coordinateTransformations": [{"type": "scale", "scale": [1, 1, 1, 1, 1]}]}],
    }]})

    # sanity: uncompressed?
    import json
    meta = json.load(open(os.path.join(OUT, "0", ".zarray")))
    print("shape", meta["shape"], "chunks", meta["chunks"], "compressor", meta["compressor"])
    print("wrote", OUT)


if __name__ == "__main__":
    main()
