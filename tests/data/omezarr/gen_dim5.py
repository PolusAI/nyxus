"""Generate OME-Zarr v2 fixtures whose every voxel encodes its own (x,y,z,c,t)
coordinate, so a reader that mis-orders or fails to address the C/T/Z axes reads
a provably wrong value.

    value(x,y,z,c,t) = 1 + ((((t*C + c)*Z + z)*Y + y)*X + x)     # C=3,Z=4,Y=6,X=8

Stores (same encoding throughout; absent axes are pinned to index 0):

  dim5.ome.zarr        axes t,c,z,y,x   default TCZYX
  dim5_ctzyx.ome.zarr  axes c,t,z,y,x   non-default (C/T swapped) -> proves 'axes' is honored
  dim3_zyx.ome.zarr    axes z,y,x       3D -> proves rank-3 axis mapping
  dim2_yx.ome.zarr     axes y,x         2D -> proves rank-2 axis mapping
  dim5_noaxes.ome.zarr (no 'axes' key)  5D TCZYX -> proves the legacy fallback

All chunked one z-slice per chunk and written UNCOMPRESSED (compressor=None) so
they read with header-only z5 (no blosc/zlib link).

Run with the nyxus_build env python (zarr 3.x):
    C:\\Users\\dvladi\\miniforge3\\envs\\nyxus_build\\python.exe gen_dim5.py
"""
import os
import shutil
import json
import numpy as np
import zarr

T, C, Z, Y, X = 2, 3, 4, 6, 8
HERE = os.path.dirname(os.path.abspath(__file__))

_AX = {
    "t": {"name": "t", "type": "time"},
    "c": {"name": "c", "type": "channel"},
    "z": {"name": "z", "type": "space"},
    "y": {"name": "y", "type": "space"},
    "x": {"name": "x", "type": "space"},
}


def encoded_tczyx():
    """Logical volume indexed [t,c,z,y,x]."""
    t, c, z, y, x = np.meshgrid(
        np.arange(T), np.arange(C), np.arange(Z), np.arange(Y), np.arange(X), indexing="ij")
    return (1 + ((((t * C + c) * Z + z) * Y + y) * X + x)).astype("uint16")


def _data_for(order):
    """Array laid out in `order` (a subset/permutation of 'tczyx'); axes absent
    from `order` are pinned to index 0."""
    base = encoded_tczyx()                                        # [t,c,z,y,x]
    sel = tuple(slice(None) if a in order else 0 for a in "tczyx")
    sub = base[sel]                                               # absent axes dropped
    remaining = [a for a in "tczyx" if a in order]               # kept axes, tczyx order
    perm = tuple(remaining.index(a) for a in order)              # -> requested order
    return np.transpose(sub, perm).copy()


def write_store(name, order, include_axes=True):
    path = os.path.join(HERE, name)
    shutil.rmtree(path, ignore_errors=True)
    data = _data_for(order)
    chunks = tuple(1 if a == "z" else s for a, s in zip(order, data.shape))

    g = zarr.open_group(path, mode="w", zarr_format=2)
    a = g.create_array("0", shape=data.shape, dtype=data.dtype, chunks=chunks, compressors=None)
    a[:] = data

    ms = {"version": "0.4",
          "datasets": [{"path": "0",
                        "coordinateTransformations": [{"type": "scale", "scale": [1.0] * len(order)}]}]}
    if include_axes:
        ms["axes"] = [_AX[a] for a in order]
    g.attrs.put({"multiscales": [ms]})

    meta = json.load(open(os.path.join(path, "0", ".zarray")))
    print("wrote %-22s axes=%-6s shape=%s chunks=%s%s"
          % (name, order if include_axes else "(none)", meta["shape"], meta["chunks"],
             "" if include_axes else "  [no axes -> legacy fallback]"))


def main():
    write_store("dim5.ome.zarr", "tczyx")
    write_store("dim5_ctzyx.ome.zarr", "ctzyx")
    write_store("dim3_zyx.ome.zarr", "zyx")
    write_store("dim2_yx.ome.zarr", "yx")
    write_store("dim5_noaxes.ome.zarr", "tczyx", include_axes=False)


if __name__ == "__main__":
    main()
