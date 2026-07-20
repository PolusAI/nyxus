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


def write_calibrated(name, order, scale, unit):
    """A physically calibrated store: coordinateTransformations 'scale' != 1 and the
    space axes carry a unit, so the loader must surface physicalSize*/unit. Drives the
    --use-physical-spacing calibration path. Same coordinate encoding as write_store."""
    path = os.path.join(HERE, name)
    shutil.rmtree(path, ignore_errors=True)
    data = _data_for(order)
    chunks = tuple(1 if a == "z" else s for a, s in zip(order, data.shape))

    g = zarr.open_group(path, mode="w", zarr_format=2)
    a = g.create_array("0", shape=data.shape, dtype=data.dtype, chunks=chunks, compressors=None)
    a[:] = data

    axes = []
    for ax in order:
        e = dict(_AX[ax])
        if e["type"] == "space":
            e["unit"] = unit
        axes.append(e)
    ms = {"version": "0.4", "axes": axes,
          "datasets": [{"path": "0", "coordinateTransformations": [{"type": "scale", "scale": scale}]}]}
    g.attrs.put({"multiscales": [ms]})
    print("wrote %-22s (calibrated scale=%s unit=%s)" % (name, scale, unit))


def write_multichunk(name, order, chunk_y, chunk_x):
    """Same coordinate encoding as write_store, but each Y/X plane is split across a GRID of
    chunks instead of being one chunk. This is what real OME-Zarr looks like (chunks are
    typically 512x512), and it exercises the multi-tile volumetric assembly: a reader that
    fetches only chunk (0,0) returns wrong data for everything past the first chunk."""
    path = os.path.join(HERE, name)
    shutil.rmtree(path, ignore_errors=True)
    data = _data_for(order)
    chunks = tuple((chunk_y if a == "y" else chunk_x if a == "x" else 1) for a in order)

    g = zarr.open_group(path, mode="w", zarr_format=2)
    a = g.create_array("0", shape=data.shape, dtype=data.dtype, chunks=chunks, compressors=None)
    a[:] = data

    ms = {"version": "0.4", "axes": [_AX[ax] for ax in order],
          "datasets": [{"path": "0",
                        "coordinateTransformations": [{"type": "scale", "scale": [1.0] * len(order)}]}]}
    g.attrs.put({"multiscales": [ms]})
    print("wrote %-24s (chunk grid %dx%d over the %dx%d plane)"
          % (name, -(-Y // chunk_y), -(-X // chunk_x), Y, X))


def write_v3(name, order, compressors=None):
    """OME-Zarr 0.5 = Zarr **v3** store: metadata in zarr.json (not .zarray/.zattrs), the
    NGFF block nested under an 'ome' key, chunks under 0/c/<idx>/... The nyxus loader reads
    v3 via the z5 Dataset API (z5 3.x auto-detects v2 vs v3). Uncompressed by default so it
    reads with the current blosc+zlib z5 build (zstd-compressed v3 would need -DWITH_ZSTD)."""
    path = os.path.join(HERE, name)
    shutil.rmtree(path, ignore_errors=True)
    data = _data_for(order)
    chunks = tuple(1 if a == "z" else s for a, s in zip(order, data.shape))

    g = zarr.open_group(path, mode="w", zarr_format=3)
    a = g.create_array("0", shape=data.shape, dtype=data.dtype, chunks=chunks, compressors=compressors)
    a[:] = data
    g.attrs["ome"] = {"version": "0.5",
                      "multiscales": [{"axes": [_AX[ax] for ax in order],
                                       "datasets": [{"path": "0",
                                                     "coordinateTransformations": [{"type": "scale", "scale": [1.0] * len(order)}]}]}]}
    print("wrote %-24s (Zarr v3, axes=%s, compressors=%s)" % (name, order, compressors))


def write_v3_sharded(name, order):
    """Zarr v3 store using the ``sharding_indexed`` codec: several INNER chunks are packed
    into one shard object on disk (one file per shard, not per chunk). This is how large v3
    stores (incl. Axle's) actually lay out data. z5 3.x reads it transparently via
    ShardedDataset -- the nyxus loader is unchanged, it just calls readSubarray.

    Same TCZYX coordinate encoding as write_v3. Inner chunk (z,y,x)=(1,3,4) -> each 6x8 plane
    is a 2x2 grid of inner chunks; shard (1,6,8) packs all 4 inner chunks of a plane into one
    shard, so the read must unpack multiple inner chunks from a single shard file."""
    path = os.path.join(HERE, name)
    shutil.rmtree(path, ignore_errors=True)
    data = _data_for(order)                                           # [t,c,z,y,x], order tczyx
    inner = tuple((1 if a == "z" else 3 if a == "y" else 4 if a == "x" else 1) for a in order)
    shards = tuple((s if a in ("y", "x", "z") else 1) for a, s in zip(order, data.shape))

    g = zarr.open_group(path, mode="w", zarr_format=3)
    a = g.create_array("0", shape=data.shape, dtype=data.dtype,
                       chunks=inner, shards=shards, compressors=None)
    a[:] = data
    g.attrs["ome"] = {"version": "0.5",
                      "multiscales": [{"axes": [_AX[ax] for ax in order],
                                       "datasets": [{"path": "0",
                                                     "coordinateTransformations": [{"type": "scale", "scale": [1.0] * len(order)}]}]}]}
    print("wrote %-24s (Zarr v3 SHARDED, inner=%s shard=%s)" % (name, inner, shards))


def write_bad(name, shape, axes):
    """Write a deliberately malformed store (its 'axes' disagrees with the array)
    so the loader must reject it cleanly, not crash."""
    path = os.path.join(HERE, name)
    shutil.rmtree(path, ignore_errors=True)
    g = zarr.open_group(path, mode="w", zarr_format=2)
    chunks = tuple(1 if i == len(shape) - 3 else s for i, s in enumerate(shape))
    a = g.create_array("0", shape=shape, dtype="uint16", chunks=chunks, compressors=None)
    a[:] = 0
    g.attrs.put({"multiscales": [{"version": "0.4", "axes": axes,
        "datasets": [{"path": "0",
                      "coordinateTransformations": [{"type": "scale", "scale": [1.0] * len(axes)}]}]}]})
    print("wrote %-24s shape=%s axes=%d  (MALFORMED)" % (name, list(shape), len(axes)))


def main():
    # All 6 orderings of {t,c,z} before y,x -- the complete legal OME axis-order set.
    for o in ["tczyx", "tzcyx", "ctzyx", "cztyx", "ztcyx", "zctyx"]:
        write_store("dim5.ome.zarr" if o == "tczyx" else "dim5_%s.ome.zarr" % o, o)
    # 4D (rank-4 coverage): one time-only, one channel-only.
    write_store("dim4_tzyx.ome.zarr", "tzyx")
    write_store("dim4_czyx.ome.zarr", "czyx")
    # lower rank + fallback
    write_store("dim3_zyx.ome.zarr", "zyx")
    write_store("dim2_yx.ome.zarr", "yx")
    write_store("dim5_noaxes.ome.zarr", "tczyx", include_axes=False)
    # physically calibrated: anisotropic voxels (z 4x thicker than x/y after normalization).
    # order tczyx -> scale [t,c,z,y,x] = [1,1,2.0,0.5,0.5] -> physX=0.5 physY=0.5 physZ=2.0 um.
    write_calibrated("dim5_calibrated.ome.zarr", "tczyx", [1.0, 1.0, 2.0, 0.5, 0.5], "micrometer")
    # OME-Zarr 0.5 (Zarr v3): same 5D TCZYX encoding, but zarr.json metadata + 'ome'-wrapped NGFF
    write_v3("dim5_v3.ome.zarr", "tczyx")
    # plane split across a 2x2 chunk grid (3x4 chunks over the 6x8 plane) -> multi-tile assembly
    write_multichunk("dim5_multichunk.ome.zarr", "tczyx", 3, 4)
    # PARTIAL edge chunks: chunk (4,5) does NOT divide the 6x8 plane, so the last row-chunk is
    # 2 tall and the last col-chunk is 3 wide. Exercises the validH/validW seam clamp in the
    # volumetric assembly, which every exact-multiple fixture (above) leaves untested.
    write_multichunk("dim5_oddchunk.ome.zarr", "tczyx", 4, 5)
    # --- illegal / adversarial: must be rejected cleanly, not crash ---
    # 3D array but 'axes' declares 5 entries -> indexing the shape by axis role OOBs
    write_bad("bad_axes_count.ome.zarr", (Z, Y, X), [_AX[k] for k in ("t", "c", "z", "y", "x")])
    # axes present but none labeled x/y -> X/Y unresolvable
    write_bad("bad_no_xy.ome.zarr", (Z, Y, X),
              [{"name": "z", "type": "space"}, {"name": "a", "type": "space"}, {"name": "b", "type": "space"}])


if __name__ == "__main__":
    main()
