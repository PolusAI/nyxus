"""Generate OME-TIFF fixtures whose every voxel encodes its own (x,y,z,c,t)
coordinate, so a reader that maps (z,c,t) to the wrong IFD (page) reads a
provably wrong value.

    value(x,y,z,c,t) = 1 + ((((t*C + c)*Z + z)*Y + y)*X + x)     # C=3,Z=4,Y=6,X=8

Stores (same encoding throughout; absent axes pinned to index 0):

  dim5.ome.tif        axes TCZYX -> DimensionOrder XYZCT (default: Z fastest)
  dim5_ctzyx.ome.tif  axes CTZYX -> DimensionOrder XYZTC (non-default: T before C)
                      -> proves the plane->IFD map honors DimensionOrder
  dim3_zyx.ome.tif    axes ZYX   3D
  dim2_yx.ome.tif     axes YX    2D
  dim3_plain.tif      plain multi-page TIFF (NO OME-XML) -> legacy page=Z fallback

tifffile writes OME-TIFF as multi-page (strip) with one IFD per (z,c,t) plane in
DimensionOrder raster order, so the loader's ifdForPlane() must reproduce that.

Run with the nyxus_build env python (tifffile):
    C:\\Users\\dvladi\\miniforge3\\envs\\nyxus_build\\python.exe gen_ome_tiff.py
"""
import os
import numpy as np
import tifffile

T, C, Z, Y, X = 2, 3, 4, 6, 8
HERE = os.path.dirname(os.path.abspath(__file__))


def encoded_tczyx():
    t, c, z, y, x = np.meshgrid(
        np.arange(T), np.arange(C), np.arange(Z), np.arange(Y), np.arange(X), indexing="ij")
    return (1 + ((((t * C + c) * Z + z) * Y + y) * X + x)).astype("uint16")


def _data_for(order):
    base = encoded_tczyx()                                       # [t,c,z,y,x]
    sel = tuple(slice(None) if a in order else 0 for a in "TCZYX")
    sub = base[sel]
    remaining = [a for a in "TCZYX" if a in order]
    perm = tuple(remaining.index(a) for a in order)
    return np.transpose(sub, perm).copy()


def write_ome(name, order):
    path = os.path.join(HERE, name)
    data = _data_for(order)
    tifffile.imwrite(path, data, photometric="minisblack", metadata={"axes": order}, ome=True)
    with tifffile.TiffFile(path) as tf:
        import re
        dimord = re.search(r'DimensionOrder="([A-Z]+)"', tf.ome_metadata).group(1)
        print("wrote %-20s axes=%-6s DimensionOrder=%s  IFDs=%d  tiled=%s"
              % (name, order, dimord, len(tf.pages), tf.pages[0].is_tiled))


def write_plain(name, order):
    """Plain multi-page TIFF, no OME-XML (tests the non-OME fallback)."""
    path = os.path.join(HERE, name)
    data = _data_for(order)          # e.g. ZYX -> a plain z-stack
    tifffile.imwrite(path, data, photometric="minisblack")
    with tifffile.TiffFile(path) as tf:
        print("wrote %-20s axes=%-6s (plain, no OME)  IFDs=%d" % (name, order, len(tf.pages)))


def write_tiled(name, order):
    """A TILED multi-plane OME-TIFF (one IFD per (z,c,t) plane, each plane internally
    tiled). tile=(16,16) makes the 6x8 plane a single tile, so the volumetric read
    (which assembles tile (0,0) of each plane) covers the whole plane. Exercises the
    tile-loader (z,c,t)->IFD path, distinct from the strip loaders."""
    path = os.path.join(HERE, name)
    data = _data_for(order)
    tifffile.imwrite(path, data, photometric="minisblack", metadata={"axes": order},
                     tile=(16, 16), ome=True)
    with tifffile.TiffFile(path) as tf:
        import re
        dimord = re.search(r'DimensionOrder="([A-Z]+)"', tf.ome_metadata).group(1)
        print("wrote %-20s axes=%-6s DimensionOrder=%s  IFDs=%d  tiled=%s"
              % (name, order, dimord, len(tf.pages), tf.pages[0].is_tiled))


# Multi-tile fixture dims. TIFF tile sizes must be multiples of 16, so the plane has to
# exceed 16 in BOTH directions to produce a real tile grid -- hence its own (larger Y/X,
# smaller C/Z/T) shape rather than the 6x8 plane the other fixtures share.
MT_T, MT_C, MT_Z, MT_Y, MT_X = 1, 2, 2, 32, 48


def write_multitile(name, T=MT_T, C=MT_C, Z=MT_Z, Y=MT_Y, X=MT_X):
    """A tiled OME-TIFF whose every plane spans a GRID of 16x16 tiles (default 32x48 plane
    -> 2x3 = 6 tiles), unlike dim5_tiled.ome.tif where a single tile covers the whole plane.
    A volumetric read that fetches only tile (0,0) returns wrong data for everything outside
    the first 16x16 corner. Same coordinate encoding as encoded_tczyx, its own dims.

    When Y or X is not a multiple of 16 the last tile of that row/column is PARTIAL, which
    exercises the validH/validW seam clamp in the volumetric assembly."""
    t, c, z, y, x = np.meshgrid(np.arange(T), np.arange(C), np.arange(Z),
                                np.arange(Y), np.arange(X), indexing="ij")
    data = (1 + ((((t * C + c) * Z + z) * Y + y) * X + x)).astype("uint16")
    path = os.path.join(HERE, name)
    tifffile.imwrite(path, data, photometric="minisblack", metadata={"axes": "TCZYX"},
                     tile=(16, 16), ome=True)
    with tifffile.TiffFile(path) as tf:
        print("wrote %-24s %dx%d plane, tile grid %dx%d (last %dx%d), IFDs=%d tiled=%s max=%d"
              % (name, Y, X, -(-Y // 16), -(-X // 16), Y % 16 or 16, X % 16 or 16,
                 len(tf.pages), tf.pages[0].is_tiled, data.max()))


def write_mask(name):
    """Single-channel, single-timeframe 3D (ZYX) label mask matching dim5's geometry.
    Used to test the 1-channel-mask : N-channel-intensity pairing: the mask is
    channel-agnostic and must be reused for every intensity channel."""
    path = os.path.join(HERE, name)
    m = np.zeros((Z, Y, X), "uint16")
    m[:, 1:5, 1:7] = 1                    # one ROI (label 1), interior so it has a real bbox
    tifffile.imwrite(path, m, photometric="minisblack", metadata={"axes": "ZYX"}, ome=True)
    print("wrote %-20s (single-channel ZYX label mask, ROI voxels=%d)" % (name, int(m.sum())))


def write_reordered(name):
    """An OME-TIFF whose planes are physically stored in REVERSED IFD order, with explicit
    <TiffData> blocks mapping each logical (z,c,t) to its scrambled IFD. This is what a reader
    that ignores TiffData and assumes canonical (contiguous-from-IFD-0) plane order gets wrong:
    it would read the reversed plane's pixels. Writers like bioformats emit per-plane TiffData;
    a non-canonical mapping (or a non-zero starting IFD) is exactly where the assumption breaks.

    Dims T=1,C=2,Z=3, plane 6x8, DimensionOrder XYZCT -> canonical ordinal ord = z + c*Z. The
    logical plane with ordinal `ord` is stored at physical IFD (5 - ord). Same encoded value as
    encoded_tczyx so the standard facade test can check it."""
    rT, rC, rZ = 1, 2, 3
    total = rZ * rC * rT
    # physical IFD p holds the logical plane whose ordinal is (total-1 - p)
    phys = np.empty((total, Y, X), "uint16")
    tiffdata = []
    for c in range(rC):
        for z in range(rZ):
            ord_ = z + c * rZ                      # canonical XYZCT ordinal (t=0)
            ifd = (total - 1) - ord_               # reversed physical IFD
            yy, xx = np.meshgrid(np.arange(Y), np.arange(X), indexing="ij")
            phys[ifd] = (1 + ((((0 * rC + c) * rZ + z) * Y + yy) * X + xx)).astype("uint16")
            tiffdata.append('<TiffData FirstC="%d" FirstT="0" FirstZ="%d" IFD="%d" PlaneCount="1"/>'
                            % (c, z, ifd))
    ome = ('<?xml version="1.0" encoding="UTF-8"?>'
           '<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">'
           '<Image ID="Image:0" Name="reordered">'
           '<Pixels ID="Pixels:0" DimensionOrder="XYZCT" Type="uint16" '
           'SizeX="%d" SizeY="%d" SizeZ="%d" SizeC="%d" SizeT="%d">%s'
           '</Pixels></Image></OME>'
           % (X, Y, rZ, rC, rT, "".join(tiffdata)))
    path = os.path.join(HERE, name)
    tifffile.imwrite(path, phys, description=ome, metadata=None, photometric="minisblack")
    with tifffile.TiffFile(path) as tf:
        got = (tf.pages[0].description or "")[:4]
        print("wrote %-24s IFDs=%d reversed, TiffData blocks=%d, desc starts %r"
              % (name, len(tf.pages), len(tiffdata), got))


def write_bad_rgb(name):
    """RGB OME-TIFF -- nyxus is grayscale-only, so the loader must reject it."""
    path = os.path.join(HERE, name)
    data = np.zeros((Y, X, 3), "uint8")
    tifffile.imwrite(path, data, photometric="rgb", metadata={"axes": "YXS"}, ome=True)
    print("wrote %-20s (RGB -> not grayscale)" % name)


def write_bad_corrupt(name):
    """Not a TIFF at all -- TIFFOpen must fail cleanly."""
    path = os.path.join(HERE, name)
    with open(path, "wb") as f:
        f.write(b"this is definitely not a TIFF file\n\x00\x01\x02\x03")
    print("wrote %-20s (corrupt -> open fails)" % name)


def main():
    # All 6 orderings of {T,C,Z} before Y,X -> the 6 legal OME DimensionOrder values.
    for o in ["TCZYX", "TZCYX", "CTZYX", "CZTYX", "ZTCYX", "ZCTYX"]:
        write_ome("dim5.ome.tif" if o == "TCZYX" else "dim5_%s.ome.tif" % o.lower(), o)
    # 4D (rank-4 coverage): one time-only, one channel-only.
    write_ome("dim4_tzyx.ome.tif", "TZYX")
    write_ome("dim4_czyx.ome.tif", "CZYX")
    # lower rank + non-OME fallback
    write_ome("dim3_zyx.ome.tif", "ZYX")
    write_ome("dim2_yx.ome.tif", "YX")
    write_plain("dim3_plain.tif", "ZYX")
    # single-channel label mask (for the 1-mask : N-channel-intensity pairing test)
    write_mask("dim3_mask.ome.tif")
    # non-canonical <TiffData> plane->IFD mapping (planes stored reversed) -> honors TiffData
    write_reordered("dim5_reordered.ome.tif")
    # TILED multi-plane OME-TIFF (5D) -> exercises the tile-loader (z,c,t)->IFD path
    write_tiled("dim5_tiled.ome.tif", "TCZYX")
    # planes spanning a real 2x3 tile grid -> multi-tile volumetric assembly
    write_multitile("dim5_multitile.ome.tif")
    # PARTIAL edge tiles: 40x24 plane / 16 -> 3x2 tiles, last row-tile 8 tall, last col-tile
    # 8 wide. Exercises the seam clamp the exact-multiple multitile fixture leaves untested.
    write_multitile("dim5_oddtile.ome.tif", T=1, C=2, Z=1, Y=40, X=24)
    # --- illegal / adversarial: must be rejected cleanly, not crash ---
    write_bad_rgb("bad_rgb.ome.tif")
    write_bad_corrupt("bad_corrupt.tif")


if __name__ == "__main__":
    main()
