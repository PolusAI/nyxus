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


def write_mask(name):
    """Single-channel, single-timeframe 3D (ZYX) label mask matching dim5's geometry.
    Used to test the 1-channel-mask : N-channel-intensity pairing: the mask is
    channel-agnostic and must be reused for every intensity channel."""
    path = os.path.join(HERE, name)
    m = np.zeros((Z, Y, X), "uint16")
    m[:, 1:5, 1:7] = 1                    # one ROI (label 1), interior so it has a real bbox
    tifffile.imwrite(path, m, photometric="minisblack", metadata={"axes": "ZYX"}, ome=True)
    print("wrote %-20s (single-channel ZYX label mask, ROI voxels=%d)" % (name, int(m.sum())))


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
    # --- illegal / adversarial: must be rejected cleanly, not crash ---
    write_bad_rgb("bad_rgb.ome.tif")
    write_bad_corrupt("bad_corrupt.tif")


if __name__ == "__main__":
    main()
