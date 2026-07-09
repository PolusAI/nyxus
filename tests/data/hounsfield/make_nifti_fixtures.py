"""Generate synthetic 3D NIfTI CT fixtures for the --preserve-hu / preserve_hu regression tests.

Writes two NIfTI-1 (.nii) files next to this script:

  ct3d_int16.nii  -- 8x8x8 signed int16 volume, stored pixel(idx) = idx - 200 (idx 0..511),
                     so stored values run -200..311 and cross zero. scl_slope=2, scl_inter=-1024,
                     i.e. true HU = 2*stored - 1024 (range -1424..-402).
  mask3d.nii      -- 8x8x8 uint8 volume, all ones (single ROI covering the whole volume).

In preserve_hu mode the loader maps each voxel to  u = round(HU - floor(HU_min))
= round((2*stored - 1024) - (-1424)) = 2*stored + 400, so the offset-domain feature pixels are
0..1022 (min voxel stored=-200 -> 0, max voxel stored=311 -> 1022) with mean 511. The maximum grey
value 1022 == 2*(raw_max - raw_min) is what test_hounsfield_nifti.py asserts: the non-unit scl_slope
makes the HU rescale observable, so a loader that skipped the rescale (or used the raw-stored min as
the offset base) would report 511 instead. This pins both the rescale and the offset-base wiring.

No third-party NIfTI dependency (nibabel) is required: the 348-byte NIfTI-1 header is written by hand.
"""
import os
import struct
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NX = NY = NZ = 8
N = NX * NY * NZ


def write_nifti1(path, data, datatype, bitpix, scl_slope=1.0, scl_inter=0.0):
    """Write a minimal, valid single-file NIfTI-1 (.nii) image."""
    hdr = bytearray(352)
    struct.pack_into("<i", hdr, 0, 348)                       # sizeof_hdr
    struct.pack_into("<8h", hdr, 40, 3, NX, NY, NZ, 1, 1, 1, 1)  # dim[8]
    struct.pack_into("<h", hdr, 70, datatype)                 # datatype
    struct.pack_into("<h", hdr, 72, bitpix)                   # bitpix
    struct.pack_into("<8f", hdr, 76, 1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)  # pixdim[8]
    struct.pack_into("<f", hdr, 108, 352.0)                   # vox_offset
    struct.pack_into("<f", hdr, 112, float(scl_slope))        # scl_slope
    struct.pack_into("<f", hdr, 116, float(scl_inter))        # scl_inter
    struct.pack_into("<4s", hdr, 344, b"n+1\0")               # magic (single-file)
    with open(path, "wb") as fh:
        fh.write(hdr)
        fh.write(np.ascontiguousarray(data).tobytes())


def main():
    idx = np.arange(N, dtype=np.int64)
    inten = (idx - 200).astype("<i2")                         # -200..311, crosses zero
    write_nifti1(os.path.join(HERE, "ct3d_int16.nii"), inten, datatype=4, bitpix=16,
                 scl_slope=2.0, scl_inter=-1024.0)

    mask = np.ones(N, dtype="<u1")                            # single ROI over the whole volume
    write_nifti1(os.path.join(HERE, "mask3d.nii"), mask, datatype=2, bitpix=8)

    print("wrote ct3d_int16.nii and mask3d.nii to", HERE)


if __name__ == "__main__":
    main()
