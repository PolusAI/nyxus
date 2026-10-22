"""bench_volmin_3d, the 3D texture fixture whose ROI holds the volume's lowest intensity, and its control.

    python tests/vetting/oracles/volmin_phantom.py           (check the checked-in NIfTI files)
    python tests/vetting/oracles/volmin_phantom.py --write   (write them)

An 8x14x14 int16 volume, CT-like: an ellipsoid ROI (label 1) whose bounding box has background in
its corners, at intensities -1024 + 25 ((3x + 5y + 7z + xyz) mod 17), i.e. 17 levels over
[-1024, -624]; every other voxel at -600 + ((x + 2y + 3z) mod 50). So the volume's lowest intensity,
-1024, is carried by ROI voxels.

Nyxus stores a volume's intensities offset by its lowest one, so those ROI voxels are stored as 0,
the value the bounding box's background cells hold too. What a 3D texture family must do: bin them
like any other ROI voxel -- under N PyRadiomics-style bins the ROI's lowest level is bin 1 -- and
leave the background out. PyRadiomics and MIRP read the ROI's own intensities, so they are the judge.

The control, bench_volmin_ctrl_3d, is the same volume with one voxel outside the ROI's bounding box,
(z, y, x) = (0, 0, 0), at -2000: the volume's lowest intensity is then not the ROI's, no ROI voxel is
stored as 0, and the ROI's intensities -- hence every oracle value -- are unchanged. A family must
give the same values on both.

At a bin count of 8 the bin width is 50, so every other ROI level (n (x - min) a multiple of the range)
sits exactly on a bin edge.

numpy is the only dependency. The generators that pin oracles on this fixture load it from here:
gen_{glcm,gldm,glrlm,glszm}3d_pyradiomics.py and gen_{gldzm,ngldm}3d_mirp.py.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.dirname(os.path.dirname(HERE))
PHANTOMS = os.path.join(TESTS, "data", "nifti", "phantoms")
INTEN = os.path.join(PHANTOMS, "volmin_inten.nii")
CTRL_INTEN = os.path.join(PHANTOMS, "volmin_ctrl_inten.nii")
MASK = os.path.join(PHANTOMS, "volmin_mask.nii")

LABEL = 1
NBINS = 8
Z, Y, X = 8, 14, 14
ROI_MIN = -1024
CTRL_MIN = -2000

NIFTI_INT16 = 4
NIFTI_UINT8 = 2
NIFTI_DTYPE = {NIFTI_UINT8: np.uint8, NIFTI_INT16: np.int16}


def build():
    """-> (intensities, control intensities, mask), each shaped (z, y, x)."""
    z, y, x = np.indices((Z, Y, X))
    roi = (((z - (Z - 1) / 2) / 3.6) ** 2 + ((y - (Y - 1) / 2) / 6.2) ** 2
           + ((x - (X - 1) / 2) / 6.2) ** 2) <= 1.0
    inten = np.where(roi,
                     ROI_MIN + 25 * ((3 * x + 5 * y + 7 * z + x * y * z) % 17),
                     -600 + (x + 2 * y + 3 * z) % 50).astype(np.int16)
    ctrl = inten.copy()
    ctrl[0, 0, 0] = CTRL_MIN
    mask = np.where(roi, LABEL, 0).astype(np.uint8)
    return inten, ctrl, mask


def check_fixture(inten, ctrl, mask):
    """Raises unless the arrays have the properties the module docstring claims."""
    roi = mask == LABEL
    zz, yy, xx = np.nonzero(roi)
    box = (np.ptp(zz) + 1) * (np.ptp(yy) + 1) * (np.ptp(xx) + 1)
    if not roi.sum() < box:
        raise RuntimeError("the ROI fills its bounding box; it must leave background in it")
    if inten.min() != ROI_MIN or inten[~roi].min() <= ROI_MIN:
        raise RuntimeError("the volume's lowest intensity must be the ROI's alone")
    if (inten[roi] == ROI_MIN).sum() < 2:
        raise RuntimeError("the ROI must carry the volume's lowest intensity on several voxels")
    if roi[0, 0, 0] or (zz.min() == 0 and yy.min() == 0 and xx.min() == 0):
        raise RuntimeError("the control's moved voxel must lie outside the ROI's bounding box")
    if ctrl.min() != CTRL_MIN or not np.array_equal(ctrl[roi], inten[roi]):
        raise RuntimeError("the control must move only the volume minimum, outside the ROI")
    levels = np.unique(inten[roi])
    bins = np.minimum(np.floor(NBINS * (levels - levels.min()) / (levels.max() - levels.min())) + 1, NBINS)
    if len(np.unique(bins)) < NBINS:
        raise RuntimeError("the ROI must populate every one of the %d bins" % NBINS)


def read_nifti(path):
    """-> array shaped (z, y, x). Uncompressed single-file NIfTI-1 only."""
    with open(path, "rb") as fh:
        raw = fh.read()
    if int(np.frombuffer(raw, np.int32, 1, 0)[0]) != 348 or raw[344:347] != b"n+1":
        raise RuntimeError(path + " is not an uncompressed single-file NIfTI-1")
    dim = np.frombuffer(raw, np.int16, 8, 40)
    datatype = int(np.frombuffer(raw, np.int16, 1, 70)[0])
    vox_offset = int(np.frombuffer(raw, np.float32, 1, 108)[0])
    nx, ny, nz = int(dim[1]), int(dim[2]), int(dim[3])
    return np.frombuffer(raw, NIFTI_DTYPE[datatype], nx * ny * nz, vox_offset).reshape((nz, ny, nx))


def write_nifti(path, vol, datatype):
    """Writes 'vol' as an uncompressed single-file NIfTI-1 at 1x1x1 spacing."""
    nz, ny, nx = vol.shape
    bitpix = 8 * np.dtype(NIFTI_DTYPE[datatype]).itemsize
    hdr = bytearray(352)
    hdr[0:4] = np.array([348], np.int32).tobytes()
    hdr[40:56] = np.array([3, nx, ny, nz, 1, 1, 1, 1], np.int16).tobytes()
    hdr[70:72] = np.array([datatype], np.int16).tobytes()
    hdr[72:74] = np.array([bitpix], np.int16).tobytes()
    hdr[76:108] = np.array([0, 1, 1, 1, 1, 1, 1, 1], np.float32).tobytes()
    hdr[108:112] = np.array([352.0], np.float32).tobytes()
    hdr[148:228] = b"nyxus 3d texture volume-minimum phantom".ljust(80, b"\0")
    hdr[254:256] = np.array([1], np.int16).tobytes()               # sform_code
    hdr[280:296] = np.array([1, 0, 0, 0], np.float32).tobytes()    # srow_x
    hdr[296:312] = np.array([0, 1, 0, 0], np.float32).tobytes()    # srow_y
    hdr[312:328] = np.array([0, 0, 1, 0], np.float32).tobytes()    # srow_z
    hdr[344:348] = b"n+1\0"
    with open(path, "wb") as fh:
        fh.write(bytes(hdr))
        fh.write(vol.astype(NIFTI_DTYPE[datatype]).tobytes())


def files():
    """-> [(path, array, NIfTI datatype)] of the three checked-in files."""
    inten, ctrl, mask = build()
    return [(INTEN, inten, NIFTI_INT16), (CTRL_INTEN, ctrl, NIFTI_INT16), (MASK, mask, NIFTI_UINT8)]


def check_files():
    """-> the number of checked-in files that do not hold what build() produces (0 when all do)."""
    bad = 0
    for path, vol, _ in files():
        name = os.path.basename(path)
        if not os.path.exists(path):
            print("  FAIL %s is missing; run volmin_phantom.py --write" % name)
            bad += 1
        elif not np.array_equal(read_nifti(path), vol):
            print("  FAIL %s does not hold what volmin_phantom.build() produces" % name)
            bad += 1
        else:
            print("  OK   %s is what volmin_phantom.build() produces" % name)
    return bad


def main():
    check_fixture(*build())
    if "--write" in sys.argv[1:]:
        for path, vol, datatype in files():
            write_nifti(path, vol, datatype)
            print("# wrote " + path)
        return 0
    return 1 if check_files() else 0


if __name__ == "__main__":
    sys.exit(main())
