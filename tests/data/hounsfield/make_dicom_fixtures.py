"""Generate tiny synthetic CT DICOM fixtures for the Hounsfield-Unit loader tests.

Both are 16x16, single frame, MONOCHROME2, ExplicitVRLittleEndian, and encode the
SAME logical HU field as the TIFF fixtures: HU(r,c) = -1024 + idx*8, idx = r*16 + c
(HU span -1024..1016, crossing 0 = water).

  ct_u16.dcm : unsigned uint16 (PixelRepresentation=0), RescaleIntercept=-1024,
               RescaleSlope=1  -> stored = HU + 1024 = idx*8   (textbook CT encoding)
  ct_i16.dcm : signed int16   (PixelRepresentation=1), RescaleIntercept=0,
               RescaleSlope=1  -> stored = HU = -1024 + idx*8  (negative stored values)

In HU mode with the scanned HU min = -1024, the loader maps HU -> HU - floor(-1024)
= HU + 1024 = idx*8, so the feature-domain pixel at idx equals idx*8 (0..2040) for
BOTH fixtures — no wraparound. The C++ test recomputes this.

Run:  python make_dicom_fixtures.py    (needs numpy + pydicom)
"""
import numpy as np, pathlib
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid, SecondaryCaptureImageStorage

HERE = pathlib.Path(__file__).parent
N = 16
idx = np.arange(N * N).reshape(N, N)
hu = (-1024 + idx * 8)   # -1024..1016


def write(name, stored_arr, pixel_repr, intercept, dtype):
    fm = FileMetaDataset()
    fm.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    fm.MediaStorageSOPInstanceUID = generate_uid()
    fm.TransferSyntaxUID = ExplicitVRLittleEndian
    fm.ImplementationClassUID = generate_uid()

    ds = Dataset()
    ds.file_meta = fm
    ds.SOPClassUID = SecondaryCaptureImageStorage
    ds.SOPInstanceUID = fm.MediaStorageSOPInstanceUID
    ds.Modality = "CT"
    ds.Rows, ds.Columns = N, N
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = pixel_repr
    ds.RescaleSlope = 1
    ds.RescaleIntercept = intercept
    ds.PixelData = stored_arr.astype(dtype).tobytes()
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    ds.save_as(HERE / name, write_like_original=False)
    print("wrote", name, "pixrep", pixel_repr, "intercept", intercept,
          "stored[min,max]", int(stored_arr.min()), int(stored_arr.max()))


# unsigned: stored = HU + 1024 = idx*8  (0..2040)
write("ct_u16.dcm", hu + 1024, pixel_repr=0, intercept=-1024, dtype=np.uint16)
# signed: stored = HU = -1024 + idx*8  (-1024..1016)
write("ct_i16.dcm", hu, pixel_repr=1, intercept=0, dtype=np.int16)
