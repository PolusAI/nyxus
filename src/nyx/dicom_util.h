#include "dcmtk/config/osconfig.h"    // For DCMTK_VERSION_NUMBER and config macros
#include "dcmtk/dcmdata/dctk.h"       // Core DICOM toolkit (includes DcmDataset, DcmPixelData, etc.)
#include "dcmtk/dcmdata/dcpixel.h"    // For DcmPixelData
#include "dcmtk/dcmdata/dcdeftag.h"   // For DCM_PixelData tag

Uint32 getFrameSize(DcmDataset* dataset)
{
    DcmPixelData* pixelData = nullptr;
    if (dataset->findAndGetElement(DCM_PixelData, pixelData).bad() || !pixelData)
        return 0;

    Uint32 frameSize = 0;
    OFBool isUncompressed = (pixelData->transferState() == ERW_memory);
    status = pixelData->getUncompressedFrameSize(dataset, frameSize, isUncompressed);

    return status.good() ? frameSize : 0;
}