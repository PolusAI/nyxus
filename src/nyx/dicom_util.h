#pragma once

#include "dcmtk/config/osconfig.h"
#include "dcmtk/dcmdata/dctk.h"
#include "dcmtk/dcmdata/dcpixel.h"
#include "dcmtk/dcmdata/dcdeftag.h"

// Helper: Determine if the transfer syntax is uncompressed
bool isUncompressedXfer(E_TransferSyntax xfer)
{
    return xfer == EXS_LittleEndianExplicit ||
           xfer == EXS_LittleEndianImplicit ||
           xfer == EXS_BigEndianExplicit;
}

Uint32 getFrameSize(DcmDataset* dataset)
{
    DcmElement* elem = nullptr;
    if (dataset->findAndGetElement(DCM_PixelData, elem).bad() || !elem)
        return 0;

    DcmPixelData* pixelData = dynamic_cast<DcmPixelData*>(elem);
    if (!pixelData)
        return 0;

    Uint32 frameSize = 0;
    OFCondition status;

    E_TransferSyntax xfer = dataset->getOriginalXfer();
    bool isUncompressed = isUncompressedXfer(xfer);


    status = pixelData->getUncompressedFrameSize(dataset, frameSize, isUncompressed);

    return status.good() ? frameSize : 0;
}
