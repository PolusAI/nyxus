#pragma once

#include "dcmtk/config/osconfig.h"
#include "dcmtk/dcmdata/dctk.h"
#include "dcmtk/dcmdata/dcpixel.h"
#include "dcmtk/dcmdata/dcdeftag.h"

bool isUncompressedXfer(E_TransferSyntax xfer);
Uint32 getFrameSize(DcmDataset* dataset);