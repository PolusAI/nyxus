#pragma once

#ifdef DICOM_SUPPORT
#include "dcmtk/dcmseg/segdoc.h"

// Binary-segmentation frame access that builds against both DCMTK 3.6.x and 3.7+.
// DCMTK 3.7 turned DcmIODTypes::Frame (public 'pixData' and 'length') into a class template
// over the pixel type, and DcmSegmentation::getFrame() now returns the type-erased FrameBase.
namespace Nyxus
{
#if PACKAGE_VERSION_NUMBER >= 370
    using DcmBinaryFrame = DcmIODTypes::Frame<Uint8>;

    // binary segmentation frames are stored bit-packed as 8-bit pixels; nullptr if the frame isn't
    inline const DcmBinaryFrame* get_binary_seg_frame (DcmSegmentation* segdoc, size_t frame_no)
    {
        return dynamic_cast<const DcmBinaryFrame*> (segdoc->getFrame(frame_no));
    }

    inline size_t frame_length (const DcmBinaryFrame* f) { return f->getLengthInBytes(); }
    inline const Uint8* frame_pixels (const DcmBinaryFrame* f) { return f->getPixelDataTyped(); }
#else
    using DcmBinaryFrame = DcmIODTypes::Frame;

    inline const DcmBinaryFrame* get_binary_seg_frame (DcmSegmentation* segdoc, size_t frame_no)
    {
        return segdoc->getFrame(frame_no);
    }

    inline size_t frame_length (const DcmBinaryFrame* f) { return f->length; }
    inline const Uint8* frame_pixels (const DcmBinaryFrame* f) { return f->pixData; }
#endif
}

#endif // DICOM_SUPPORT
