#pragma once

#ifdef DICOM_SUPPORT
#include "dcmtk/dcmdata/dctk.h"
#include "dcmtk/dcmjpeg/djdecode.h"  /* for JPEG decoders */
#include "dcmtk/dcmjpls/djdecode.h"  /* for JPEG-LS decoders */
#include "dcmtk/dcmdata/dcrledrg.h"  /* for RLE decoder */
#include "dcmtk/dcmjpeg/dipijpeg.h"  /* for dcmimage JPEG plugin */
#include "dcmtk/dcmseg/segdoc.h"
#include "dcmtk/dcmseg/segment.h"
#include "dcmtk/dcmseg/segutils.h"
#ifdef JPEG2K_SUPPORT
#include "fmjpeg2k/djdecode.h"
#endif

#include "raw_format.h"

class RawDicomLoader : public RawFormatLoader
{
public:

    /// @brief NyxusGrayscaleDicomLoader constructor
    /// @param numberThreads Number of threads associated
    /// @param filePath Path of dicom file
    RawDicomLoader (std::string const& filePath) : RawFormatLoader("RawDicomLoader", filePath)
    {
        // register JPEG decoder
        DJDecoderRegistration::registerCodecs();
        // register JPEG-LS decoder
        DJLSDecoderRegistration::registerCodecs();
        // register RLE decoder
        DcmRLEDecoderRegistration::registerCodecs();
#ifdef JPEG2K_SUPPORT
        // JPEG2K decoder
        FMJPEG2KDecoderRegistration::registerCodecs();
#endif
        dcm_ff_ = DcmFileFormat();
        auto file_ok = dcm_ff_.loadFile(filePath.c_str());
        if (file_ok.good())
            // Open the file
        {
            DcmDataset* ds = dcm_ff_.getDataset();
            uint16_t tmp = 0;
            ds->findAndGetUint16(DCM_SamplesPerPixel, tmp);
            samplesPerPixel_ = static_cast<short>(tmp); // OK to downcast
            // Test if the file is grayscale
            if (samplesPerPixel_ != 1)
            {
                std::stringstream message;
                message << "Tile Loader ERROR: The file is not grayscale: SamplesPerPixel = " << samplesPerPixel_ << ".";
                throw (std::runtime_error(message.str()));
            }

            tmp = 0;
            ds->findAndGetUint16(DCM_Columns, tmp);
            tileWidth_ = tmp;

            tmp = 0;
            ds->findAndGetUint16(DCM_Rows, tmp);
            tileHeight_ = tmp;

            // for WSI, total height and widths are different from
            // tile height and width

            unsigned int total_height = 0, total_width = 0;
            OFCondition status = ds->findAndGetUint32(DCM_TotalPixelMatrixRows, total_height);
            if (status.bad() || total_height == 0) 
                fullHeight_ = tileHeight_;
            status = ds->findAndGetUint32(DCM_TotalPixelMatrixColumns, total_width);
            if (status.bad() || total_width == 0) 
                fullWidth_ = tileWidth_;

            numCols_ = static_cast<size_t>(ceil(fullWidth_ / tileWidth_));
            numRows_ = static_cast<size_t>(ceil(fullHeight_ / tileHeight_));

            // our images have single Z plane. Whole Slide images will have multiple frames 
            // but still single Z plane
            fullDepth_ = 1;
            tileDepth_ = 1;

            // this comes into play for WSI and also for sanity check
            int32_t num_frames = 0;
            status = ds->findAndGetSint32(DCM_NumberOfFrames, numFrames_);
            if (status.bad() || numFrames_ == 0) 
                numFrames_ = numCols_ * numRows_;

            tmp = 0;
            ds->findAndGetUint16(DCM_BitsAllocated, tmp);
            bitsPerSample_ = static_cast<short>(tmp); // OK to downcast

            tmp = 0;
            ds->findAndGetUint16(DCM_PixelRepresentation, tmp);
            if (tmp == 0) {
                isSigned_ = false;
            }
            else if (tmp == 1) {
                isSigned_ = true;
            }
        }
        else
        {
            throw (std::runtime_error("Tile Loader ERROR: The file can not be opened."));
        }
    }

    /// @brief NyxusGrayscaleDicomLoader destructor
    ~RawDicomLoader() override
    {
        dcm_ff_.clear();
        DJDecoderRegistration::cleanup();
        DJLSDecoderRegistration::cleanup();
        DcmRLEDecoderRegistration::cleanup();
#ifdef JPEG2K_SUPPORT
        FMJPEG2KDecoderRegistration::cleanup();
#endif       
    }

    /// @brief Load a tiff tile from a view
    /// @param tile Tile to copy into
    /// @param indexRowGlobalTile Tile row index
    /// @param indexColGlobalTile Tile column index
    /// @param indexLayerGlobalTile Tile layer index
    /// @param level Tile's level
    void loadTileFromFile(
        size_t indexRowGlobalTile,
        size_t indexColGlobalTile,
        size_t indexLayerGlobalTile,
        [[maybe_unused]] size_t level) override
    {
        tile.resize (tileWidth_ * tileHeight_);
        std::fill (tile.begin(), tile.end(), 0);

        std::vector<uint32_t>& tileDataVec = tile;
        uint32_t frame_no = indexRowGlobalTile * numCols_ + indexColGlobalTile;
        if (frame_no >= numFrames_) {
            std::stringstream message;
            message
                << "Tile Loader ERROR: The requested tile ("
                << indexRowGlobalTile << ", "
                << indexColGlobalTile << ") could not be found.";
            throw (std::runtime_error(message.str()));
        }
        if (isSigned_) {
            switch (bitsPerSample_) {
            case 8:
                copyFrame<int8_t>(tile, frame_no);
                break;
            case 16:
                copyFrame<int16_t>(tile, frame_no);
                break;
            default:
                std::stringstream message;
                message
                    << "Tile Loader ERROR: The data format is not supported for signed integer, number bits per pixel = "
                    << bitsPerSample_;
                throw (std::runtime_error(message.str()));
            }
        }
        else {
            switch (bitsPerSample_) {
            case 1:
                copyBinaryFrame(tile, frame_no);
                break;
            case 8:
                copyFrame<uint8_t>(tile, frame_no);
                break;
            case 16:
                copyFrame<uint16_t>(tile, frame_no);
                break;
            default:
                std::stringstream message;
                message
                    << "Tile Loader ERROR: The data format is not supported for unsigned integer, number bits per pixel = "
                    << bitsPerSample_;
                throw (std::runtime_error(message.str()));
            }
        }

    }

    void free_tile () override
    {
    }

    uint32_t get_uint32_pixel(size_t idx) const
    {
        uint32_t rv = tile [idx];
        return rv;
    }

    double get_dpequiv_pixel (size_t idx) const
    {
        double rv = (double) tile [idx];
        return rv;
    }

    /// @brief Tiff file height
    /// @param level Tiff level [not used]
    /// @return Full height
    [[nodiscard]] size_t fullHeight([[maybe_unused]] size_t level) const override { return fullHeight_; }
    /// @brief Tiff full width
    /// @param level Tiff level [not used]
    /// @return Full width
    [[nodiscard]] size_t fullWidth([[maybe_unused]] size_t level) const override { return fullWidth_; }
    /// @brief Tiff full depth
    /// @param level Tiff level [not used]
    /// @return Full Depth
    [[nodiscard]] size_t fullDepth([[maybe_unused]] size_t level) const override { return fullDepth_; }

    /// @brief Tiff tile width
    /// @param level Tiff level [not used]
    /// @return Tile width
    [[nodiscard]] size_t tileWidth([[maybe_unused]] size_t level) const override { return tileWidth_; }
    /// @brief Tiff tile height
    /// @param level Tiff level [not used]
    /// @return Tile height
    [[nodiscard]] size_t tileHeight([[maybe_unused]] size_t level) const override { return tileHeight_; }
    /// @brief Tiff tile depth
    /// @param level Tiff level [not used]
    /// @return Tile depth
    [[nodiscard]] size_t tileDepth([[maybe_unused]] size_t level) const override { return tileDepth_; }

    /// @brief Tiff bits per sample
    /// @return Size of a sample in bits
    [[nodiscard]] short bitsPerSample() const override { return bitsPerSample_; }
    /// @brief Level accessor
    /// @return 1
    [[nodiscard]] size_t numberPyramidLevels() const override { return 1; }

private:

    /// @brief Private function to copy and cast the values
    /// @tparam FileType Type inside the file
    /// @param dest_as_vector Feature extraction facing buffer to fill
    /// @param frame_no Frame to copy
    template<typename FileType>
    void copyFrame(
        std::vector<uint32_t>& dest_as_vector,
        uint32_t frame_no)
    {
        size_t data_length = tileHeight_ * tileWidth_;
        if (dest_as_vector.size() < data_length) {
            std::stringstream message;
            message
                << "Tile Loader ERROR: The destination buffer size ("
                << dest_as_vector.size()
                << ") is smaller than the frame size ("
                << data_length << ").";
            throw (std::runtime_error(message.str()));
        }
        DcmElement* pixel_data_ptr = nullptr;
        DcmDataset* ds = dcm_ff_.getDataset();
        auto status = ds->findAndGetElement(DCM_PixelData, pixel_data_ptr);
        if (status.good()) {
            DcmPixelData* pixel_data = OFreinterpret_cast(DcmPixelData*, pixel_data_ptr);
            uint32_t frame_size = 0;
            DcmXfer xfer(ds->getCurrentXfer());
            pixel_data->getUncompressedFrameSize(ds, frame_size, !xfer.isPixelDataCompressed());
            frame_size % 2 == 0 ? frame_size = frame_size : frame_size = frame_size + 1; // need to be even

            auto buffer = std::vector<FileType>(data_length);
            uint32_t start_fragment = 0;
            OFString decompressed_color_model;
            status = pixel_data->getUncompressedFrame(ds,
                frame_no,
                start_fragment,
                buffer.data(),
                frame_size,
                decompressed_color_model, NULL);

            if (status.good()) 
            {
                // Get ahold of the raw pointer
                uint32_t* dest = dest_as_vector.data();
                for (size_t i = 0; i < data_length; i++) {
                    *(dest + i) = static_cast<uint32_t>(buffer[i]);
                }
            }
            else {
                std::stringstream message;
                message
                    << "Tile Loader ERROR: The requested frame ("
                    << frame_no << ") could not be retrieved.";
                throw (std::runtime_error(message.str()));
            }
        }

    }

    /// @brief Private function to copy and cast the values for Binary Segmentation Image
    /// @tparam FileType Type inside the file
    /// @param dest_as_vector Feature extraction facing buffer to fill
    /// @param frame_no Frame to copy
    void copyBinaryFrame(
        std::vector<uint32_t>& dest_as_vector,
        uint32_t frame_no)
    {
        DcmDataset* ds = dcm_ff_.getDataset();
        DcmSegmentation* segdoc = nullptr;

        OFCondition status = DcmSegmentation::loadDataset(*ds, segdoc);
        if (status.good()) {
            const DcmIODTypes::Frame* frame = segdoc->getFrame(static_cast<size_t>(frame_no));
            const DcmIODTypes::Frame* unpacked_frame = nullptr;
            unpacked_frame = DcmSegUtils::unpackBinaryFrame(frame, tileHeight_, tileWidth_);
            if (unpacked_frame) {
                if (dest_as_vector.size() < unpacked_frame->length) {
                    std::stringstream message;
                    message
                        << "Tile Loader ERROR: The destination buffer size ("
                        << dest_as_vector.size()
                        << ") is smaller than the frame size ("
                        << unpacked_frame->length << ").";
                    throw (std::runtime_error(message.str()));
                }
                uint32_t* dest = dest_as_vector.data();
                for (size_t i = 0; i < unpacked_frame->length; ++i) {
                    *(dest + i) = static_cast<uint32_t>(unpacked_frame->pixData[i]);
                }
                delete unpacked_frame;
            }
        }

    }

    DcmFileFormat dcm_ff_;
    size_t
        fullHeight_ = 0,          ///< Full height in pixel
        fullWidth_ = 0,           ///< Full width in pixel
        fullDepth_ = 0,           ///< Full depth in pixel
        tileWidth_ = 0,           ///< Tile width
        tileHeight_ = 0,          ///< Tile height
        tileDepth_ = 0,           ///< Tile depth
        numCols_ = 0,
        numRows_ = 0;

    short samplesPerPixel_ = 0, bitsPerSample_ = 0;
    int32_t numFrames_ = 0;
    bool isSigned_ = false;

    std::vector<uint32_t> tile;
};
#endif // DICOM_SUPPORT
