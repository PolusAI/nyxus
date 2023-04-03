#pragma once
#include "abs_tile_loader.h"
#include "dcmtk/dcmdata/dctk.h"

template<class DataType>
class NyxusGrayscaleDicomLoader : public AbstractTileLoader<DataType> 
{
public:

    /// @brief NyxusGrayscaleDicomLoader constructor
    /// @param numberThreads Number of threads associated
    /// @param filePath Path of dicom file
    NyxusGrayscaleDicomLoader(
        size_t numberThreads,
        std::string const& filePath)
        : AbstractTileLoader<DataType>("NyxusGrayscaleDicomLoader", numberThreads, filePath) 
    {
        
        dcm_ff_ = DcmFileFormat();
        auto file_ok = dcm_ff_.loadFile(filePath.c_str());
        if (file_ok.good())
        // Open the file
        {
            ds_ = dcm_ff_.getDataset();
            auto dcm_tag_spp = DcmTagKey(0x0028,0x0002);
            ds_->findAndGetUint16(dcm_tag_spp, samplesPerPixel);
            // Test if the file is grayscale
            if (samplesPerPixel != 1) 
            {          
                std::stringstream message;
                message << "Tile Loader ERROR: The file is not grayscale: SamplesPerPixel = " << samplesPerPixel << ".";
                throw (std::runtime_error(message.str()));
            }

            const char* wsi_uid = "1.2.840.10008.5.1.4.1.1.7"; // WSI UID
            auto dcm_tag_sop_class_uid = DcmTagKey(0x0008, 0x0016);
            const char* sop_class_uid = NULL;
            
            ds_->findAndGetString(dcm_tag_sop_class_uid, sop_class_uid);
            
            auto dcm_tag_frame_width = DcmTagKey(0x0028,0x0011);
            auto dcm_tag_frame_height = DcmTagKey(0x0028,0x0010);
            uint16_t frame_height, frame_width;
            ds_->findAndGetUint16(dcm_tag_frame_width, frame_width);
            ds_->findAndGetUint16(dcm_tag_frame_height, frame_height);
            auto dcm_tag_total_width = DcmTagKey(0x0048,0x0006) ;
            auto dcm_tag_total_height = DcmTagKey(0x0048,0x0007) ;
            
            tileHeight_ = frame_height;
            tileWidth_ = frame_width;


            unsigned int total_height = 0, total_width = 0;
            OFCondition status = ds_->findAndGetUint32(dcm_tag_total_height, total_height);
            if (status.bad() | total_height == 0) fullHeight_ = tileHeight_;
            status = ds_->findAndGetUint32(dcm_tag_total_width, total_width);
            if (status.bad() | total_width == 0) fullWidth_ = tileWidth_;

            numCols_ = static_cast<size_t>(ceil(fullWidth_/tileWidth_));
            numRows_ = static_cast<size_t>(ceil(fullHeight_/tileHeight_));
            fullDepth_ = 1;
            tileDepth_ = 1;


            auto dcm_tag_num_frames = DcmTagKey(0x0028,0x0008);
            int32_t num_frames = 0;
            status = ds_->findAndGetSint32(dcm_tag_num_frames, numFrames_);
            if (status.bad() | numFrames_ == 0) numFrames_ = numCols_*numRows_;

            auto dcm_tag_bits_allocated = DcmTagKey(0x0028,0x0100);
            ds_->findAndGetUint16(dcm_tag_bits_allocated, bitsPerSample_);
        }
        else 
        { 
            throw (std::runtime_error("Tile Loader ERROR: The file can not be opened.")); 
        }
    }

    /// @brief NyxusGrayscaleDicomLoader destructor
    ~NyxusGrayscaleDicomLoader() override 
    {
        ds_ = nullptr;
        dcm_ff_.clear();
    }

    /// @brief Load a tiff tile from a view
    /// @param tile Tile to copy into
    /// @param indexRowGlobalTile Tile row index
    /// @param indexColGlobalTile Tile column index
    /// @param indexLayerGlobalTile Tile layer index
    /// @param level Tile's level
    void loadTileFromFile(std::shared_ptr<std::vector<DataType>> tile,
        size_t indexRowGlobalTile,
        size_t indexColGlobalTile,
        size_t indexLayerGlobalTile,
        [[maybe_unused]] size_t level) override 
    {
        // Get ahold of the logical (feature extraction facing) tile buffer from its smart pointer
        std::vector<DataType>& tileDataVec = *tile;
        uint32_t frame_no = indexRowGlobalTile*numCols_ + indexColGlobalTile;
        if (frame_no >= numFrames_){
            std::stringstream message;
            message
                << "Tile Loader ERROR: The requested tile ("
                << indexRowGlobalTile <<", "
                << indexColGlobalTile << ") could not be found.";
            throw (std::runtime_error(message.str()));
        }

        switch(bitsPerSample_){
            case 8:
                copyFrame<uint8_t>(*tile, frame_no);
                break;
            case 16:
                copyFrame<uint16_t>(*tile, frame_no);
                break;
            default:
                std::stringstream message;
                message
                    << "Tile Loader ERROR: The data format is not supported for unsigned integer, number bits per pixel = "
                    << bitsPerSample_;
                throw (std::runtime_error(message.str()));
        }
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
    [[nodiscard]] short bitsPerSample() const override { return static_cast<short>(bitsPerSample_); }
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
        std::vector<DataType>& dest_as_vector,
        uint32_t frame_no) 
    {   
        size_t data_length = tileHeight_ * tileWidth_;
        if(dest_as_vector.size() < data_length){
            std::stringstream message;
            message
                << "Tile Loader ERROR: The destination buffer size ("
                <<dest_as_vector.size()
                <<") is smaller than the frame size ("
                << data_length <<").";
            throw (std::runtime_error(message.str()));
        }
        DcmElement* pixel_data_ptr = nullptr;
        auto stat = ds_->findAndGetElement(DCM_PixelData, pixel_data_ptr);
        DcmPixelData* pixel_data = OFreinterpret_cast(DcmPixelData*, pixel_data_ptr);
        uint32_t frame_size = 0;
        pixel_data->getUncompressedFrameSize(ds_,frame_size);
        frame_size % 2 == 0 ? frame_size = frame_size : frame_size = frame_size + 1; // need to be even
        


        auto buffer = std::vector<FileType>(data_length);
        uint32_t start_fragment = 0;
        OFString decompressed_color_model;
        auto status = pixel_data->getUncompressedFrame(ds_, 
                                            frame_no,  
                                            start_fragment, 
                                            buffer.data(),
                                            frame_size,
                                            decompressed_color_model, NULL);
        std::cout << status.text() << std::endl;

        if(status.good()){
        // Get ahold of the raw pointer
            DataType* dest = dest_as_vector.data();
            for (size_t i=0; i<data_length; i++){
                *(dest+i) = static_cast<DataType>(buffer[i]);
            }
        } else {
            std::stringstream message;
            message
                << "Tile Loader ERROR: The requested frame ("
                << frame_no <<") could not be retrieved.";
            throw (std::runtime_error(message.str()));
        }
  
    }

    DcmDataset*
        ds_ = nullptr;             ///< Tiff file pointer

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

    uint16_t samplesPerPixel = 0, bitsPerSample_ = 0;
    int32_t numFrames_ = 0;
};
