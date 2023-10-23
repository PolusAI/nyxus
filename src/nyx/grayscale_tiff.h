#pragma once
#include "abs_tile_loader.h"

#ifdef __APPLE__
#define uint64 uint64_hack_
#define int64 int64_hack_
#include <tiffio.h>
#undef uint64
#undef int64
#else
#include <tiffio.h>
#endif
#include <cstring>
#include <sstream>
#include <limits.h> // for INT_MAX 

constexpr size_t STRIP_TILE_HEIGHT = 1024;
constexpr size_t STRIP_TILE_WIDTH = 1024;
constexpr size_t STRIP_TILE_DEPTH = 1;

/// @brief Tile Loader for 2D Grayscale tiff files
/// @tparam DataType AbstractView's internal type
template<class DataType>
class NyxusGrayscaleTiffTileLoader : public AbstractTileLoader<DataType> 
{
public:

    /// @brief NyxusGrayscaleTiffTileLoader unique constructor
    /// @param numberThreads Number of threads associated
    /// @param filePath Path of tiff file
    NyxusGrayscaleTiffTileLoader(size_t numberThreads, std::string const& filePath)
        : AbstractTileLoader<DataType>("NyxusGrayscaleTiffTileLoader", numberThreads, filePath) 
    {
        short samplesPerPixel = 0;

        // Open the file
        tiff_ = TIFFOpen(filePath.c_str(), "r");
        if (tiff_ != nullptr) 
        {
            if (TIFFIsTiled(tiff_) == 0) 
            { 
                throw (std::runtime_error("Tile Loader ERROR: The file is not tiled.")); 
            }
            // Load/parse header
            uint32_t temp;  // Using this variable to correctly read 'uint32_t' TIFF field values into 'size_t' variables
            uint16_t compression;
            TIFFGetField(tiff_, TIFFTAG_COMPRESSION, &compression);
            TIFFGetField(tiff_, TIFFTAG_IMAGEWIDTH, &temp);
            this->fullWidth_ = temp;
            TIFFGetField(tiff_, TIFFTAG_IMAGELENGTH, &temp);
            this->fullHeight_ = temp;
            TIFFGetField(tiff_, TIFFTAG_TILEWIDTH, &temp);
            this->tileWidth_ = temp;
            TIFFGetField(tiff_, TIFFTAG_TILELENGTH, &temp);
            this->tileHeight_ = temp;
            TIFFGetField(tiff_, TIFFTAG_SAMPLESPERPIXEL, &samplesPerPixel);
            TIFFGetField(tiff_, TIFFTAG_BITSPERSAMPLE, &(this->bitsPerSample_));
            TIFFGetField(tiff_, TIFFTAG_SAMPLEFORMAT, &(this->sampleFormat_));

            // Test if the file is greyscale
            if (samplesPerPixel != 1) 
            {
                std::stringstream message;
                message << "Tile Loader ERROR: The file is not greyscale: SamplesPerPixel = " << samplesPerPixel << ".";
                throw (std::runtime_error(message.str()));
            }
            // Interpret undefined data format as unsigned integer data
            if (sampleFormat_ < 1 || sampleFormat_ > 3) 
            { 
                sampleFormat_ = 1; 
            }
        }
        else 
        { 
            throw (std::runtime_error("Tile Loader ERROR: The file can not be opened.")); 
        }
    }

    /// @brief NyxusGrayscaleTiffTileLoader destructor
    ~NyxusGrayscaleTiffTileLoader() override 
    {
        if (tiff_) 
        {
            TIFFClose(tiff_);
            tiff_ = nullptr;
        }
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
        size_t level) override
    {
        // Get ahold of the logical (feature extraction facing) tile buffer from its smart pointer
        std::vector<DataType>& tileDataVec = *tile;

        tdata_t tiffTile = nullptr;
        auto t_szb = TIFFTileSize(tiff_);
        tiffTile = _TIFFmalloc(t_szb);
        auto errcode = TIFFReadTile(tiff_, tiffTile, indexColGlobalTile * tileWidth_, indexRowGlobalTile * tileHeight_, 0, 0);
        if (errcode < 0)
        {
            std::stringstream message;
            message
                << "Tile Loader ERROR: error reading tile data returning code "
                << errcode;
            throw (std::runtime_error(message.str()));
        }
        std::stringstream message;
        switch (sampleFormat_) 
        {
        case 1:
            switch (bitsPerSample_) 
            {
            case 8:
                loadTile <uint8_t> (tiffTile, tileDataVec);
                break;
            case 16:
                loadTile <uint16_t> (tiffTile, tileDataVec);    
                break;
            case 32:
                loadTile <uint32_t> (tiffTile, tileDataVec);
                break;
            case 64:
                loadTile <uint64_t> (tiffTile, tileDataVec);
                break;
            default:
                message
                    << "Tile Loader ERROR: The data format is not supported for unsigned integer, number bits per pixel = "
                    << bitsPerSample_;
                throw (std::runtime_error(message.str()));
            }
            break;
        case 2:
            switch (bitsPerSample_) 
            {
            case 8:
                loadTile<int8_t>(tiffTile, tileDataVec);
                break;
            case 16:
                loadTile<int16_t>(tiffTile, tileDataVec);
                break;
            case 32:
                loadTile<int32_t>(tiffTile, tileDataVec);
                break;
            case 64:
                loadTile<int64_t>(tiffTile, tileDataVec);
                break;
            default:
                message
                    << "Tile Loader ERROR: The data format is not supported for signed integer, number bits per pixel = "
                    << bitsPerSample_;
                throw (std::runtime_error(message.str()));
            }
            break;
        case 3:
            switch (bitsPerSample_) 
            {
            case 8:
            case 16:
            case 32:
                loadTile_real_intens <float> (tiffTile, tileDataVec);
                break;
            case 64:
                loadTile_real_intens <double> (tiffTile, tileDataVec);
                break;
            default:
                message
                    << "Tile Loader ERROR: The data format is not supported for float, number bits per pixel = "
                    << bitsPerSample_;
                throw (std::runtime_error(message.str()));
            }
            break;
        default:
            message << "Tile Loader ERROR: The data format is not supported, sample format = " << sampleFormat_;
            throw (std::runtime_error(message.str()));
        }

        _TIFFfree(tiffTile);
    }


    /// @brief Tiff file height
    /// @param level Tiff level [not used]
    /// @return Full height
    [[nodiscard]] size_t fullHeight([[maybe_unused]] size_t level) const override { return fullHeight_; }
    /// @brief Tiff full width
    /// @param level Tiff level [not used]
    /// @return Full width
    [[nodiscard]] size_t fullWidth([[maybe_unused]] size_t level) const override { return fullWidth_; }
    /// @brief Tiff tile width
    /// @param level Tiff level [not used]
    /// @return Tile width
    [[nodiscard]] size_t tileWidth([[maybe_unused]] size_t level) const override { return tileWidth_; }
    /// @brief Tiff tile height
    /// @param level Tiff level [not used]
    /// @return Tile height
    [[nodiscard]] size_t tileHeight([[maybe_unused]] size_t level) const override { return tileHeight_; }
    /// @brief Tiff bits per sample
    /// @return Size of a sample in bits
    [[nodiscard]] short bitsPerSample() const override { return bitsPerSample_; }
    /// @brief Level accessor
    /// @return 1
    [[nodiscard]] size_t numberPyramidLevels() const override { return 1; }

private:

    #if 0   // A faster implementation is available. Keeping this for records.
    /// @brief Private function to copy and cast the values
    /// @tparam FileType Type inside the file
    /// @param src Piece of memory coming from libtiff
    /// @param dest Feature extraction facing logical buffer to fill
    /// 
    template<typename FileType>
    void loadTile(tdata_t src, std::shared_ptr<std::vector<DataType>>& dest)
    {
        for (size_t i = 0; i < tileHeight_ * tileWidth_; ++i)
        {
            // Logic to prevent "noise" in images whose dimensions are smaller than the default tile buffer size 1024x1024
            auto row = i / tileWidth_,
                col = i % tileHeight_;
            if (col < fullWidth_ && row < fullHeight_)
                dest->data()[i] = (DataType)((FileType*)(src))[i];
            else
                dest->data()[i] = (DataType)0;  // Zero-fill gaps
        }
    }
    #endif

    /// @brief Private function to copy and cast the values
    /// @tparam FileType Type inside the file
    /// @param src Piece of memory coming from libtiff
    /// @param dst_as_vector Feature extraction facing logical buffer to fill
    /// 
    template<typename FileType>
    void loadTile(tdata_t src, std::vector<DataType>& dst_as_vector)
    {
        // Get ahold of the raw pointer
        DataType* dest = dst_as_vector.data();

        // Special case of tileWidth_ (e.g. 1024) > fullWidth_ (e.g. 256)
        if (tileWidth_ > fullWidth_ && tileHeight_ > fullHeight_)
        {
            // Zero-prefill margins of the logical buffer 
            size_t szb = tileHeight_ * tileWidth_ * sizeof(*dest); 
            memset(dest, 0, szb);

            // Copy pixels assuming the row-major layout both in the physical (TIFF) and logical (ROI scanner facing) buffers
            for (size_t r = 0; r < fullHeight_; r++)
                for (size_t c = 0; c < fullWidth_; c++)
                {
                    size_t logOffs = r * tileWidth_ + c,
                        physOffs = r * tileWidth_ + c;
                    *(dest + logOffs) = (DataType) *(((FileType*)src) + physOffs);
                }
        }
        else
            // General case the logical buffer is same size (specifically, tile size) as the physical one even if tileWidth_ (e.g. 1024) < fullWidth_ (e.g. 1080)
            {
                size_t n = tileHeight_ * tileWidth_;
                for (size_t i = 0; i < n; i++)
                    *(dest + i) = (DataType) *(((FileType*)src) + i);
            }
    }

    /// @brief Private function to copy and cast values to a real data type (float or double determined by parameter 'FileType'). It solves the issue when intensities in range [0.0 , 1.0] are cast to integer 0.
    /// @tparam FileType Type inside the file
    /// @param src Piece of memory coming from libtiff
    /// @param dst_as_vector [OUTPUT] Feature extraction facing logical buffer, usually of type unsigned 32-bit int
    /// 
    template<typename FileType>
    void loadTile_real_intens (tdata_t src, std::vector<DataType>& dst_as_vector)
    {
        // Get ahold of the raw pointer
        DataType* dest = dst_as_vector.data();

        // Special case of tileWidth_ (e.g. 1024) > fullWidth_ (e.g. 256)
        if (tileWidth_ > fullWidth_ && tileHeight_ > fullHeight_)
        {
            // Zero-prefill margins of the logical buffer 
            size_t szb = tileHeight_ * tileWidth_ * sizeof(*dest);
            memset(dest, 0, szb);

            // Copy pixels assuming the row-major layout both in the physical (TIFF) and logical (ROI scanner facing) buffers
            for (size_t r = 0; r < fullHeight_; r++)
                for (size_t c = 0; c < fullWidth_; c++)
                {
                    size_t logOffs = r * tileWidth_ + c,
                        physOffs = r * tileWidth_ + c;

                    // Prevent real-valued intensities smaller than 1.0 from being cast to integer 0
                    auto tmp1 = * (((FileType*)src) + physOffs);    // real-valued intensity e.g. 0.0724
                    auto tmp2 = (DataType) (tmp1 * float(INT_MAX)); // integer-valued intensity
                    *(dest + logOffs) = tmp2;
                }
        }
        else
            // General case the logical buffer is same size (specifically, tile size) as the physical one even if tileWidth_ (e.g. 1024) < fullWidth_ (e.g. 1080)
        {
            size_t n = tileHeight_ * tileWidth_;
            for (size_t i = 0; i < n; i++)
            {
                // Prevent real-valued intensities smaller than 1.0 from being cast to integer 0
                auto tmp1 = * (((FileType*)src) + i);           // real-valued intensity e.g. 0.0724
                auto tmp2 = (DataType) (tmp1 + float(INT_MAX)); // integer-valued intensity
                *(dest + i) = tmp2;
            }
        }
    }

    TIFF*
        tiff_ = nullptr;             ///< Tiff file pointer

    size_t
        fullHeight_ = 0,           ///< Full height in pixel
        fullWidth_ = 0,            ///< Full width in pixel
        tileHeight_ = 0,            ///< Tile height
        tileWidth_ = 0;             ///< Tile width

    short
        sampleFormat_ = 0,          ///< Sample format as defined by libtiff
        bitsPerSample_ = 0;         ///< Bit Per Sample as defined by libtiff

};

/// @brief Tile Loader for 3D Grayscale tiff files encoded in strips
/// @tparam DataType AbstractView's internal type
template<class DataType>
class NyxusGrayscaleTiffStripLoader : public AbstractTileLoader<DataType> 
{
public:

    /// @brief NyxusGrayscaleTiffTileLoader constructor
    /// @param numberThreads Number of threads associated
    /// @param filePath Path of tiff file
    NyxusGrayscaleTiffStripLoader(
        size_t numberThreads,
        std::string const& filePath)
        : AbstractTileLoader<DataType>("NyxusGrayscaleTiffStripLoader", numberThreads, filePath) 
    {
        short samplesPerPixel = 0;

        // Open the file
        tiff_ = TIFFOpen(filePath.c_str(), "r");
        if (tiff_ != nullptr) 
        {
            // Load/parse header
            TIFFGetField(tiff_, TIFFTAG_IMAGEWIDTH, &(this->fullWidth_));
            TIFFGetField(tiff_, TIFFTAG_IMAGELENGTH, &(this->fullHeight_));
            TIFFGetField(tiff_, TIFFTAG_SAMPLESPERPIXEL, &samplesPerPixel);
            TIFFGetField(tiff_, TIFFTAG_BITSPERSAMPLE, &(this->bitsPerSample_));
            TIFFGetField(tiff_, TIFFTAG_SAMPLEFORMAT, &(this->sampleFormat_));

            fullDepth_ = TIFFNumberOfDirectories(tiff_);

            tileWidth_ = std::min(fullWidth_, STRIP_TILE_WIDTH);
            tileHeight_ = std::min(fullHeight_, STRIP_TILE_HEIGHT);
            tileDepth_ = std::min(fullDepth_, STRIP_TILE_DEPTH);

            // Test if the file is grayscale
            if (samplesPerPixel > 1) 
            { 
                // Sometimes we have images whose samplesPerPixel==0:  if (samplesPerPixel != 1) {
                std::stringstream message;
                message << "Tile Loader ERROR: The file is not grayscale: SamplesPerPixel = " << samplesPerPixel << ".";
                throw (std::runtime_error(message.str()));
            }
            // Interpret undefined data format as unsigned integer data
            if (sampleFormat_ < 1 || sampleFormat_ > 3) 
            {
                sampleFormat_ = 1;
            }
        }
        else 
        { 
            throw (std::runtime_error("Tile Loader ERROR: The file can not be opened.")); 
        }
    }

    /// @brief NyxusGrayscaleTiffTileLoader destructor
    ~NyxusGrayscaleTiffStripLoader() override 
    {
        if (tiff_) 
        {
            TIFFClose(tiff_);
            tiff_ = nullptr;
        }
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

        tdata_t buf;
        uint32_t row, layer;

        buf = _TIFFmalloc(TIFFScanlineSize(tiff_));

        size_t
            startLayer = indexLayerGlobalTile * tileDepth_,
            endLayer = std::min((indexLayerGlobalTile + 1) * tileDepth_, fullDepth_),
            startRow = indexRowGlobalTile * tileHeight_,
            endRow = std::min((indexRowGlobalTile + 1) * tileHeight_, fullHeight_),
            startCol = indexColGlobalTile * tileWidth_,
            endCol = std::min((indexColGlobalTile + 1) * tileWidth_, fullWidth_);

        for (layer = startLayer; layer < endLayer; ++layer) 
        {
            TIFFSetDirectory(tiff_, layer);
            for (row = startRow; row < endRow; row++) 
            {
                TIFFReadScanline(tiff_, buf, row);
                std::stringstream message;
                switch (sampleFormat_) 
                {
                case 1:
                    switch (bitsPerSample_) 
                    {
                    case 8:copyRow<uint8_t>(buf, tileDataVec, layer - startLayer, row - startRow, startCol, endCol);
                        break;
                    case 16:copyRow<uint16_t>(buf, tileDataVec, layer - startLayer, row - startRow, startCol, endCol);
                        break;
                    case 32:copyRow<size_t>(buf, tileDataVec, layer - startLayer, row - startRow, startCol, endCol);
                        break;
                    case 64:copyRow<uint64_t>(buf, tileDataVec, layer - startLayer, row - startRow, startCol, endCol);
                        break;
                    default:
                        message
                            << "Tile Loader ERROR: The data format is not supported for unsigned integer, number bits per pixel = "
                            << bitsPerSample_;
                        throw (std::runtime_error(message.str()));
                    }
                    break;
                case 2:
                    switch (bitsPerSample_) 
                    {
                    case 8:copyRow<int8_t>(buf, tileDataVec, layer - startLayer, row - startRow, startCol, endCol);
                        break;
                    case 16:copyRow<int16_t>(buf, tileDataVec, layer - startLayer, row - startRow, startCol, endCol);
                        break;
                    case 32:copyRow<int32_t>(buf, tileDataVec, layer - startLayer, row - startRow, startCol, endCol);
                        break;
                    case 64:copyRow<int64_t>(buf, tileDataVec, layer - startLayer, row - startRow, startCol, endCol);
                        break;
                    default:
                        message
                            << "Tile Loader ERROR: The data format is not supported for signed integer, number bits per pixel = "
                            << bitsPerSample_;
                        throw (std::runtime_error(message.str()));
                    }
                    break;
                case 3:
                    switch (bitsPerSample_) 
                    {
                    case 8:
                    case 16:
                    case 32:copyRow<float>(buf, tileDataVec, layer - startLayer, row - startRow, startCol, endCol);
                        break;
                    case 64:copyRow<double>(buf, tileDataVec, layer - startLayer, row - startRow, startCol, endCol);
                        break;
                    default:
                        message
                            << "Tile Loader ERROR: The data format is not supported for float, number bits per pixel = "
                            << bitsPerSample_;
                        throw (std::runtime_error(message.str()));
                    }
                    break;
                default:
                    message << "Tile Loader ERROR: The data format is not supported, sample format = " << sampleFormat_;
                    throw (std::runtime_error(message.str()));
                }
            }
        }
        _TIFFfree(buf);
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

    #if 0   // A faster implementation is available. Keeping this for records.
    /// @brief Private function to copy and cast the values
    /// @tparam FileType Type inside the file
    /// @param src Piece of memory coming from libtiff
    /// @param dest Piece of memory to fill
    /// @param layer Destination layer
    /// @param row Destination row
    /// @param startCol Starting column tile to copy
    /// @param endCol End column tile to copy
    template<typename FileType>
    void copyRow(tdata_t src,
        std::shared_ptr<std::vector<DataType>>& dest,
        size_t layer,
        size_t row,
        size_t startCol,
        size_t endCol) 
    {
        for (size_t col = startCol; col < endCol; col++) 
        {
            // Logic to prevent "noise" in images whose dimensions are smaller than the default tile buffer size 1024x1024
            DataType dataItem = (DataType) 0;    // Zero-fill gaps

            // - Informative zone of the strip
            if (layer < fullDepth_ && row < fullHeight_ && col < fullWidth_)
                dataItem = (DataType)((FileType*)(src))[col];
            
            // - Save the informative or zero-filled value
            dest->data()[
                tileWidth_ * tileHeight_ * layer
                    + tileWidth_ * row
                    + col - startCol] = dataItem;
        }
    }
    #endif

    /// @brief Private function to copy and cast the values
    /// @tparam FileType Type inside the file
    /// @param src Piece of memory coming from libtiff
    /// @param dest_as_vector Feature extraction facing buffer to fill
    /// @param layer Destination layer
    /// @param row Destination row
    /// @param startCol Starting column tile to copy
    /// @param endCol End column tile to copy
    template<typename FileType>
    void copyRow(tdata_t src,
        std::vector<DataType>& dest_as_vector,
        size_t layer,
        size_t row,
        size_t start_col,
        size_t end_col) 
    {
        // Get ahold of the raw pointer
        DataType* dest = dest_as_vector.data();

        for (size_t col = start_col; col < end_col; col++)
        {
            // Logic to prevent "noise" in images whose dimensions are smaller than the default tile buffer size 1024x1024
            DataType dataItem = (DataType) 0;    // Zero-fill gaps

            // - Informative zone of the strip
            if (layer < fullDepth_ && row < fullHeight_ && col < fullWidth_)
                dataItem = (DataType)((FileType*)(src))[col];
            
            // - Save the informative or zero-filled value
            dest[
                tileWidth_ * tileHeight_ * layer
                    + tileWidth_ * row
                    + col - start_col] = dataItem;
        }
    }

    TIFF*
        tiff_ = nullptr;             ///< Tiff file pointer

    size_t
        fullHeight_ = 0,          ///< Full height in pixel
        fullWidth_ = 0,           ///< Full width in pixel
        fullDepth_ = 0,           ///< Full depth in pixel
        tileWidth_ = 0,           ///< Tile width
        tileHeight_ = 0,          ///< Tile height
        tileDepth_ = 0;           ///< Tile depth

    short
        sampleFormat_ = 0,        ///< Sample format as defined by libtiff
        bitsPerSample_ = 0;       ///< Bit Per Sample as defined by libtiff

};
