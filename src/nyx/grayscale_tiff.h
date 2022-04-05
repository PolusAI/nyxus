#pragma once

#include <fast_loader/fast_loader.h> 

#ifdef __APPLE__
#define uint64 uint64_hack_
#define int64 int64_hack_
#include <tiffio.h>
#undef uint64
#undef int64
#else
#include <tiffio.h>
#endif

/// @brief Tile Loader for 2D Grayscale tiff files
/// @tparam DataType AbstractView's internal type
template<class DataType>
class NyxusGrayscaleTiffTileLoader : public fl::AbstractTileLoader<fl::DefaultView<DataType>> 
{
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
public:

    /// @brief NyxusGrayscaleTiffTileLoader unique constructor
    /// @param numberThreads Number of threads associated
    /// @param filePath Path of tiff file
    NyxusGrayscaleTiffTileLoader(size_t numberThreads, std::string const& filePath)
        : fl::AbstractTileLoader<fl::DefaultView<DataType>>("NyxusGrayscaleTiffTileLoader", numberThreads, filePath) 
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
            uint16_t compression;
            TIFFGetField(tiff_, TIFFTAG_COMPRESSION, &compression);
            TIFFGetField(tiff_, TIFFTAG_IMAGEWIDTH, &(this->fullWidth_));
            TIFFGetField(tiff_, TIFFTAG_IMAGELENGTH, &(this->fullHeight_));
            TIFFGetField(tiff_, TIFFTAG_TILEWIDTH, &this->tileWidth_);
            TIFFGetField(tiff_, TIFFTAG_TILELENGTH, &this->tileHeight_);
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
            case 8:loadTile<uint8_t>(tiffTile, tile);
                break;
            case 16:loadTile<uint16_t>(tiffTile, tile);
                break;
            case 32:loadTile<size_t>(tiffTile, tile);
                break;
            case 64:loadTile<uint64_t>(tiffTile, tile);
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
            case 8:loadTile<int8_t>(tiffTile, tile);
                break;
            case 16:loadTile<int16_t>(tiffTile, tile);
                break;
            case 32:loadTile<int32_t>(tiffTile, tile);
                break;
            case 64:loadTile<int64_t>(tiffTile, tile);
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
                loadTile<float>(tiffTile, tile);
                break;
            case 64:
                loadTile<double>(tiffTile, tile);
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

    /// @brief Copy Method for the NyxusGrayscaleTiffTileLoader
    /// @return Return a copy of the current NyxusGrayscaleTiffTileLoader
    std::shared_ptr<fl::AbstractTileLoader<fl::DefaultView<DataType>>> copyTileLoader() override 
    {
        return std::make_shared<NyxusGrayscaleTiffTileLoader<DataType>>(this->numberThreads(), this->filePath());
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
    /// @brief Private function to copy and cast the values
    /// @tparam FileType Type inside the file
    /// @param src Piece of memory coming from libtiff
    /// @param dest Piece of memory to fill
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
};

/// @brief Tile Loader for 3D Grayscale tiff files encoded in strips
/// @tparam DataType AbstractView's internal type
template<class DataType>
class NyxusGrayscaleTiffStripLoader : public fl::AbstractTileLoader<fl::DefaultView<DataType>> 
{
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
public:
    /// @brief NyxusGrayscaleTiffTileLoader constructor
    /// @param numberThreads Number of threads associated
    /// @param filePath Path of tiff file
    /// @param tileWidth Tile width requested
    /// @param tileHeight Tile height requested
    /// @param tileDepth Tile depth requested
    NyxusGrayscaleTiffStripLoader(
        size_t numberThreads,
        std::string const& filePath,
        size_t tileWidth, size_t tileHeight, size_t tileDepth)
        : fl::AbstractTileLoader<fl::DefaultView<DataType>>("NyxusGrayscaleTiffStripLoader", numberThreads, filePath),
        tileWidth_(tileWidth), tileHeight_(tileHeight), tileDepth_(tileDepth) 
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
                    case 8:copyRow<uint8_t>(buf, tile, layer - startLayer, row - startRow, startCol, endCol);
                        break;
                    case 16:copyRow<uint16_t>(buf, tile, layer - startLayer, row - startRow, startCol, endCol);
                        break;
                    case 32:copyRow<size_t>(buf, tile, layer - startLayer, row - startRow, startCol, endCol);
                        break;
                    case 64:copyRow<uint64_t>(buf, tile, layer - startLayer, row - startRow, startCol, endCol);
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
                    case 8:copyRow<int8_t>(buf, tile, layer - startLayer, row - startRow, startCol, endCol);
                        break;
                    case 16:copyRow<int16_t>(buf, tile, layer - startLayer, row - startRow, startCol, endCol);
                        break;
                    case 32:copyRow<int32_t>(buf, tile, layer - startLayer, row - startRow, startCol, endCol);
                        break;
                    case 64:copyRow<int64_t>(buf, tile, layer - startLayer, row - startRow, startCol, endCol);
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
                    case 32:copyRow<float>(buf, tile, layer - startLayer, row - startRow, startCol, endCol);
                        break;
                    case 64:copyRow<double>(buf, tile, layer - startLayer, row - startRow, startCol, endCol);
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

    /// @brief Copy Method for the NyxusGrayscaleTiffTileLoader
    /// @return Return a copy of the current NyxusGrayscaleTiffTileLoader
    std::shared_ptr<fl::AbstractTileLoader<fl::DefaultView<DataType>>> copyTileLoader() override 
    {
        return std::make_shared<NyxusGrayscaleTiffStripLoader<DataType>>(this->numberThreads(),
            this->filePath(),
            this->tileWidth_,
            this->tileHeight_,
            this->tileDepth_);
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
        size_t endCol) {
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
};
