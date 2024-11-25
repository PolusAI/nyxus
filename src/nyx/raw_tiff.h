#pragma once

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
#include <limits.h>

#pragma once
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "raw_format.h"

class RawTiffTileLoader : public RawFormatLoader
{
public:

    RawTiffTileLoader (std::string const& filePath): RawFormatLoader("name", filePath)
    {
        short samplesPerPixel = 0;

        // Open the file
        tiff_ = TIFFOpen (filePath.c_str(), "r");
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

            // prepare the right typed getter function
            std::string message;
            switch (sampleFormat_)
            {
            case 1:
                switch (bitsPerSample_)
                {
                case 8:
                    get_uint32_pixel_typeresolved = get_uint32_pixel_imp <uint8_t>;
                    get_dpequiv_pixel_typeresolved = get_dp_pixel_imp <uint8_t>;
                    break;
                case 16:
                    get_uint32_pixel_typeresolved = get_uint32_pixel_imp <uint16_t>;
                    get_dpequiv_pixel_typeresolved = get_dp_pixel_imp <uint16_t>;
                    break;
                case 32:
                    get_uint32_pixel_typeresolved = get_uint32_pixel_imp <uint32_t>;
                    get_dpequiv_pixel_typeresolved = get_dp_pixel_imp <uint32_t>;
                    break;
                case 64:
                    get_uint32_pixel_typeresolved = get_uint32_pixel_imp <uint64_t>;
                    get_dpequiv_pixel_typeresolved = get_dp_pixel_imp <uint64_t>;
                    break;
                default:
                    message =
                        "Tile Loader ERROR: The data format is not supported for unsigned integer, number bits per pixel = "
                        + bitsPerSample_;
                    throw (std::runtime_error(message));
                }
                break;
            case 2:
                switch (bitsPerSample_)
                {
                case 8:
                    get_uint32_pixel_typeresolved = get_uint32_pixel_imp <int8_t>;
                    get_dpequiv_pixel_typeresolved = get_dp_pixel_imp <int8_t>;
                    break;
                case 16:
                    get_uint32_pixel_typeresolved = get_uint32_pixel_imp <int16_t>;
                    get_dpequiv_pixel_typeresolved = get_dp_pixel_imp <int16_t>;
                    break;
                case 32:
                    get_uint32_pixel_typeresolved = get_uint32_pixel_imp <int32_t>;
                    get_dpequiv_pixel_typeresolved = get_dp_pixel_imp <int32_t>;
                    break;
                case 64:
                    get_uint32_pixel_typeresolved = get_uint32_pixel_imp <int64_t>;
                    get_dpequiv_pixel_typeresolved = get_dp_pixel_imp <int64_t>;
                    break;
                default:
                    message =
                        "Tile Loader ERROR: The data format is not supported for signed integer, number bits per pixel = " + std::to_string(bitsPerSample_);
                    throw (std::runtime_error(message));
                }
                break;
            case 3:
                switch (bitsPerSample_) 
                {
                case 8:
                case 16:
                case 32:
                    get_uint32_pixel_typeresolved = get_uint32_pixel_imp <float>;
                    get_dpequiv_pixel_typeresolved = get_dp_pixel_imp <float>;
                    break;
                case 64:
                    get_uint32_pixel_typeresolved = get_uint32_pixel_imp <double>;
                    get_dpequiv_pixel_typeresolved = get_dp_pixel_imp <double>;
                    break;
                default:
                    message = "Tile Loader ERROR: The data format is not supported for float, number bits per pixel = " + std::to_string(bitsPerSample_);
                    throw (std::runtime_error(message));
                }
                break;
            default:
                message = "Tile Loader ERROR: The data format is not supported, sample format = " + std::to_string(sampleFormat_);
                throw (std::runtime_error(message));
            }
        }
        else 
        { 
            throw (std::runtime_error("Tile Loader ERROR: The file can not be opened.")); 
        }
    }

    /// @brief NyxusGrayscaleTiffTileLoader destructor
    ~RawTiffTileLoader()
    {
        if (tiff_) 
        {
            TIFFClose(tiff_);
            tiff_ = nullptr;
        }
    }

    double get_max() { return maxval; }
    double get_min() { return minval; }

    void loadTileFromFile (
        size_t indexRowGlobalTile,
        size_t indexColGlobalTile,
        size_t indexLayerGlobalTile,
        size_t level) override
    {
        // Low level read TIFF bytes
        auto t_szb = TIFFTileSize(tiff_);
        tiffTile = _TIFFmalloc(t_szb);
        auto errcode = TIFFReadTile(tiff_, tiffTile, indexColGlobalTile * tileWidth_, indexRowGlobalTile * tileHeight_, 0, 0);
        if (errcode < 0)
        {
            if (errcode == -1) // the TIFF file is not tiled, don't break for each image like that
                memset (tiffTile, 0, t_szb);    
            else // something else
            {
                std::stringstream message;
                message
                    << "Tile Loader ERROR: error reading tile data returning code "
                    << errcode;
                throw (std::runtime_error(message.str()));
            }
        }
    }

    void free_tile() override
    {
        _TIFFfree(tiffTile);
    }

    uint32_t get_uint32_pixel (size_t idx) const
    {
        uint32_t rv = get_uint32_pixel_typeresolved (tiffTile, idx);
        return rv;
    }

    double get_dpequiv_pixel (size_t idx) const
    {
        double rv = get_dpequiv_pixel_typeresolved (tiffTile, idx);
        return rv;
    }

    [[nodiscard]] size_t fullHeight([[maybe_unused]] size_t level) const { return fullHeight_; }
    [[nodiscard]] size_t fullWidth([[maybe_unused]] size_t level) const { return fullWidth_; }
    [[nodiscard]] size_t tileWidth([[maybe_unused]] size_t level) const { return tileWidth_; }
    [[nodiscard]] size_t tileHeight([[maybe_unused]] size_t level) const { return tileHeight_; }
    [[nodiscard]] short bitsPerSample() const { return bitsPerSample_; }
    [[nodiscard]] size_t numberPyramidLevels() const { return 1; }

private:

    template<typename FileType>
    void scan_typed_minmax (tdata_t src)
    {
        minval = (std::numeric_limits<double>::max)();
        maxval = (std::numeric_limits<double>::min)();

        // Special case of tileWidth_ (e.g. 1024) > fullWidth_ (e.g. 256)
        if (tileWidth_ > fullWidth_ && tileHeight_ > fullHeight_)
        {
            // Copy pixels assuming the row-major layout both in the physical (TIFF) and logical (ROI scanner facing) buffers
            for (size_t r = 0; r < fullHeight_; r++)
                for (size_t c = 0; c < fullWidth_; c++)
                {
                    size_t logOffs = r * tileWidth_ + c,
                        physOffs = r * tileWidth_ + c;
                    FileType x = *(((FileType*)src) + physOffs);
                    minval = (std::min) (double(x), minval);
                    maxval = (std::max) (double(x), maxval);
                }
        }
        else
            // General case the logical buffer is same size (specifically, tile size) as the physical one even if tileWidth_ (e.g. 1024) < fullWidth_ (e.g. 1080)
            {
                size_t n = tileHeight_ * tileWidth_;
                for (size_t i = 0; i < n; i++)
                {
                    FileType x = *(((FileType*)src) + i);
                    minval = (std::min)(double(x), minval);
                    maxval = (std::max)(double(x), maxval);
                }
            }
    }

    template<typename FileType>
    static uint32_t get_uint32_pixel_imp (tdata_t src, size_t idx)
    {
        FileType x = *(((FileType*)src) + idx);
        return (uint32_t) x;
    }

    template<typename FileType>
    static double get_dp_pixel_imp (tdata_t src, size_t idx)
    {
        FileType x = *(((FileType*)src) + idx);
        return (double) x;
    }

    double (*get_dpequiv_pixel_typeresolved) (tdata_t src, size_t idx) = nullptr;
    uint32_t (*get_uint32_pixel_typeresolved) (tdata_t src, size_t idx) = nullptr;

    size_t STRIP_TILE_HEIGHT = 1024;
    size_t STRIP_TILE_WIDTH = 1024;
    size_t STRIP_TILE_DEPTH = 1;

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

    double minval, maxval; 

    // low level buffer
    tdata_t tiffTile = nullptr;
    size_t t_szb = 0; 
};

class RawTiffStripLoader : public RawFormatLoader
{
public:

    RawTiffStripLoader(
        size_t numberThreads,
        std::string const& filePath)
        : RawFormatLoader ("NyxusGrayscaleTiffStripLoader", filePath)
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

            tileWidth_ = (std::min) (fullWidth_, STRIP_TILE_WIDTH);
            tileHeight_ = (std::min) (fullHeight_, STRIP_TILE_HEIGHT);
            tileDepth_ = (std::min) (fullDepth_, STRIP_TILE_DEPTH);

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

    ~RawTiffStripLoader() override
    {
        if (tiff_)
        {
            TIFFClose(tiff_);
            tiff_ = nullptr;
        }
    }

    void loadTileFromFile (
        size_t indexRowGlobalTile,
        size_t indexColGlobalTile,
        size_t indexLayerGlobalTile,
        [[maybe_unused]] size_t level) override
    {
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
                    case 8: scan_row_minmax<uint8_t> (buf, layer - startLayer, row - startRow, startCol, endCol);
                        break;
                    case 16: scan_row_minmax<uint16_t> (buf, layer - startLayer, row - startRow, startCol, endCol);
                        break;
                    case 32: scan_row_minmax<size_t> (buf, layer - startLayer, row - startRow, startCol, endCol);
                        break;
                    case 64: scan_row_minmax<uint64_t> (buf, layer - startLayer, row - startRow, startCol, endCol);
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
                    case 8: scan_row_minmax<int8_t> (buf, layer - startLayer, row - startRow, startCol, endCol);
                        break;
                    case 16: scan_row_minmax<int16_t> (buf, layer - startLayer, row - startRow, startCol, endCol);
                        break;
                    case 32: scan_row_minmax<int32_t> (buf, layer - startLayer, row - startRow, startCol, endCol);
                        break;
                    case 64: scan_row_minmax<int64_t> (buf, layer - startLayer, row - startRow, startCol, endCol);
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
                    case 32: scan_row_minmax<float> (buf, layer - startLayer, row - startRow, startCol, endCol);
                        break;
                    case 64: scan_row_minmax<double> (buf, layer - startLayer, row - startRow, startCol, endCol);
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
    }

    [[nodiscard]] size_t fullHeight([[maybe_unused]] size_t level) const override { return fullHeight_; }
    [[nodiscard]] size_t fullWidth([[maybe_unused]] size_t level) const override { return fullWidth_; }
    [[nodiscard]] size_t fullDepth([[maybe_unused]] size_t level) const override { return fullDepth_; }
    [[nodiscard]] size_t tileWidth([[maybe_unused]] size_t level) const override { return tileWidth_; }
    [[nodiscard]] size_t tileHeight([[maybe_unused]] size_t level) const override { return tileHeight_; }
    [[nodiscard]] size_t tileDepth([[maybe_unused]] size_t level) const override { return tileDepth_; }
    [[nodiscard]] short bitsPerSample() const override { return bitsPerSample_; }
    [[nodiscard]] size_t numberPyramidLevels() const override { return 1; }

    void free_tile() override
    {
        _TIFFfree(buf);
    }

    uint32_t get_uint32_pixel(size_t idx) const
    {
        uint32_t rv = 0;

        return rv;
    }

    double get_dpequiv_pixel(size_t idx) const
    {
        double rv = 0;

        return rv;
    }


private:

    template<typename FileType>
    void scan_row_minmax (tdata_t src,
        size_t layer,
        size_t row,
        size_t start_col,
        size_t end_col)
    {
        // Get ahold of the raw pointer
        minval = (std::numeric_limits<double>::max)();
        maxval = (std::numeric_limits<double>::min)();

        for (size_t col = start_col; col < end_col; col++)
        {
            // Logic to prevent "noise" in images whose dimensions are smaller than the default tile buffer size 1024x1024
            FileType x = 0.0; // Zero-fill gaps

            // - Informative zone of the strip
            if (layer < fullDepth_ && row < fullHeight_ && col < fullWidth_)
                x = ((FileType*)(src))[col];

            minval = (std::min)(double(x), minval);
            maxval = (std::max)(double(x), maxval);
        }
    }

    size_t STRIP_TILE_HEIGHT = 1024;
    size_t STRIP_TILE_WIDTH = 1024;
    size_t STRIP_TILE_DEPTH = 1;

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

    double minval, maxval;

    // low level buffer
    tdata_t buf = nullptr;
};

