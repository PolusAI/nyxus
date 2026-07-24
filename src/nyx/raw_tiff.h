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
#include <cstdint>
#include <cstring>
#include <limits.h>

#pragma once
#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "raw_format.h"
#include "ome/ome_tiff_meta.h"   // parse_ome_xml -> OmeAxes (OME-TIFF plane->IFD mapping)

class RawTiffTileLoader : public RawFormatLoader
{
public:

    RawTiffTileLoader (std::string const& filePath): RawFormatLoader("RawTiffTileLoader", filePath)
    {
        short samplesPerPixel = 0;

        // Open the file
        tiff_ = TIFFOpen (filePath.c_str(), "r");
        if (tiff_ != nullptr) 
        {
            if (TIFFIsTiled(tiff_) == 0) 
            { 
                std::string erm = "RawTiffTileLoader error: file " + filePath +" is not tiled";
                std::cerr << erm << "\n";
                throw (std::runtime_error(erm)); 
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
            // FIX (IO): default absent SAMPLEFORMAT to 1 (unsigned int), like the strip loader,
            // so tifffile-written tiled uint OME-TIFFs (which omit the tag) read correctly.
            if (sampleFormat_ < 1 || sampleFormat_ > 3)
                sampleFormat_ = 1;

            // Test if the file is greyscale
            if (samplesPerPixel != 1)
            {
                std::string erm = "RawTiffTileLoader error: file " + filePath + " is not greyscale, SamplesPerPixel = " + std::to_string(samplesPerPixel);
                std::cerr << erm << "\n";
                throw (std::runtime_error(erm));
            }

            // FIX (IO): parse a tiled OME-TIFF's IFD-0 OME-XML so (z,c,t) map to the right IFD;
            // a plain tiled TIFF stays 2D (fullDepth = directory count, usually 1).
            fullDepth_ = TIFFNumberOfDirectories(tiff_);
            char* desc = nullptr;
            if (TIFFGetField(tiff_, TIFFTAG_IMAGEDESCRIPTION, &desc) && desc != nullptr)
            {
                ome_ = Nyxus::parse_ome_xml(desc);
                if (ome_.valid)
                {
                    is_ome_ = true;
                    fullDepth_ = ome_.sizeZ;   // depth is SizeZ, not the total IFD count (= Z*C*T)
                }
            }

            // Prepare the right typed getter function
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
                        + std::to_string (bitsPerSample_);
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
                this->fp_pixels_ = true;

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
        size_t indexLayerGlobalTile,          // Z plane
        size_t indexChannel,                  // C plane (OME); 0 for plain 2D TIFF
        size_t indexTimeframe,                // T plane (OME); 0 for plain 2D TIFF
        size_t level) override
    {
        // FIX (IO): select the (z,c,t) plane's IFD before reading the tile (see grayscale_tiff.h).
        if (is_ome_ && (indexLayerGlobalTile >= ome_.sizeZ
            || indexChannel >= ome_.sizeC || indexTimeframe >= ome_.sizeT))
        {
            throw std::runtime_error("RawTiffTileLoader: (z,c,t)=(" + std::to_string(indexLayerGlobalTile)
                + "," + std::to_string(indexChannel) + "," + std::to_string(indexTimeframe) + ") out of range");
        }
        size_t ifd = is_ome_ ? ome_.ifdForPlane(indexLayerGlobalTile, indexChannel, indexTimeframe)
                             : indexLayerGlobalTile;
        if (TIFFSetDirectory(tiff_, (uint16_t)ifd) != 1)
            throw std::runtime_error("RawTiffTileLoader: TIFFSetDirectory(ifd=" + std::to_string(ifd) + ") failed");

        // Low level read TIFF bytes
        auto t_szb = TIFFTileSize(tiff_);
        tiffTile = _TIFFmalloc(t_szb);

        if (!tiffTile)
        {
            std::string erm = std::string("_TIFFmalloc() failed at ") + __FILE__ + ":" + std::to_string(__LINE__);
            std::cerr << "\n\n" << erm << "\n\n";
            throw std::runtime_error(erm);
        }
        
        auto errcode = TIFFReadTile(tiff_, tiffTile, indexColGlobalTile * tileWidth_, indexRowGlobalTile * tileHeight_, 0, 0);
        if (errcode < 0)
        {
            if (errcode == -1) // the TIFF file is not tiled, don't break for each image like that
                memset (tiffTile, 0, t_szb);    
            else // something else
            {
                std::string erm = "Tile Loader ERROR: error reading tile data returning code " + std::to_string(errcode);
                std::cerr << "\n\n" << erm << "\n\n";
                throw std::runtime_error(erm);
            }
        }
    }

    void free_tile() override
    {
        _TIFFfree (tiffTile);
        tiffTile = nullptr;
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
    [[nodiscard]] size_t fullDepth([[maybe_unused]] size_t level) const override { return fullDepth_; }
    [[nodiscard]] size_t tileWidth([[maybe_unused]] size_t level) const { return tileWidth_; }
    [[nodiscard]] size_t tileHeight([[maybe_unused]] size_t level) const { return tileHeight_; }
    [[nodiscard]] short bitsPerSample() const { return bitsPerSample_; }
    [[nodiscard]] size_t numberPyramidLevels() const { return 1; }
    // FIX (IO): advertise C/T + physical spacing from the parsed OME-XML (1 / plain otherwise)
    [[nodiscard]] size_t numberChannels() const override { return is_ome_ ? ome_.sizeC : 1; }
    [[nodiscard]] size_t fullTimestamps([[maybe_unused]] size_t level) const override { return is_ome_ ? ome_.sizeT : 1; }
    [[nodiscard]] double physicalSizeX() const override { return is_ome_ ? ome_.physX : 1.0; }
    [[nodiscard]] double physicalSizeY() const override { return is_ome_ ? ome_.physY : 1.0; }
    [[nodiscard]] double physicalSizeZ() const override { return is_ome_ ? ome_.physZ : 1.0; }
    [[nodiscard]] std::string physicalSizeUnit() const override { return is_ome_ ? ome_.unitXY : std::string(); }

private:

    template<typename FileType>
    void scan_typed_minmax (tdata_t src)
    {
        minval = (std::numeric_limits<double>::max)();
        maxval = (std::numeric_limits<double>::lowest)();

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
        fullDepth_ = 1,            ///< Full depth (Z); >1 for multi-plane OME-TIFF
        tileHeight_ = 0,            ///< Tile height
        tileWidth_ = 0;             ///< Tile width

    bool is_ome_ = false;          ///< true when IFD-0 carries an OME-XML block
    Nyxus::OmeAxes ome_;           ///< parsed OME dimensions (drives the (z,c,t)->IFD map)

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
        : RawFormatLoader ("RawTiffStripLoader", filePath)
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

            // OME-TIFF: the IFDs are a (z,c,t) rasterization, not a plain Z-stack.
            // Parse the OME-XML (IFD-0 ImageDescription) so (z,c,t) map to the right
            // IFD and fullDepth reflects SizeZ, not the total page count.
            {
                char* desc = nullptr;
                if (TIFFGetField(tiff_, TIFFTAG_IMAGEDESCRIPTION, &desc) == 1 && desc
                    && std::string(desc).find("<OME") != std::string::npos)
                {
                    ome_ = Nyxus::parse_ome_xml(desc);
                    if (ome_.valid)
                    {
                        is_ome_ = true;
                        fullDepth_ = ome_.sizeZ;
                    }
                }
            }

            tileWidth_ = (std::min) (fullWidth_, STRIP_TILE_WIDTH);
            tileHeight_ = (std::min) (fullHeight_, STRIP_TILE_HEIGHT);
            tileDepth_ = (std::min) (fullDepth_, STRIP_TILE_DEPTH);

            // Test if the file is grayscale
            if (samplesPerPixel > 1)
            {
                // Sometimes we have images whose samplesPerPixel==0:  if (samplesPerPixel != 1) {
                std::string erm = "Tile Loader ERROR: The file is not grayscale: SamplesPerPixel = " + std::to_string(samplesPerPixel);
                std::cerr << erm + "\n";
                throw std::runtime_error(erm);
            }
            // Interpret undefined data format as unsigned integer data
            if (sampleFormat_ < 1 || sampleFormat_ > 3)
            {
                sampleFormat_ = 1;
            }

            // prepare the right typed getter function
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
                    std::string erm = "RawTiffStripLoader error: data format is not supported for sampleFormat_=" + std::to_string(sampleFormat_) + ", bitsPerSample_ = " + std::to_string(bitsPerSample_);
                    std::cerr << erm << "\n";
                    throw std::runtime_error(erm);
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
                    std::string erm = "RawTiffStripLoader error: data format is not supported for sampleFormat_=" + std::to_string(sampleFormat_) + ", bitsPerSample_ = " + std::to_string(bitsPerSample_);
                    std::cerr << erm << "\n";
                    throw std::runtime_error(erm);
                }
                break;
            case 3:
                this->fp_pixels_ = true;

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
                    std::string erm = "RawTiffStripLoader error: data format is not supported for sampleFormat_=" + std::to_string(sampleFormat_) + ", bitsPerSample_ = " + std::to_string(bitsPerSample_);
                    std::cerr << erm << "\n";
                    throw std::runtime_error(erm);
                }
                break;
            default:
                std::string erm = "RawTiffStripLoader error: unsupported sampleFormat_=" + std::to_string(sampleFormat_);
                std::cerr << erm << "\n";
                throw std::runtime_error(erm);
            }

            scanline_szb = TIFFScanlineSize(tiff_);
            buf = _TIFFmalloc (scanline_szb * tileHeight_);

        }
        else
        {
            throw std::runtime_error("RawTiffStripLoader error: file " + filePath + " cannot be opened");
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
        size_t indexLayerGlobalTile,   // Z (plane page for non-OME multi-page TIFF)
        size_t indexChannel,           // C plane (OME-TIFF only)
        size_t indexTimeframe,         // T plane (OME-TIFF only)
        [[maybe_unused]] size_t level) override
    {
        // FIX: (re)allocate lazily so this loader honors "free_tile() follows each
        // loadTileFromFile()" -- see free_tile(). The ctor allocation covers the first read.
        if (buf == nullptr)
        {
            buf = _TIFFmalloc (scanline_szb * tileHeight_);
            if (!buf)
                throw std::runtime_error("RawTiffStripLoader: _TIFFmalloc failed");
        }

        size_t
            startLayer = indexLayerGlobalTile * tileDepth_,
            endLayer = std::min((indexLayerGlobalTile + 1) * tileDepth_, fullDepth_),
            startRow = indexRowGlobalTile * tileHeight_,
            endRow = std::min((indexRowGlobalTile + 1) * tileHeight_, fullHeight_),
            startCol = indexColGlobalTile * tileWidth_,
            endCol = std::min((indexColGlobalTile + 1) * tileWidth_, fullWidth_);

        // For OME-TIFF the target page is the (z,c,t) IFD per DimensionOrder; for a
        // plain multi-page TIFF the page is the Z layer directly.
        size_t ifd = is_ome_ ? ome_.ifdForPlane(indexLayerGlobalTile, indexChannel, indexTimeframe)
                             : indexLayerGlobalTile;
        auto errcode = TIFFSetDirectory (tiff_, (uint16_t)ifd);
        if (errcode != 1)
        {
            std::string erm = "error " + std::to_string(errcode) + " calling TIFFSetDirectory(ifd = " + std::to_string(ifd) + ")";
            throw (std::runtime_error(erm));
        }

        auto* fub = static_cast<std::uint8_t*>(buf);
        for (size_t r = 0; r < tileHeight_; r++)
        {
            size_t offs = r * scanline_szb;
            auto scanline_buf = &(fub[offs]);
            errcode = TIFFReadScanline (tiff_, scanline_buf, r);
            if (errcode != 1)
            {
                std::string erm = "error " + std::to_string(errcode) + " calling TIFFReadScanline(row = " + std::to_string(r) + ")";
                throw (std::runtime_error(erm));
            }
        }
    }

    [[nodiscard]] size_t fullHeight([[maybe_unused]] size_t level) const override { return fullHeight_; }
    [[nodiscard]] size_t fullWidth([[maybe_unused]] size_t level) const override { return fullWidth_; }
    [[nodiscard]] size_t fullDepth([[maybe_unused]] size_t level) const override { return fullDepth_; }
    // FIX: advertise C/T extents from the parsed OME-XML so the volumetric pipeline
    // iterates channels/timeframes; non-OME TIFF keeps the single-plane default of 1.
    [[nodiscard]] size_t numberChannels() const override { return is_ome_ ? ome_.sizeC : 1; }
    [[nodiscard]] size_t fullTimestamps([[maybe_unused]] size_t level) const override { return is_ome_ ? ome_.sizeT : 1; }
    // FIX (IO): physical voxel spacing from OME-XML PhysicalSize* (1.0 for plain TIFF)
    [[nodiscard]] double physicalSizeX() const override { return is_ome_ ? ome_.physX : 1.0; }
    [[nodiscard]] double physicalSizeY() const override { return is_ome_ ? ome_.physY : 1.0; }
    [[nodiscard]] double physicalSizeZ() const override { return is_ome_ ? ome_.physZ : 1.0; }
    [[nodiscard]] std::string physicalSizeUnit() const override { return is_ome_ ? ome_.unitXY : std::string(); }
    [[nodiscard]] size_t tileWidth([[maybe_unused]] size_t level) const override { return tileWidth_; }
    [[nodiscard]] size_t tileHeight([[maybe_unused]] size_t level) const override { return tileHeight_; }
    [[nodiscard]] size_t tileDepth([[maybe_unused]] size_t level) const override { return tileDepth_; }
    [[nodiscard]] short bitsPerSample() const override { return bitsPerSample_; }
    [[nodiscard]] size_t numberPyramidLevels() const override { return 1; }

    // FIX: honor the RawFormatLoader contract that free_tile() follows EACH
    // loadTileFromFile(). This buffer used to be allocated once in the ctor and never
    // re-allocated, so a second load after a free wrote into freed memory (hit when
    // looping Z to assemble a volume, and latent for multi-tile 2D scans which free per
    // tile). Freeing is now idempotent and loadTileFromFile lazily re-allocates.
    void free_tile() override
    {
        if (buf)
        {
            _TIFFfree(buf);
            buf = nullptr;
        }
    }

    uint32_t get_uint32_pixel(size_t idx) const
    {
        uint32_t rv = get_uint32_pixel_typeresolved (buf, idx);
        return rv;
    }

    double get_dpequiv_pixel(size_t idx) const
    {
        double rv = get_dpequiv_pixel_typeresolved (buf, idx);
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
        maxval = (std::numeric_limits<double>::lowest)();

        for (size_t col = start_col; col < end_col; col++)
        {
            // Logic to prevent "noise" in images whose dimensions are smaller than the default tile buffer size 1024x1024
            FileType x = (FileType) 0.0; // Zero-fill gaps

            // - Informative zone of the strip
            if (layer < fullDepth_ && row < fullHeight_ && col < fullWidth_)
                x = ((FileType*)(src))[col];

            minval = (std::min)(double(x), minval);
            maxval = (std::max)(double(x), maxval);
        }
    }

    template<typename FileType>
    static uint32_t get_uint32_pixel_imp(tdata_t src, size_t idx)
    {
        FileType x = *(((FileType*)src) + idx);
        return (uint32_t)x;
    }

    template<typename FileType>
    static double get_dp_pixel_imp(tdata_t src, size_t idx)
    {
        FileType x = *(((FileType*)src) + idx);
        return (double)x;
    }

    double (*get_dpequiv_pixel_typeresolved) (tdata_t src, size_t idx) = nullptr;
    uint32_t(*get_uint32_pixel_typeresolved) (tdata_t src, size_t idx) = nullptr;

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

    bool is_ome_ = false;         ///< true when IFD-0 carries an OME-XML block
    Nyxus::OmeAxes ome_;          ///< parsed OME dimensions (drives the plane->IFD map)

    double minval, maxval;

    // low level buffer
    tdata_t buf = nullptr;
    size_t scanline_szb = 0;
};

