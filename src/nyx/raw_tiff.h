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
#include "tiff_handle_guard.h"
#include <cstring>
#include <limits.h>

#pragma once
#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "raw_format.h"
#include "ome/ome_tiff_planes.h"   // OME-XML reading and (z,c,t) -> directory selection
#include "tiff_sample.h"           // the sample type of each (SampleFormat, BitsPerSample)

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
            // A constructor that throws never runs its destructor, and every check below can
            // throw, so the handle opened above would leak on each of those paths. The guard
            // closes it there and is dismissed once the handle is this loader's to keep.
            Nyxus::TiffHandleGuard tiffGuard (tiff_);

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
            // SAMPLEFORMAT is optional and defaults to 1 (unsigned integer); tifffile omits it
            // for unsigned images
            if (sampleFormat_ < 1 || sampleFormat_ > 3)
                sampleFormat_ = 1;

            // Test if the file is greyscale
            if (samplesPerPixel != 1)
            {
                std::string erm = "RawTiffTileLoader error: file " + filePath + " is not greyscale, SamplesPerPixel = " + std::to_string(samplesPerPixel);
                std::cerr << erm << "\n";
                throw (std::runtime_error(erm));
            }

            // A tiled OME-TIFF stores one (z,c,t) plane per directory, like the strip variant;
            // its depth is SizeZ, not the directory count (Z*C*T). A plain tiled TIFF's depth is
            // its run of full-size directories, the rule every TIFF loader applies.
            is_ome_ = Nyxus::read_ome_tiff_axes (tiff_, ome_);
            fullDepth_ = is_ome_ ? ome_.sizeZ : Nyxus::plain_tiff_depth (tiff_);

            // the typed sample getters, at the file's own sample width
            Nyxus::with_tiff_sample_type (sampleFormat_, bitsPerSample_, "RawTiffTileLoader", [this] (auto sample)
            {
                using T = decltype(sample);
                get_uint32_pixel_typeresolved = Nyxus::tiff_sample_as_uint32 <T>;
                get_dpequiv_pixel_typeresolved = Nyxus::tiff_sample_as_double <T>;
                fp_pixels_ = Nyxus::tiff_sample_is_real<T>;
            });

            // fully constructed: ~Loader() owns the handle from here
            tiffGuard.dismiss();
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

    void loadTileFromFile (
        size_t indexRowGlobalTile,
        size_t indexColGlobalTile,
        size_t indexLayerGlobalTile,          // Z plane
        size_t indexChannel,                  // C plane (OME); 0 for plain TIFF
        size_t indexTimeframe,                // T plane (OME); 0 for plain TIFF
        size_t level) override
    {
        Nyxus::select_tiff_plane (tiff_, is_ome_ ? &ome_ : nullptr, fullDepth_,
            indexLayerGlobalTile, indexChannel, indexTimeframe, "RawTiffTileLoader");

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
    // C/T extents and physical voxel spacing from the parsed OME-XML (1 / 1.0 for plain TIFF)
    [[nodiscard]] size_t numberChannels() const override { return is_ome_ ? ome_.sizeC : 1; }
    [[nodiscard]] size_t fullTimestamps([[maybe_unused]] size_t level) const override { return is_ome_ ? ome_.sizeT : 1; }
    [[nodiscard]] double physicalSizeX() const override { return is_ome_ ? ome_.physX : 1.0; }
    [[nodiscard]] double physicalSizeY() const override { return is_ome_ ? ome_.physY : 1.0; }
    [[nodiscard]] double physicalSizeZ() const override { return is_ome_ ? ome_.physZ : 1.0; }
    [[nodiscard]] std::string physicalSizeUnit() const override { return is_ome_ ? ome_.unitXY : std::string(); }

private:

    double (*get_dpequiv_pixel_typeresolved) (const void* src, size_t idx) = nullptr;
    uint32_t (*get_uint32_pixel_typeresolved) (const void* src, size_t idx) = nullptr;

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

    // low level buffer
    tdata_t tiffTile = nullptr;
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
            // A constructor that throws never runs its destructor, and every check below can
            // throw, so the handle opened above would leak on each of those paths. The guard
            // closes it there and is dismissed once the handle is this loader's to keep.
            Nyxus::TiffHandleGuard tiffGuard (tiff_);

            // Load/parse header
            TIFFGetField(tiff_, TIFFTAG_IMAGEWIDTH, &(this->fullWidth_));
            TIFFGetField(tiff_, TIFFTAG_IMAGELENGTH, &(this->fullHeight_));
            TIFFGetField(tiff_, TIFFTAG_SAMPLESPERPIXEL, &samplesPerPixel);
            TIFFGetField(tiff_, TIFFTAG_BITSPERSAMPLE, &(this->bitsPerSample_));
            TIFFGetField(tiff_, TIFFTAG_SAMPLEFORMAT, &(this->sampleFormat_));

            // OME-TIFF: the directories are a (z,c,t) rasterization, not a plain Z-stack, so
            // the depth is SizeZ rather than the directory count. A plain TIFF's depth is its
            // run of full-size directories, the rule every TIFF loader applies.
            is_ome_ = Nyxus::read_ome_tiff_axes (tiff_, ome_);
            fullDepth_ = is_ome_ ? ome_.sizeZ : Nyxus::plain_tiff_depth (tiff_);

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

            // the typed sample getters, at the file's own sample width
            Nyxus::with_tiff_sample_type (sampleFormat_, bitsPerSample_, "RawTiffStripLoader", [this] (auto sample)
            {
                using T = decltype(sample);
                get_uint32_pixel_typeresolved = Nyxus::tiff_sample_as_uint32 <T>;
                get_dpequiv_pixel_typeresolved = Nyxus::tiff_sample_as_double <T>;
                fp_pixels_ = Nyxus::tiff_sample_is_real<T>;
            });

            // A strip image is addressed in tileHeight_ x tileWidth_ tiles like a tiled one: a
            // read copies its tile's rows and columns out of the scanlines into a buffer
            // tileWidth_ samples wide.
            scanline_szb = TIFFScanlineSize(tiff_);
            sample_szb = bitsPerSample_ / 8;
            tile_szb = tileHeight_ * tileWidth_ * sample_szb;
            line_.resize (scanline_szb);
            buf = _TIFFmalloc (tile_szb);

            // fully constructed: ~Loader() owns the handle from here
            tiffGuard.dismiss();
        }
        else
        {
            throw std::runtime_error("RawTiffStripLoader error: file " + filePath + " cannot be opened");
        }
    }

    ~RawTiffStripLoader() override
    {
        // The constructor allocates 'buf' eagerly, but free_tile() is the only other place that
        // releases it and a caller only reaches free_tile() by loading a tile. A loader that is
        // constructed and then merely inspected -- reading the (c,t) counts off the OME metadata
        // does exactly that -- would otherwise carry the scanline buffer to the grave.
        if (buf)
        {
            _TIFFfree(buf);
            buf = nullptr;
        }

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
        // free_tile() releases the buffer after each read, so it is re-allocated here; the
        // constructor's allocation covers the first read
        if (buf == nullptr)
        {
            buf = _TIFFmalloc (tile_szb);
            if (!buf)
                throw std::runtime_error("RawTiffStripLoader: _TIFFmalloc failed");
        }

        size_t
            startRow = indexRowGlobalTile * tileHeight_,
            endRow = (std::min) ((indexRowGlobalTile + 1) * tileHeight_, fullHeight_),
            startCol = indexColGlobalTile * tileWidth_,
            endCol = (std::min) ((indexColGlobalTile + 1) * tileWidth_, fullWidth_);

        if (startRow >= fullHeight_ || startCol >= fullWidth_)
            throw std::runtime_error ("RawTiffStripLoader: tile (row,col)=(" + std::to_string(indexRowGlobalTile) + ","
                + std::to_string(indexColGlobalTile) + ") is outside the image");

        // OME-TIFF: the plane's directory per DimensionOrder; plain multi-page TIFF: directory = Z
        Nyxus::select_tiff_plane (tiff_, is_ome_ ? &ome_ : nullptr, fullDepth_,
            indexLayerGlobalTile, indexChannel, indexTimeframe, "RawTiffStripLoader");

        // an edge tile fills only part of the buffer; the rest reads as 0
        auto* fub = static_cast<std::uint8_t*>(buf);
        if (endRow - startRow < tileHeight_ || endCol - startCol < tileWidth_)
            std::memset (fub, 0, tile_szb);

        const size_t lineOffs = startCol * sample_szb,
            rowBytes = (std::min) ((endCol - startCol) * sample_szb, scanline_szb - lineOffs);
        for (size_t r = startRow; r < endRow; r++)
        {
            int errcode = TIFFReadScanline (tiff_, line_.data(), (uint32_t) r);
            if (errcode != 1)
            {
                std::string erm = "error " + std::to_string(errcode) + " calling TIFFReadScanline(row = " + std::to_string(r) + ")";
                throw (std::runtime_error(erm));
            }
            std::memcpy (fub + (r - startRow) * tileWidth_ * sample_szb, line_.data() + lineOffs, rowBytes);
        }
    }

    [[nodiscard]] size_t fullHeight([[maybe_unused]] size_t level) const override { return fullHeight_; }
    [[nodiscard]] size_t fullWidth([[maybe_unused]] size_t level) const override { return fullWidth_; }
    [[nodiscard]] size_t fullDepth([[maybe_unused]] size_t level) const override { return fullDepth_; }
    // C/T extents and physical voxel spacing from the parsed OME-XML (1 / 1.0 for plain TIFF)
    [[nodiscard]] size_t numberChannels() const override { return is_ome_ ? ome_.sizeC : 1; }
    [[nodiscard]] size_t fullTimestamps([[maybe_unused]] size_t level) const override { return is_ome_ ? ome_.sizeT : 1; }
    [[nodiscard]] double physicalSizeX() const override { return is_ome_ ? ome_.physX : 1.0; }
    [[nodiscard]] double physicalSizeY() const override { return is_ome_ ? ome_.physY : 1.0; }
    [[nodiscard]] double physicalSizeZ() const override { return is_ome_ ? ome_.physZ : 1.0; }
    [[nodiscard]] std::string physicalSizeUnit() const override { return is_ome_ ? ome_.unitXY : std::string(); }
    [[nodiscard]] size_t tileWidth([[maybe_unused]] size_t level) const override { return tileWidth_; }
    [[nodiscard]] size_t tileHeight([[maybe_unused]] size_t level) const override { return tileHeight_; }
    [[nodiscard]] size_t tileDepth([[maybe_unused]] size_t level) const override { return tileDepth_; }
    [[nodiscard]] short bitsPerSample() const override { return bitsPerSample_; }
    [[nodiscard]] size_t numberPyramidLevels() const override { return 1; }

    // RawFormatLoader calls free_tile() after each loadTileFromFile(). Freeing is idempotent
    // and the next loadTileFromFile() re-allocates, so a volume assembled plane by plane can
    // free and re-read as often as it needs.
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

    double (*get_dpequiv_pixel_typeresolved) (const void* src, size_t idx) = nullptr;
    uint32_t (*get_uint32_pixel_typeresolved) (const void* src, size_t idx) = nullptr;

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

    // low level buffers: one tile, tileWidth_ samples per row, and one scanline of the file
    tdata_t buf = nullptr;
    std::vector<std::uint8_t> line_;
    size_t scanline_szb = 0,
        sample_szb = 0,           ///< Bytes per sample, in the file and in 'buf'
        tile_szb = 0;             ///< Bytes in 'buf'
};

