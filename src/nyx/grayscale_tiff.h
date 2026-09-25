#pragma once
#include <cmath>
#include "abs_tile_loader.h"
#include "grey_level_cast.h"

#ifdef __APPLE__
    #define uint64 uint64_hack_
    #define int64 int64_hack_
    #include <tiffio.h>
    #undef uint64
    #undef int64
#else
    #include <tiffio.h>
#endif
#include "tiff_handle_guard.h"
#include <cstring>
#include <sstream>
#include <limits.h> // for INT_MAX
#include "ome/ome_tiff_planes.h"   // OME-XML reading and (z,c,t) -> directory selection
#include "tiff_sample.h"           // the sample type of each (SampleFormat, BitsPerSample)

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
    NyxusGrayscaleTiffTileLoader(
        size_t numberThreads, 
        std::string const& filePath, 
        bool permit_fp,
        double _floatpt_image_min_intensity,
        double _floatpt_image_max_intensity,
        double _floatpt_image_target_dyn_range,
        bool _quantize = false,			// min-max rescale a real-valued image, instead of the offset map
        bool _round_offset = false)		// offset map: round to nearest instead of truncating
        : AbstractTileLoader<DataType>("NyxusGrayscaleTiffTileLoader", numberThreads, filePath),
        permit_floatpt_pixels (permit_fp),
        floatpt_image_min_intensity(_floatpt_image_min_intensity),
        floatpt_image_max_intensity(_floatpt_image_max_intensity),
        floatpt_image_target_dyn_range(_floatpt_image_target_dyn_range),
        quantize(_quantize),
        round_offset(_round_offset)
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
            // SAMPLEFORMAT is optional and defaults to 1 (unsigned integer); tifffile omits it
            // for unsigned images
            if (sampleFormat_ < 1 || sampleFormat_ > 3)
                sampleFormat_ = 1;

            // Test if the file is greyscale
            if (samplesPerPixel != 1)
            {
                std::stringstream message;
                message << "Tile Loader ERROR: The file is not greyscale: SamplesPerPixel = " << samplesPerPixel << ".";
                throw (std::runtime_error(message.str()));
            }

            // A tiled OME-TIFF stores one (z,c,t) plane per directory, like the strip variant;
            // its depth is SizeZ, not the directory count (Z*C*T). A plain tiled TIFF's depth is
            // its run of full-size directories, the rule every TIFF loader applies.
            is_ome_ = Nyxus::read_ome_tiff_axes (tiff_, ome_);
            fullDepth_ = is_ome_ ? ome_.sizeZ : Nyxus::plain_tiff_depth (tiff_);

            // fully constructed: ~Loader() owns the handle from here
            tiffGuard.dismiss();
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
        size_t indexLayerGlobalTile,          // Z plane
        size_t indexChannel,                  // C plane (OME); 0 for plain TIFF
        size_t indexTimeframe,                // T plane (OME); 0 for plain TIFF
        size_t level) override
    {
        std::string err;

        Nyxus::select_tiff_plane (tiff_, is_ome_ ? &ome_ : nullptr, fullDepth_,
            indexLayerGlobalTile, indexChannel, indexTimeframe, "NyxusGrayscaleTiffTileLoader");

        // Get ahold of the logical (feature extraction facing) tile buffer from its smart pointer
        std::vector<DataType>& tileDataVec = *tile;

        tdata_t tiffTile = nullptr;
        auto t_szb = TIFFTileSize(tiff_);
        tiffTile = _TIFFmalloc(t_szb);
        auto errcode = TIFFReadTile(tiff_, tiffTile, indexColGlobalTile * tileWidth_, indexRowGlobalTile * tileHeight_, 0, 0);
        if (errcode < 0)
        {
            if (errcode == -1) // the TIFF file is not tiled, don't break for each image like that
                memset (tiffTile, 0, t_szb);    
            else // something else
            {
                err = "Tile Loader ERROR: error reading tile data returning code " + std::to_string (errcode);
                throw (err);
            }
        }

        // check if FP pixels are permitted
        if (permit_floatpt_pixels == false && sampleFormat_ >= 3)
        {
            err = "This file is not permitted to have TIFF sample format (" + std::to_string(sampleFormat_) + ")";
            throw (err);
        }

        // copy at the file's own sample type; the tile buffer is released on every path
        try
        {
            Nyxus::with_tiff_sample_type (sampleFormat_, bitsPerSample_, "NyxusGrayscaleTiffTileLoader", [&] (auto sample)
            {
                using T = decltype(sample);
                if constexpr (Nyxus::tiff_sample_is_real<T>)
                    loadTile_real_intens <T> (tiffTile, tileDataVec);
                else
                    loadTile <T> (tiffTile, tileDataVec);
            });
        }
        catch (...)
        {
            _TIFFfree(tiffTile);
            throw;
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

    // Z depth, C/T extents and physical voxel spacing from the parsed OME-XML (1, 1, 1 and 1.0
    // for a plain tiled TIFF)
    [[nodiscard]] size_t fullDepth([[maybe_unused]] size_t level) const override { return fullDepth_; }
    [[nodiscard]] size_t numberChannels() const override { return is_ome_ ? ome_.sizeC : 1; }
    [[nodiscard]] size_t fullTimestamps([[maybe_unused]] size_t level) const override { return is_ome_ ? ome_.sizeT : 1; }
    [[nodiscard]] double physicalSizeX() const override { return is_ome_ ? ome_.physX : 1.0; }
    [[nodiscard]] double physicalSizeY() const override { return is_ome_ ? ome_.physY : 1.0; }
    [[nodiscard]] double physicalSizeZ() const override { return is_ome_ ? ome_.physZ : 1.0; }
    [[nodiscard]] std::string physicalSizeUnit() const override { return is_ome_ ? ome_.unitXY : std::string(); }

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
                    // Integer pixels go through the offset map, whose offset is 0 for a slide
                    // whose own minimum is non-negative (so an ordinary image is copied
                    // unchanged) and the floored minimum otherwise. That keeps an int16 CT from
                    // wrapping (-1024 -> ~4.29e9), a maximum the grey-bin and histogram
                    // allocations would then be sized from (#373).
                    FileType v = *(((FileType*)src) + physOffs);
                    *(dest + logOffs) = offset_map ((double)v);
                }
        }
        else
            // General case the logical buffer is same size (specifically, tile size) as the physical one even if tileWidth_ (e.g. 1024) < fullWidth_ (e.g. 1080)
            {
                size_t n = tileHeight_ * tileWidth_;
                for (size_t i = 0; i < n; i++)
                {
                    // Offset map, as in the tiled branch above (#373).
                    FileType v = *(((FileType*)src) + i);
                    *(dest + i) = offset_map ((double)v);
                }
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
                    auto tmp1 = * (((FileType*)src) + physOffs);    // real-valued raw (uncast) intensity e.g. 0.0724
                    *(dest + logOffs) = map_real_intensity (tmp1);
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
                    *(dest + i) = map_real_intensity (tmp1);
                }
            }
    }

    TIFF*
        tiff_ = nullptr;             ///< Tiff file pointer

    size_t
        fullHeight_ = 0,           ///< Full height in pixel
        fullWidth_ = 0,            ///< Full width in pixel
        fullDepth_ = 1,            ///< Full depth (Z) in planes; >1 for multi-plane OME-TIFF
        tileHeight_ = 0,            ///< Tile height
        tileWidth_ = 0;             ///< Tile width

    bool is_ome_ = false;          ///< true when IFD-0 carries an OME-XML block
    Nyxus::OmeAxes ome_;           ///< parsed OME dimensions (drives the (z,c,t)->IFD map)

    short
        sampleFormat_ = 0,          ///< Sample format as defined by libtiff
        bitsPerSample_ = 0;         ///< Bit Per Sample as defined by libtiff

    bool permit_floatpt_pixels = true;  // whether image pixels can be real-valued (intensity image files) or not (mask image files)

    double floatpt_image_min_intensity = 0.0,
        floatpt_image_max_intensity = 1.0,
        floatpt_image_target_dyn_range = 1e4;

    // Whether this slide is min-max rescaled into [0, target_dyn_range] (a real-valued image
    // left in its default mode) or carried on the offset map below. SlideProps::inten_map is
    // where the choice is made and recorded; ImageLoader::open passes it here.
    bool quantize = false;
    bool round_offset = false;		// offset map: round to nearest (--preserve-hu) instead of truncating

    // The offset map, shared by the real-valued and native-integer paths, keeping 1 grey level ==
    // 1 intensity unit. The intensity families add the offset back, so reported statistics are in
    // the slide's own domain. round_offset rounds to nearest under --preserve-hu, which exists to
    // carry absolute intensities and whose scale-1 inverse cannot recover a dropped fraction;
    // otherwise it truncates, which is what a real-valued slide read without the flag has always
    // done. Nyxus::grey_level() carries the narrowing every load-time map shares: non-finite to 0,
    // clamped below, saturated above.
    DataType offset_map (double x) const
    {
        return Nyxus::grey_level<DataType> (x - floatpt_image_min_intensity, round_offset);
    }

    // Map one real-valued pixel to the integer feature domain.
    DataType map_real_intensity (double x) const
    {
        if (! quantize)
            return offset_map (x);
        // Ahead of the clamp, not left to the narrowing: the clamp would turn +Inf into
        // floatpt_image_max_intensity and store the top grey level, where a non-finite sample
        // takes grey level 0 on every other map.
        if (! std::isfinite (x)) return (DataType) 0;
        double t = x < floatpt_image_min_intensity ? floatpt_image_min_intensity : x;
        t = t > floatpt_image_max_intensity ? floatpt_image_max_intensity : t;
        return Nyxus::grey_level_truncated<DataType> (floatpt_image_target_dyn_range * (t - floatpt_image_min_intensity) / (floatpt_image_max_intensity - floatpt_image_min_intensity));
    }

};

/// @brief Tile Loader for 3D Grayscale tiff files encoded in strips
/// @tparam DataType AbstractView's internal type
template<class DataType>
class NyxusGrayscaleTiffStripLoader : public AbstractTileLoader<DataType> 
{
public:

    /// @brief NyxusGrayscaleTiffStripLoader constructor
    /// @param numberThreads Number of threads associated
    /// @param filePath Path of tiff file
    /// @param _inten_offset Offset of the load-time map (SlideProps::inten_offset)
    /// @param _inten_max Upper end of the min-max rescale, used only when _quantize is set
    /// @param _target_dyn_range Grey levels the rescale spans, used only when _quantize is set
    /// @param _quantize Min-max rescale a real-valued image, instead of the offset map
    NyxusGrayscaleTiffStripLoader(
        size_t numberThreads,
        std::string const& filePath,
        double _inten_offset = 0.0,
        double _inten_max = 1.0,
        double _target_dyn_range = 1e4,
        bool _quantize = false,
        bool _round_offset = false)
        : AbstractTileLoader<DataType>("NyxusGrayscaleTiffStripLoader", numberThreads, filePath),
        inten_offset_(_inten_offset),
        inten_max_(_inten_max),
        target_dyn_range_(_target_dyn_range),
        quantize_(_quantize),
        round_offset_(_round_offset)
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

            // fully constructed: ~Loader() owns the handle from here
            tiffGuard.dismiss();
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
        size_t indexLayerGlobalTile,   // Z (plane page for non-OME multi-page TIFF)
        size_t indexChannel,           // C plane (OME-TIFF only)
        size_t indexTimeframe,         // T plane (OME-TIFF only)
        [[maybe_unused]] size_t level) override
    {
        // Get ahold of the logical (feature extraction facing) tile buffer from its smart pointer
        std::vector<DataType>& tileDataVec = *tile;

        tdata_t buf;
        uint32_t row, layer;

        // The plane is checked before anything is read: an out-of-range Z would otherwise skip
        // the layer loop below and leave the tile as it was
        Nyxus::select_tiff_plane (tiff_, is_ome_ ? &ome_ : nullptr, fullDepth_,
            indexLayerGlobalTile, indexChannel, indexTimeframe, "NyxusGrayscaleTiffStripLoader");

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
            // OME-TIFF: the plane's directory per DimensionOrder; plain multi-page TIFF: directory = Z
            if (layer != startLayer)
            {
                try
                {
                    Nyxus::select_tiff_plane (tiff_, is_ome_ ? &ome_ : nullptr, fullDepth_,
                        layer, indexChannel, indexTimeframe, "NyxusGrayscaleTiffStripLoader");
                }
                catch (...)
                {
                    _TIFFfree(buf);
                    throw;
                }
            }
            for (row = startRow; row < endRow; row++)
            {
                TIFFReadScanline(tiff_, buf, row);
                // copy at the file's own sample type; the scanline buffer is released on every path
                try
                {
                    Nyxus::with_tiff_sample_type (sampleFormat_, bitsPerSample_, "NyxusGrayscaleTiffStripLoader", [&] (auto sample)
                    {
                        copyRow <decltype(sample)> (buf, tileDataVec, layer - startLayer, row - startRow, startCol, endCol);
                    });
                }
                catch (...)
                {
                    _TIFFfree(buf);
                    throw;
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

    /// @brief Channel (C) extent from OME-XML (1 for plain TIFF)
    [[nodiscard]] size_t numberChannels() const override { return is_ome_ ? ome_.sizeC : 1; }
    /// @brief Time (T) extent from OME-XML (1 for plain TIFF)
    [[nodiscard]] size_t fullTimestamps([[maybe_unused]] size_t level) const override { return is_ome_ ? ome_.sizeT : 1; }
    /// @brief Physical voxel spacing from OME-XML PhysicalSize* (1.0 for plain TIFF)
    [[nodiscard]] double physicalSizeX() const override { return is_ome_ ? ome_.physX : 1.0; }
    [[nodiscard]] double physicalSizeY() const override { return is_ome_ ? ome_.physY : 1.0; }
    [[nodiscard]] double physicalSizeZ() const override { return is_ome_ ? ome_.physZ : 1.0; }
    [[nodiscard]] std::string physicalSizeUnit() const override { return is_ome_ ? ome_.unitXY : std::string(); }

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
            {
                // The same load-time map the tile loader applies: an offset by the slide's floored
                // minimum, which is what keeps a signed int16 CT pixel from wrapping to ~4.29e9 and
                // blowing up the max-intensity-sized grey-bin/histogram allocation (#373), or the
                // min-max rescale when the slide is real-valued.
                dataItem = map_intensity ((double)((FileType*)(src))[col]);
            }
            
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

    // The load-time map recorded for this slide; see the tile loader's twin of these.
    double inten_offset_ = 0.0,
        inten_max_ = 1.0,
        target_dyn_range_ = 1e4;
    bool quantize_ = false;
    bool round_offset_ = false;		// offset map: round to nearest (--preserve-hu) instead of truncating

    // The same two maps the tile loader carries, over the same shared narrowing, so a tiled and a
    // stripped read of the same slide store the same grey levels.
    DataType map_intensity (double x) const
    {
        if (! quantize_)
            return Nyxus::grey_level<DataType> (x - inten_offset_, round_offset_);
        // Ahead of the clamp, as in the tile loader: the clamp would store +Inf as the top level.
        if (! std::isfinite (x)) return (DataType) 0;
        double t = x < inten_offset_ ? inten_offset_ : x;
        t = t > inten_max_ ? inten_max_ : t;
        return Nyxus::grey_level_truncated<DataType> (target_dyn_range_ * (t - inten_offset_) / (inten_max_ - inten_offset_));
    }

    bool is_ome_ = false;         ///< true when IFD-0 carries an OME-XML block
    Nyxus::OmeAxes ome_;          ///< parsed OME dimensions (drives the plane->IFD map)

};
