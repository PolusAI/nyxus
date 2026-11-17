#pragma once

#include <cmath>
#include <stdexcept>
#include <tuple>
#include <vector>
#include "abs_tile_loader.h"
#include "grey_level_cast.h"
#include "raw_format.h"
#include "io/nifti/nifti2_io.h"

class RawNiftiLoader : public RawFormatLoader
{
public:

    RawNiftiLoader (std::string const& filePath)
        : RawFormatLoader("RawNiftiLoader", filePath)
    {
        slide_path_ = filePath;

        nii_ = nifti_image_read (filePath.c_str(), 1);
        if (!nii_)
        {
            std::string erm = "error: failed to read NIfTI image from " + filePath;
            std::cerr << erm << "\n";
            throw (std::runtime_error(erm));
        }

        // A NIfTI stores values the header rescales into physical units (Hounsfield for CT):
        // true value = slope*stored + intercept. Per the spec scl_slope==0 means "no scaling",
        // so fall back to an identity transform.
        scl_slope_ = (nii_->scl_slope != 0.0) ? (double)nii_->scl_slope : 1.0;
        scl_inter_ = (nii_->scl_slope != 0.0) ? (double)nii_->scl_inter : 0.0;

        switch (nii_->datatype)
        {
        case 2: // NIFTI_TYPE_UINT8
                get_uint32_pixel_typeresolved = get_uint32_pixel_imp <uint8_t>;
                get_dpequiv_pixel_typeresolved = get_dp_pixel_imp <uint8_t>;
                break;
        case 512: // NIFTI_TYPE_UINT16
                get_uint32_pixel_typeresolved = get_uint32_pixel_imp <uint16_t>;
                get_dpequiv_pixel_typeresolved = get_dp_pixel_imp <uint16_t>;
                break;
        case 768: // NIFTI_TYPE_UINT32
                get_uint32_pixel_typeresolved = get_uint32_pixel_imp <uint32_t>;
                get_dpequiv_pixel_typeresolved = get_dp_pixel_imp <uint32_t>;
                break;
        case 1280: // NIFTI_TYPE_UINT64
                get_uint32_pixel_typeresolved = get_uint32_pixel_imp <uint64_t>;
                get_dpequiv_pixel_typeresolved = get_dp_pixel_imp <uint64_t>;
                break;
        case 256: // NIFTI_TYPE_INT8
                get_uint32_pixel_typeresolved = get_uint32_pixel_imp <int8_t>;
                get_dpequiv_pixel_typeresolved = get_dp_pixel_imp <int8_t>;
                break;
        case 4: // NIFTI_TYPE_INT16
                get_uint32_pixel_typeresolved = get_uint32_pixel_imp <int16_t>;
                get_dpequiv_pixel_typeresolved = get_dp_pixel_imp <int16_t>;
                break;
        case 8: // NIFTI_TYPE_INT32
                get_uint32_pixel_typeresolved = get_uint32_pixel_imp <int32_t>;
                get_dpequiv_pixel_typeresolved = get_dp_pixel_imp <int32_t>;
                break;
        case 1024: // NIFTI_TYPE_INT64
                get_uint32_pixel_typeresolved = get_uint32_pixel_imp <int64_t>;
                get_dpequiv_pixel_typeresolved = get_dp_pixel_imp <int64_t>;
                break;
        case 16: // NIFTI_TYPE_FLOAT32
                get_uint32_pixel_typeresolved = get_uint32_pixel_imp <float>;
                get_dpequiv_pixel_typeresolved = get_dp_pixel_imp <float>;
                fp_pixels_ = holds_fraction();
                break;
        case 64: // NIFTI_TYPE_FLOAT64
                get_uint32_pixel_typeresolved = get_uint32_pixel_imp <double>;
                get_dpequiv_pixel_typeresolved = get_dp_pixel_imp <double>;
                fp_pixels_ = holds_fraction();
                break;
        default:
                std::string erm = "error: unrecognized NIFTI data type " + std::to_string(nii_->datatype) + " in " + slide_path_;
                std::cerr << erm << "\n";
                throw std::runtime_error(erm);
                break;
        }

        tileHeight_ = fullHeight_ = nii_->ny;
        tileWidth_ = fullWidth_ = nii_->nx;
        tileDepth_ = fullDepth_ = nii_->nz;
        numTimeFrames_ = nii_->nt;

    }

    ~RawNiftiLoader() override
    {
        if (nii_)
        {
            nifti_image_free (nii_);
            nii_ = nullptr;
        }
    }

    // NIFTI is not tiled so the actual data loading is performed in constructor
    void loadTileFromFile(
        size_t indexRowGlobalTile,
        size_t indexColGlobalTile,
        size_t indexLayerGlobalTile,
        [[maybe_unused]] size_t indexChannel,     // NIfTI: single channel
        [[maybe_unused]] size_t indexTimeframe,   // whole 4D volume held in memory; T sliced by consumer
        [[maybe_unused]] size_t level) override {}

    // NIFTI is not tiled
    void free_tile() override {}

    template<typename FileType>
    static uint32_t get_uint32_pixel_imp (const void* src, size_t idx)
    {
        FileType x = *(((FileType*)src) + idx);
        return (uint32_t)x;
    }

    template<typename FileType>
    static double get_dp_pixel_imp (const void* src, size_t idx)
    {
        FileType x = *(((FileType*)src) + idx);
        return (double)x;
    }

    uint32_t get_uint32_pixel (size_t idx) const
    {
        uint32_t rv = get_uint32_pixel_typeresolved (nii_->data, idx);
        return rv;
    }

    double get_dpequiv_pixel (size_t idx) const
    {
        double rv = get_dpequiv_pixel_typeresolved (nii_->data, idx);
        // Report the rescaled physical value so the scanned slide min/max — and hence the offset
        // recorded for this slide — are in the same domain the features report.
        return scl_slope_ * rv + scl_inter_;
    }

    // Every integer datatype; a real-valued volume's samples carry fractions of their own, so it has
    // no stored integers to keep.
    bool get_integer_rescale (double & slope, double & intercept) const override
    {
        if (nii_->datatype == 16 || nii_->datatype == 64)	// NIFTI_TYPE_FLOAT32, NIFTI_TYPE_FLOAT64
            return false;
        slope = scl_slope_;
        intercept = scl_inter_;
        return true;
    }

    [[nodiscard]] size_t fullHeight([[maybe_unused]] size_t level) const override { return fullHeight_; }

    [[nodiscard]] size_t fullWidth([[maybe_unused]] size_t level) const override { return fullWidth_; }

    [[nodiscard]] size_t fullDepth([[maybe_unused]] size_t level) const override { return fullDepth_; }

    [[nodiscard]] size_t fullTimestamps([[maybe_unused]] size_t level) const override
    {
        return numTimeFrames_;
    }

    [[nodiscard]] size_t tileWidth([[maybe_unused]] size_t level) const override { return tileWidth_; }

    [[nodiscard]] size_t tileHeight([[maybe_unused]] size_t level) const override { return tileHeight_; }

    [[nodiscard]] size_t tileDepth([[maybe_unused]] size_t level) const override { return tileDepth_; }

    [[nodiscard]] short bitsPerSample() const override { return bitsPerSample_; }

    [[nodiscard]] size_t numberPyramidLevels() const override { return 1; }

private:
    
    nifti_image* nii_ = nullptr;
    double (*get_dpequiv_pixel_typeresolved) (const void* src, size_t idx) = nullptr;
    uint32_t(*get_uint32_pixel_typeresolved) (const void* src, size_t idx) = nullptr;

    // Header rescale applied by get_dpequiv_pixel so the scanned slide min/max are physical.
    double scl_slope_ = 1.0, scl_inter_ = 0.0;

    size_t
        fullHeight_ = 0,          ///< Full height in pixel
        fullWidth_ = 0,           ///< Full width in pixel
        fullDepth_ = 0,           ///< Full depth in pixel
        tileWidth_ = 0,           ///< Tile width
        tileHeight_ = 0,          ///< Tile height
        tileDepth_ = 0,           ///< Tile depth
        numCols_ = 0,
        numRows_ = 0;

    size_t numTimeFrames_ = 0;

    int bitsPerSample_ = 1;

    std::string slide_path_;

    // Whether a real-valued volume carries a fraction anywhere: a finite voxel whose header-rescaled
    // value is not a whole number. Such a volume reports itself as real-valued, so the scan records
    // the quantized map a real-valued TIFF or OME-Zarr takes and a volume normalised to [0, 1]
    // keeps its fraction. A FLOAT32/FLOAT64 volume holding whole values only -- a CT stored as
    // float, a label-like phantom -- reports itself as integer and keeps 1 grey level == 1
    // intensity unit, so its texture features are binned in its own units. Non-finite voxels say
    // nothing about the volume's values and are skipped, as the scan skips them.
    bool holds_fraction() const
    {
        for (size_t i = 0; i < nii_->nvox; i++)
        {
            double v = get_dpequiv_pixel (i);
            if (std::isfinite (v) && v != std::floor (v))
                return true;
        }
        return false;
    }
};

template<class DataType>
class NiftiLoader : public AbstractTileLoader<DataType>
{
public:

    NiftiLoader (
        std::string const& slide_path,
        double inten_offset = 0.0,		// the shift the scan recorded: physical units, or stored units with the rescale off
        bool rescale_to_physical = true,		// off for a mask volume, whose voxels are labels, and on the stored map
        bool round_offset = false,		// offset map: round to nearest instead of truncating
        bool quantize = false,			// min-max rescale a real-valued volume, instead of the offset map
        double inten_max = 1.0,			// upper end of the clamp, used only when quantize is set
        double target_dyn_range = 1e4)	// grey levels the rescale spans, used only when quantize is set
        : AbstractTileLoader<DataType>("NiftiLoader", 1/*numberThreads*/, slide_path),
          inten_offset_(inten_offset), rescale_(rescale_to_physical), round_offset_(round_offset),
          quantize_(quantize), inten_max_(inten_max), target_dyn_range_(target_dyn_range)
    {
        slide_path_ = slide_path;

        // Read input dataset, excluding data
        nifti_image* niiData = nifti_image_read (slide_path_.c_str(), 0);    
        if (!niiData)
        {
            std::cerr << "** failed to read an image from " + slide_path_ + "\n";
            #ifdef WITH_PYTHON_H
                throw (std::runtime_error("** failed to read an image from " + slide_path_ + "\n"));
            #endif
        }

        tile_height_ = full_height_ = niiData->ny;
        tile_width_ = full_width_ = niiData->nx;
        tile_depth_ = full_depth_ = niiData->nz;
        numTimeFrames_ = niiData->nt;

        nifti_image_free (niiData);
    }

    ~NiftiLoader() override
    {
    }

    void loadTileFromFile(
        std::shared_ptr<std::vector<DataType>> tile,
        size_t indexRowGlobalTile,
        size_t indexColGlobalTile,
        size_t indexLayerGlobalTile,
        [[maybe_unused]] size_t indexChannel,     // NIfTI: single channel
        [[maybe_unused]] size_t indexTimeframe,   // whole 4D volume delivered at once; T sliced by consumer
        [[maybe_unused]] size_t level) override
    {
        tile->resize (tile_width_ * tile_height_ * tile_depth_ * numTimeFrames_);
        std::fill(tile->begin(), tile->end(), 0);

        std::vector<uint32_t>& dataCache = *tile;

        // read the data blob
        nifti_image* nii = nifti_image_read (slide_path_.c_str(), 1);
        if (!nii)
        {
            std::cerr << "** failed to read an image from " + slide_path_ + "\n";
#ifdef WITH_PYTHON_H
            throw (std::runtime_error("** failed to read an image from " + slide_path_ + "\n"));
#endif
        }

        // Header rescale for this read; scl_slope==0 means "no scaling" per the NIfTI spec.
        cur_scl_slope_ = (rescale_ && nii->scl_slope != 0.0) ? (double)nii->scl_slope : 1.0;
        cur_scl_inter_ = (rescale_ && nii->scl_slope != 0.0) ? (double)nii->scl_inter : 0.0;

        // cache
        if (nii->datatype == 2) {  // NIFTI_TYPE_UINT8
            rescale_offset <uint32_t, uint8_t>(dataCache, static_cast<uint8_t*> (nii->data), nii->nvox);
        }
        else if (nii->datatype == 512) {  // NIFTI_TYPE_UINT16
            rescale_offset <uint32_t, uint16_t>(dataCache, static_cast<uint16_t*> (nii->data), nii->nvox);
        }
        else if (nii->datatype == 768) {  // NIFTI_TYPE_UINT32
            rescale_offset <uint32_t, uint32_t>(dataCache, static_cast<uint32_t*> (nii->data), nii->nvox);
        }
        else if (nii->datatype == 1280) {  // NIFTI_TYPE_UINT64
            rescale_offset <uint32_t, uint64_t>(dataCache, static_cast<uint64_t*> (nii->data), nii->nvox);
        }
        else if (nii->datatype == 256) {  // NIFTI_TYPE_INT8
            rescale_offset <uint32_t, int8_t>(dataCache, static_cast<int8_t*> (nii->data), nii->nvox);
        }
        else if (nii->datatype == 4) {  // NIFTI_TYPE_INT16
            rescale_offset <uint32_t, int16_t>(dataCache, static_cast<int16_t*> (nii->data), nii->nvox);
        }
        else if (nii->datatype == 8) {  // NIFTI_TYPE_INT32
            rescale_offset <uint32_t, int32_t>(dataCache, static_cast<int32_t*> (nii->data), nii->nvox);
        }
        else if (nii->datatype == 1024) {  // NIFTI_TYPE_INT64
            rescale_offset <uint32_t, int64_t>(dataCache, static_cast<int64_t*> (nii->data), nii->nvox);
        }
        else if (nii->datatype == 16) {  // NIFTI_TYPE_FLOAT32
            rescale_offset <uint32_t, float>(dataCache, static_cast<float*> (nii->data), nii->nvox);
        }
        else if (nii->datatype == 64) {  // NIFTI_TYPE_FLOAT64
            rescale_offset <uint32_t, double> (dataCache, static_cast<double*> (nii->data), nii->nvox);
        }
        else 
        {
            std::string erm = "error: unrecognized NIFTI data type in " + slide_path_;
            std::cerr << erm << "\n";
            throw (std::runtime_error(erm));
        }

        // release memory
        nifti_image_free (nii);
    }

    [[nodiscard]] size_t fullHeight ([[maybe_unused]] size_t level) const override { return full_height_; }

    [[nodiscard]] size_t fullWidth ([[maybe_unused]] size_t level) const override { return full_width_; }

    [[nodiscard]] size_t fullDepth ([[maybe_unused]] size_t level) const override { return full_depth_; }

    [[nodiscard]] size_t fullTimestamps ([[maybe_unused]] size_t level) const override { return numTimeFrames_; }

    [[nodiscard]] size_t tileWidth ([[maybe_unused]] size_t level) const override { return tile_width_; }

    [[nodiscard]] size_t tileHeight ([[maybe_unused]] size_t level) const override { return tile_height_; }

    [[nodiscard]] size_t tileDepth ([[maybe_unused]] size_t level) const override { return tile_depth_; }

    [[nodiscard]] size_t tileTimestamps ([[maybe_unused]] size_t level) const override { return numTimeFrames_; }

    [[nodiscard]] short bitsPerSample() const override { return 1; }

    [[nodiscard]] size_t numberPyramidLevels() const override { return 1; }

private:

    size_t
        full_height_ = 0,          ///< Full height in pixel
        full_width_ = 0,           ///< Full width in pixel
        full_depth_ = 0,           ///< Full depth in pixel
        tile_width_ = 0,           ///< Tile width
        tile_height_ = 0,          ///< Tile height
        tile_depth_ = 0;           ///< Tile depth

    size_t numTimeFrames_ = 0;

    int bitsPerSample_ = 1;

    std::vector<uint32_t> tile;
    std::string slide_path_;

    // The shift this volume's grey levels carry. On the offset map voxels are rescaled to physical
    // units and shifted by SlideProps::inten_offset (0 unless the volume's own minimum is
    // negative). On the quantized map, which a volume holding fractional values takes when read
    // without --preserve-hu, inten_offset_ is the low end of the [fpmin, fpmax] window the physical values are
    // clamped to and rescaled from. On the stored map rescale_ is off, so the stored integers are
    // shifted by SlideProps::inten_stored_shift and the rescale lives in the recorded inverse
    // instead. The intensity families apply that inverse, so reported statistics are in the
    // volume's own domain on every map. cur_scl_* hold the current read's header rescale, shared
    // with rescale_offset().
    double inten_offset_ = 0.0;
    bool rescale_ = true;
    bool round_offset_ = false;		// offset map: round to nearest (--preserve-hu) instead of truncating
    double cur_scl_slope_ = 1.0, cur_scl_inter_ = 0.0;

    // Whether this volume is min-max rescaled into [0, target_dyn_range_] (a real-valued volume
    // left in its default mode) or carried on the offset map. SlideProps::inten_map is where the
    // choice is made and recorded; ImageLoader::open passes it here, exactly as it does for TIFF
    // and OME-Zarr.
    bool quantize_ = false;
    double inten_max_ = 1.0,
        target_dyn_range_ = 1e4;

   template <class til, class fra>
   void rescale_offset (std::vector<til>& nyxbuf, const fra* houbuf, size_t n)
   {
       // Rescale stored -> physical (slope*stored + intercept) unless the rescale is off, then map
       // the physical value the way the scan recorded. The scan measured its range on the same
       // rescaled values (RawNiftiLoader::get_dpequiv_pixel), so the quantized window and the
       // offset are both in physical units. With the rescale off the stored integers are shifted
       // as they are, and the narrowing is exact.
       for (size_t i = 0; i < n; ++i)
       {
           double v = cur_scl_slope_ * (double)houbuf[i] + cur_scl_inter_;
           nyxbuf[i] = map_intensity<til> (v);
       }
   }

   // One physical value -> the grey level stored for it. Both branches narrow through the
   // Nyxus::grey_level* calls the TIFF and OME-Zarr loaders use, so no backend can drift from
   // another. A non-finite voxel takes grey level 0 on either -- a float volume is free to hold
   // NaN, a masked-out background most often -- and one above the grey type's maximum saturates,
   // which a 64-bit volume or a large scl_slope can reach on the offset map.
   //
   // The offset map shifts by the recorded offset so sub-minimum voxels (negative CT values among
   // them) clamp to 0 instead of wrapping on the unsigned cast. round_offset_ rounds to nearest
   // under --preserve-hu, where scl_slope is routinely fractional on a PET-derived volume and the
   // scale-1 inverse could not recover a truncated fraction; otherwise it truncates. Without the
   // flag the volumes that reach it hold no fraction to lose: a FLOAT32/FLOAT64 volume of whole
   // values (RawNiftiLoader::holds_fraction), an integer one whose header rescale the stored map
   // cannot carry (a zero, negative or non-finite slope), or a constant real-valued volume, which
   // has no window to quantize into.
   //
   // The quantized map hard-clamps to [inten_offset_, inten_max_] and spreads that window over
   // target_dyn_range_ grey levels, so a volume normalised to [0, 1] keeps its fraction.
   template <class til>
   til map_intensity (double v) const
   {
       if (! quantize_)
           return Nyxus::grey_level<til> (v - inten_offset_, round_offset_);
       // Ahead of the clamp, as in the TIFF loaders: the clamp would store +Inf as the top level.
       if (! std::isfinite (v)) return (til) 0;
       double t = v < inten_offset_ ? inten_offset_ : v;
       t = t > inten_max_ ? inten_max_ : t;
       return Nyxus::grey_level_truncated<til> (target_dyn_range_ * (t - inten_offset_) / (inten_max_ - inten_offset_));
   }
};
