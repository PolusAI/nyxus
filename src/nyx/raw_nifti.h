#pragma once

#include <cmath>          // FIX: HU mode: std::floor / std::llround for the offset map
#include <stdexcept>
#include <tuple>
#include <vector>
#include "abs_tile_loader.h"
#include "raw_format.h"
#include "io/nifti/nifti2_io.h"

class RawNiftiLoader : public RawFormatLoader
{
public:

    RawNiftiLoader (std::string const& filePath, bool preserve_hu = false)		// FIX: HU mode flag so the slide scan runs in the Hounsfield domain
        : RawFormatLoader("RawNiftiLoader", filePath), preserve_hu_(preserve_hu)		// FIX: remember whether to rescale to true HU
    {
        slide_path_ = filePath;

        nii_ = nifti_image_read (filePath.c_str(), 1);
        if (!nii_)
        {
            std::string erm = "error: failed to read NIfTI image from " + filePath;
            std::cerr << erm << "\n";
            throw (std::runtime_error(erm));
        }

        // FIX: HU rescale (used only in preserve_hu mode): true value = slope*stored + intercept.
        // Per NIfTI spec scl_slope==0 means "no scaling", so fall back to an identity transform.
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
                //unhounsfield <uint32_t, uint32_t>(dataCache, static_cast<uint32_t*> (nii->data), nr_voxels);
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
                break;
        case 64: // NIFTI_TYPE_FLOAT64
                get_uint32_pixel_typeresolved = get_uint32_pixel_imp <double>;
                get_dpequiv_pixel_typeresolved = get_dp_pixel_imp <double>;
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
        // FIX: HU mode: return the true Hounsfield value so the scan's slide min/max is in the
        // HU domain (matches the NiftiLoader offset + float_domain_map reconstruction).
        if (preserve_hu_)
            rv = scl_slope_ * rv + scl_inter_;
        return rv;
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

    // FIX: HU/CT preservation (scan side): when set, get_dpequiv_pixel reports true Hounsfield
    // values (slope*stored + intercept) so the slide min/max is Hounsfield-domain, not raw stored.
    bool preserve_hu_ = false;
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
};

template<class DataType>
class NiftiLoader : public AbstractTileLoader<DataType>
{
public:

    NiftiLoader (std::string const& slide_path, double hu_min_base = 0.0, bool preserve_hu = false)		// FIX: HU mode: offset base (floored global HU min) + flag
        : AbstractTileLoader<DataType>("NiftiLoader", 1/*numberThreads*/, slide_path),
          preserve_hu_(preserve_hu), hu_min_base_(hu_min_base)		// FIX: remember whether/where to offset-preserve
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

        // FIX: HU rescale for this read (used only in preserve_hu mode); scl_slope==0 => identity.
        cur_scl_slope_ = (nii->scl_slope != 0.0) ? (double)nii->scl_slope : 1.0;
        cur_scl_inter_ = (nii->scl_slope != 0.0) ? (double)nii->scl_inter : 0.0;

        // cache
        if (nii->datatype == 2) {  // NIFTI_TYPE_UINT8
            unhounsfield <uint32_t, uint8_t>(dataCache, static_cast<uint8_t*> (nii->data), nii->nvox);
        }
        else if (nii->datatype == 512) {  // NIFTI_TYPE_UINT16
            unhounsfield <uint32_t, uint16_t>(dataCache, static_cast<uint16_t*> (nii->data), nii->nvox);
        }
        else if (nii->datatype == 768) {  // NIFTI_TYPE_UINT32
            unhounsfield <uint32_t, uint32_t>(dataCache, static_cast<uint32_t*> (nii->data), nii->nvox);
        }
        else if (nii->datatype == 1280) {  // NIFTI_TYPE_UINT64
            unhounsfield <uint32_t, uint64_t>(dataCache, static_cast<uint64_t*> (nii->data), nii->nvox);
        }
        else if (nii->datatype == 256) {  // NIFTI_TYPE_INT8
            unhounsfield <uint32_t, int8_t>(dataCache, static_cast<int8_t*> (nii->data), nii->nvox);
        }
        else if (nii->datatype == 4) {  // NIFTI_TYPE_INT16
            unhounsfield <uint32_t, int16_t>(dataCache, static_cast<int16_t*> (nii->data), nii->nvox);
        }
        else if (nii->datatype == 8) {  // NIFTI_TYPE_INT32
            unhounsfield <uint32_t, int32_t>(dataCache, static_cast<int32_t*> (nii->data), nii->nvox);
        }
        else if (nii->datatype == 1024) {  // NIFTI_TYPE_INT64
            unhounsfield <uint32_t, int64_t>(dataCache, static_cast<int64_t*> (nii->data), nii->nvox);
        }
        else if (nii->datatype == 16) {  // NIFTI_TYPE_FLOAT32
            unhounsfield <uint32_t, float>(dataCache, static_cast<float*> (nii->data), nii->nvox);
        }
        else if (nii->datatype == 64) {  // NIFTI_TYPE_FLOAT64
            unhounsfield <uint32_t, double> (dataCache, static_cast<double*> (nii->data), nii->nvox);
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

    // FIX: HU/CT preservation (featurization side). When set, loaded voxels are rescaled to true
    // HU then offset by floor(hu_min_base_) (the scanned Hounsfield-domain slide min). cur_scl_*
    // hold the current read's rescale (from the NIfTI header) shared with unhounsfield().
    bool preserve_hu_ = false;
    double hu_min_base_ = 0.0;
    double cur_scl_slope_ = 1.0, cur_scl_inter_ = 0.0;

   template <class til, class fra>
   void unhounsfield (std::vector<til>& nyxbuf, const fra* houbuf, size_t n)
   {
       // FIX: HU/CT mode: rescale stored -> true HU (slope*stored + intercept), then slope-1 offset
       // by floor(global HU min) so 1 grey level == 1 HU and sub-min voxels (incl. negative CT
       // values) clamp to 0 instead of wrapping on the unsigned cast. Inverted for reporting by
       // IntensityHistogramFeatures::float_domain_map (uses SlideProps::min_preroi_inten).
       if (preserve_hu_)
       {
           double base = std::floor (hu_min_base_);
           for (size_t i = 0; i < n; ++i)
           {
               double hu = cur_scl_slope_ * (double)houbuf[i] + cur_scl_inter_;
               double y = hu - base;
               if (y < 0.0) y = 0.0;
               nyxbuf[i] = static_cast<til>(std::llround(y));
           }
           return;
       }

       // -- widest typed min and max expecting min to be the background radiodensity or so
       double mi = houbuf[0],
           mx = mi;
       for (size_t i = 0; i < n; i++)
       {
           double a = *(houbuf + i);
           mi = (std::min)(mi, a);
           mx = (std::max)(mx, a);
       }

       // -- convert
       if (mi < 0.0)
           for (int i = 0; i < n; ++i)
           {
               double a = *(houbuf + i) - mi;
               nyxbuf[i] = static_cast<til>(a);
           }
       else
           for (int i = 0; i < n; ++i)
           {
               nyxbuf[i] = static_cast<til>(houbuf[i]);
           }
   }
};
