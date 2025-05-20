#pragma once

#include <stdexcept>
#include <vector>
#include "abs_tile_loader.h"
#include "raw_format.h"
#include "io/nifti/nifti2_io.h"

class RawNiftiLoader : public RawFormatLoader
{
public:

    RawNiftiLoader (std::string const& filePath) : RawFormatLoader("RawNiftiLoader", filePath)
    {
        // Read input dataset, including data
        nifti_image* niiData = nifti_image_read (filePath.c_str(), 1);
        if (!niiData)
        {
            std::string erm = "error: failed to read NIfTI image from " + filePath;
            std::cerr << erm << "\n";
            throw (std::runtime_error(erm));
        }

        fullHeight_ = niiData->ny;
        fullWidth_ = niiData->nx;
        fullDepth_ = niiData->nz;
    }

    ~RawNiftiLoader() override
    {
    }

    void loadTileFromFile(
        size_t indexRowGlobalTile,
        size_t indexColGlobalTile,
        size_t indexLayerGlobalTile,
        [[maybe_unused]] size_t level) override
    {
        tile.resize(tileWidth_ * tileHeight_);
        std::fill(tile.begin(), tile.end(), 0);

        std::vector<uint32_t>& tileDataVec = tile;
    }

    void free_tile() override {}

    uint32_t get_uint32_pixel (size_t idx) const
    {
        uint32_t rv = tile[idx];
        return rv;
    }

    double get_dpequiv_pixel (size_t idx) const
    {
        double rv = (double)tile[idx];
        return rv;
    }

    [[nodiscard]] size_t fullHeight([[maybe_unused]] size_t level) const override { return fullHeight_; }

    [[nodiscard]] size_t fullWidth([[maybe_unused]] size_t level) const override { return fullWidth_; }

    [[nodiscard]] size_t fullDepth([[maybe_unused]] size_t level) const override { return fullDepth_; }

    [[nodiscard]] size_t tileWidth([[maybe_unused]] size_t level) const override { return tileWidth_; }

    [[nodiscard]] size_t tileHeight([[maybe_unused]] size_t level) const override { return tileHeight_; }

    [[nodiscard]] size_t tileDepth([[maybe_unused]] size_t level) const override { return tileDepth_; }

    [[nodiscard]] short bitsPerSample() const override { return bitsPerSample_; }

    [[nodiscard]] size_t numberPyramidLevels() const override { return 1; }

private:

    size_t
        fullHeight_ = 0,          ///< Full height in pixel
        fullWidth_ = 0,           ///< Full width in pixel
        fullDepth_ = 0,           ///< Full depth in pixel
        tileWidth_ = 0,           ///< Tile width
        tileHeight_ = 0,          ///< Tile height
        tileDepth_ = 0,           ///< Tile depth
        numCols_ = 0,
        numRows_ = 0;

    int32_t numFrames_ = 0;
    int bitsPerSample_ = 1;

    std::vector<uint32_t> tile;
};

template<class DataType>
class NiftiLoader : public AbstractTileLoader<DataType>
{
public:

    NiftiLoader (std::string const& slide_path) : AbstractTileLoader<DataType>("NiftiLoader", 1/*numberThreads*/, slide_path)
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
        [[maybe_unused]] size_t level) override
    {
        tile->resize (tile_width_ * tile_height_ * tile_depth_);
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

        // cache
        size_t nr_voxels = nii->nvox;
        if (nii->datatype == 2) {  // NIFTI_TYPE_UINT8
            uint8_t* nii_data = static_cast<uint8_t*>(nii->data);
            for (int i = 0; i < nr_voxels; ++i) {
                dataCache[i] = static_cast<int32_t>(*(nii_data + i));
            }
        }
        else if (nii->datatype == 512) {  // NIFTI_TYPE_UINT16
            uint16_t* nii_data = static_cast<uint16_t*>(nii->data);
            for (int i = 0; i < nr_voxels; ++i) {
                dataCache[i] = static_cast<int32_t>(*(nii_data + i));
            }
        }
        else if (nii->datatype == 768) {  // NIFTI_TYPE_UINT32
            uint32_t* nii_data = static_cast<uint32_t*>(nii->data);
            for (int i = 0; i < nr_voxels; ++i) {
                dataCache[i] = static_cast<int32_t>(*(nii_data + i));
            }
        }
        else if (nii->datatype == 1280) {  // NIFTI_TYPE_UINT64
            uint64_t* nii_data = static_cast<uint64_t*>(nii->data);
            for (int i = 0; i < nr_voxels; ++i) {
                dataCache[i] = static_cast<int32_t>(*(nii_data + i));
            }
        }
        else if (nii->datatype == 256) {  // NIFTI_TYPE_INT8
            int8_t* nii_data = static_cast<int8_t*>(nii->data);
            for (int i = 0; i < nr_voxels; ++i) {
                dataCache[i] = static_cast<int32_t>(*(nii_data + i));
            }
        }
        else if (nii->datatype == 4) {  // NIFTI_TYPE_INT16
            int16_t* nii_data = static_cast<int16_t*>(nii->data);
            for (int i = 0; i < nr_voxels; ++i) {
                dataCache[i] = static_cast<int32_t>(*(nii_data + i));
            }
        }
        else if (nii->datatype == 8) {  // NIFTI_TYPE_INT32
            int32_t* nii_data = static_cast<int32_t*>(nii->data);
            for (int i = 0; i < nr_voxels; ++i) {
                dataCache[i] = static_cast<int32_t>(*(nii_data + i));
            }
        }
        else if (nii->datatype == 1024) {  // NIFTI_TYPE_INT64
            int64_t* nii_data = static_cast<int64_t*>(nii->data);
            for (int i = 0; i < nr_voxels; ++i) {
                dataCache[i] = static_cast<int32_t>(*(nii_data + i));
            }
        }
        else if (nii->datatype == 16) {  // NIFTI_TYPE_FLOAT32
            float* nii_data = static_cast<float*>(nii->data);
            for (int i = 0; i < nr_voxels; ++i) {
                dataCache[i] = static_cast<int32_t>(*(nii_data + i));
            }
        }
        else if (nii->datatype == 64) {  // NIFTI_TYPE_FLOAT64
            double* nii_data = static_cast<double*>(nii->data);

            // -- min, max
            double mi = nii_data[0],
                mx = mi;
            for (size_t i = 0; i < nr_voxels; i++)
            {
                double a = *(nii_data + i);
                mi = (std::min)(mi, a);
                mx = (std::max)(mx, a);
            }

            for (int i = 0; i < nr_voxels; ++i) {
                double a = *(nii_data + i) - mi;

                dataCache[i] = static_cast<uint32_t>(a);
            }
        }
        else {
            std::string erm = "error: unrecognized NIFTI data type in " + slide_path_;
            std::cerr << erm << "\n";
            throw (std::runtime_error(erm));
        }

        // release memory
        nifti_image_free (nii);
    }

    [[nodiscard]] size_t fullHeight([[maybe_unused]] size_t level) const override { return full_height_; }

    [[nodiscard]] size_t fullWidth([[maybe_unused]] size_t level) const override { return full_width_; }

    [[nodiscard]] size_t fullDepth([[maybe_unused]] size_t level) const override { return full_depth_; }

    [[nodiscard]] size_t tileWidth([[maybe_unused]] size_t level) const override { return tile_width_; }

    [[nodiscard]] size_t tileHeight([[maybe_unused]] size_t level) const override { return tile_height_; }

    [[nodiscard]] size_t tileDepth([[maybe_unused]] size_t level) const override { return tile_depth_; }

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

    int32_t numFrames_ = 0;
    int bitsPerSample_ = 1;

    std::vector<uint32_t> tile;
    std::string slide_path_;
};
