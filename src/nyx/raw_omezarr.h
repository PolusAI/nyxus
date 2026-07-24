#pragma once

#ifdef OMEZARR_SUPPORT

#include <algorithm>
#include <vector>
#include <stdexcept>
#include "nlohmann/json.hpp"
#include "ome/ome_zarr_layout.h"   // resolve_zarr_layout() -- shared with omezarr.h

// factory functions to create files, groups and datasets
#include "z5/factory.hxx"
// dataset type (cached handle member)
#include "z5/dataset.hxx"
// handles for z5 filesystem objects
#include "z5/filesystem/handle.hxx"
// z5 multiarray API (ArrayView-based, no xtensor)
#include "z5/multiarray/array_view.hxx"
#include "z5/multiarray/array_access.hxx"
// z5 types
#include "z5/types/types.hxx"
// attribute functionality
#include "z5/attributes.hxx"

#include "raw_format.h"

class RawOmezarrLoader: public RawFormatLoader
{
public:

    RawOmezarrLoader (std::string const& filePath): RawFormatLoader("RawOmezarrLoader", filePath)
    {
        // Open the file
        zarr_ptr_ = std::make_unique<z5::filesystem::handle::File>(filePath.c_str());
        nlohmann::json file_attributes;
        z5::readAttributes(*zarr_ptr_, file_attributes);

        // Resolve the level-0 dataset path
        ds_name_ = Nyxus::zarr_multiscales_root(file_attributes)["multiscales"][0]["datasets"][0]["path"].get<std::string>();

        // FIX (IO): open via z5 (auto-detects Zarr v2 .zarray vs v3 zarr.json, handles v3
        // chunk-key encoding, codecs and sharding) and query shape/chunking/dtype from the
        // Dataset object -- format-agnostic, so this reads BOTH OME-Zarr 0.4 and 0.5.
        ds_ = z5::openDataset(*zarr_ptr_, ds_name_);
        std::vector<size_t> level0Shape(ds_->shape().begin(), ds_->shape().end());
        std::vector<size_t> chunkShape(ds_->defaultChunkShape().begin(), ds_->defaultChunkShape().end());

        // Axis roles, extents, chunking, calibration and pixel type, resolved from the NGFF
        // 'axes' metadata (shared with NyxusOmeZarrLoader -- see ome/ome_zarr_layout.h)
        Nyxus::ZarrLayout L = Nyxus::resolve_zarr_layout (file_attributes, level0Shape, chunkShape, ds_->getDtype());
        ndim_ = L.ndim;
        ix_ = L.ix; iy_ = L.iy; iz_ = L.iz; ic_ = L.ic; it_ = L.it;
        full_width_ = L.full_width; full_height_ = L.full_height; full_depth_ = L.full_depth;
        tile_width_ = L.tile_width; tile_height_ = L.tile_height; tile_depth_ = L.tile_depth;
        n_levels_ = L.n_levels; n_channels_ = L.n_channels; n_timeframes_ = L.n_timeframes;
        phys_x_ = L.phys_x; phys_y_ = L.phys_y; phys_z_ = L.phys_z; phys_unit_ = L.phys_unit;
        bits_per_sample_ = L.bits_per_sample;
        data_format_ = L.data_format;
        fp_pixels_ = L.fp_pixels;

        // The buffer holds the full chunk depth (tile_depth_ planes), so callers iterating
        // pz in [0, tileDepth()) index valid data.
        dest = std::vector<uint32_t> (tile_height_ * tile_width_ * tile_depth_);
    }

    ~RawOmezarrLoader() override
    {
        ds_ = nullptr;
        zarr_ptr_ = nullptr;
    }

    void loadTileFromFile(
        size_t indexRowGlobalTile,
        size_t indexColGlobalTile,
        size_t indexLayerGlobalTile,
        size_t indexChannel,        // C plane to read (offset into the channel axis)
        size_t indexTimeframe,      // T plane to read (offset into the time axis)
        [[maybe_unused]] size_t level) override
    {
        size_t pixel_row_index = indexRowGlobalTile * tile_height_;
        size_t pixel_col_index = indexColGlobalTile * tile_width_;
        size_t pixel_layer_index = indexLayerGlobalTile * tile_depth_;

        switch (data_format_)
        {
        case 1:
            loadTile<uint8_t>(pixel_row_index, pixel_col_index, pixel_layer_index, indexChannel, indexTimeframe);
            break;
        case 2:
            loadTile<uint16_t>(pixel_row_index, pixel_col_index, pixel_layer_index, indexChannel, indexTimeframe);
            break;
        case 3:
            loadTile<uint32_t>(pixel_row_index, pixel_col_index, pixel_layer_index, indexChannel, indexTimeframe);
            break;
        case 4:
            loadTile<uint64_t>(pixel_row_index, pixel_col_index, pixel_layer_index, indexChannel, indexTimeframe);
            break;
        case 5:
            loadTile<int8_t>(pixel_row_index, pixel_col_index, pixel_layer_index, indexChannel, indexTimeframe);
            break;
        case 6:
            loadTile<int16_t>(pixel_row_index, pixel_col_index, pixel_layer_index, indexChannel, indexTimeframe);
            break;
        case 7:
            loadTile<int32_t>(pixel_row_index, pixel_col_index, pixel_layer_index, indexChannel, indexTimeframe);
            break;
        case 8:
            loadTile<int64_t>(pixel_row_index, pixel_col_index, pixel_layer_index, indexChannel, indexTimeframe);
            break;
        case 9:
            loadTile<float>(pixel_row_index, pixel_col_index, pixel_layer_index, indexChannel, indexTimeframe);
            break;
        case 10:
            loadTile<double>(pixel_row_index, pixel_col_index, pixel_layer_index, indexChannel, indexTimeframe);
            break;
        default:
            loadTile<uint16_t>(pixel_row_index, pixel_col_index, pixel_layer_index, indexChannel, indexTimeframe);
            break;
        }
    }

    void free_tile() override
    {
    }

    uint32_t get_uint32_pixel (size_t idx) const
    {
        uint32_t rv = dest[idx];
        return rv;
    }

    double get_dpequiv_pixel (size_t idx) const
    {
        double rv = (double) dest[idx];
        return rv;    
    }

    template<typename FileType>
    void loadTile(size_t pixel_row_index, size_t pixel_col_index, size_t pixel_layer_index,
                  size_t pixel_channel_index, size_t pixel_timeframe_index)
    {
        size_t data_height = tile_height_, data_width = tile_width_;
        if (pixel_row_index + data_height > full_height_) {
            data_height = full_height_ - pixel_row_index;
        }
        if (pixel_col_index + data_width > full_width_) {
            data_width = full_width_ - pixel_col_index;
        }
        // A chunk may span several Z-planes (tile_depth_ > 1); read its whole Z-extent in one
        // subarray call, clamped at the last (possibly partial) chunk.
        size_t data_depth = 1;
        if (iz_ >= 0) {
            data_depth = tile_depth_;
            if (pixel_layer_index + data_depth > full_depth_)
                data_depth = full_depth_ - pixel_layer_index;
        }

        // Create a buffer to hold the read data
        std::vector<FileType> buffer(data_height * data_width * data_depth);

        // Build the read window by axis ROLE (honoring the resolved 'axes' order),
        // reading a Z*Y*X block (data_depth Z-planes) at the requested C/T. z5 3.0.1 uses ArrayView.
        z5::types::ShapeType shape(ndim_, 1), offset(ndim_, 0);
        shape[iy_] = data_height; offset[iy_] = pixel_row_index;
        shape[ix_] = data_width;  offset[ix_] = pixel_col_index;
        if (iz_ >= 0) { shape[iz_] = data_depth; offset[iz_] = pixel_layer_index; }
        if (ic_ >= 0) offset[ic_] = pixel_channel_index;
        if (it_ >= 0) offset[it_] = pixel_timeframe_index;
        auto view = z5::multiarray::makeView(buffer.data(), shape);

        // Read subarray from the cached z5 dataset
        z5::multiarray::readSubarray<FileType>(*ds_, view, offset.begin());

        // zero-fill the buffer foreseeing its partial filling at incomplete (tail) tiles
        std::fill(dest.begin(), dest.end(), 0);

        // Copy from buffer to destination tile, handling partial tiles, multiple Z-planes and
        // type conversion (dest is plane-major: plane p, row k at (p*tile_height_+k)*tile_width_).
        for (size_t p = 0; p < data_depth; ++p) {
            for (size_t k = 0; k < data_height; ++k) {
                for (size_t j = 0; j < data_width; ++j) {
                    dest[(p * tile_height_ + k) * tile_width_ + j] =
                        static_cast<uint32_t>(buffer[(p * data_height + k) * data_width + j]);
                }
            }
        }
    }

    /// @brief Tiff file height
    /// @param level Tiff level [not used]
    /// @return Full height
    [[nodiscard]] size_t fullHeight([[maybe_unused]] size_t level) const override { return full_height_; }
    /// @brief Tiff full width
    /// @param level Tiff level [not used]
    /// @return Full width
    [[nodiscard]] size_t fullWidth([[maybe_unused]] size_t level) const override { return full_width_; }
    /// @brief Tiff full depth
    /// @param level Tiff level [not used]
    /// @return Full Depth
    [[nodiscard]] size_t fullDepth([[maybe_unused]] size_t level) const override { return full_depth_; }

    /// @brief Tiff tile width
    /// @param level Tiff level [not used]
    /// @return Tile width
    [[nodiscard]] size_t tileWidth([[maybe_unused]] size_t level) const override { return tile_width_; }
    /// @brief Tiff tile height
    /// @param level Tiff level [not used]
    /// @return Tile height
    [[nodiscard]] size_t tileHeight([[maybe_unused]] size_t level) const override { return tile_height_; }
    /// @brief Tiff tile depth
    /// @param level Tiff level [not used]
    /// @return Tile depth
    [[nodiscard]] size_t tileDepth([[maybe_unused]] size_t level) const override { return tile_depth_; }

    /// @brief Bits per sample (resolved from the dataset dtype)
    [[nodiscard]] short bitsPerSample() const override { return bits_per_sample_; }
    /// @brief Number of resolution (pyramid) levels declared in multiscales
    [[nodiscard]] size_t numberPyramidLevels() const override { return n_levels_; }
    /// @brief Channel (C) extent resolved from the NGFF axes (1 if no channel axis)
    [[nodiscard]] size_t numberChannels() const override { return n_channels_; }
    /// @brief Time (T) extent resolved from the NGFF axes (1 if no time axis)
    [[nodiscard]] size_t fullTimestamps([[maybe_unused]] size_t level) const override { return n_timeframes_; }
    /// @brief Physical voxel spacing from the NGFF coordinateTransformations (1.0 if uncalibrated)
    [[nodiscard]] double physicalSizeX() const override { return phys_x_; }
    [[nodiscard]] double physicalSizeY() const override { return phys_y_; }
    [[nodiscard]] double physicalSizeZ() const override { return phys_z_; }
    [[nodiscard]] std::string physicalSizeUnit() const override { return phys_unit_; }

private:

    size_t
        full_height_ = 0,          ///< Full height in pixel
        full_width_ = 0,           ///< Full width in pixel
        full_depth_ = 0,           ///< Full depth in pixel
        tile_width_ = 0,           ///< Tile width
        tile_height_ = 0,          ///< Tile height
        tile_depth_ = 0;           ///< Tile depth

    // Storage-dimension index of each axis role (-1 if the axis is absent).
    int ix_ = 4, iy_ = 3, iz_ = 2, ic_ = 1, it_ = 0;
    size_t ndim_ = 5;              ///< Number of on-disk dimensions (2..5)
    short bits_per_sample_ = 16;   ///< Real bit depth
    size_t n_levels_ = 1;          ///< Pyramid level count
    size_t n_channels_ = 1;        ///< Channel (C) extent
    size_t n_timeframes_ = 1;      ///< Time (T) extent
    double phys_x_ = 1.0, phys_y_ = 1.0, phys_z_ = 1.0;   ///< Physical voxel spacing
    std::string phys_unit_;        ///< Physical-size unit (e.g. "micrometer")

    short data_format_ = 0;
    std::unique_ptr<z5::filesystem::handle::File> zarr_ptr_;
    std::string ds_name_;
    std::unique_ptr<z5::Dataset> ds_;   ///< Cached dataset handle (opened once)

    std::vector<uint32_t> dest;
};
#endif //OMEZARR_SUPPORT
