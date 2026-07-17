#pragma once

#ifdef OMEZARR_SUPPORT

#include <algorithm>
#include "nlohmann/json.hpp"

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
        nlohmann::json file_attributes, ds_attributes;
        z5::readAttributes(*zarr_ptr_, file_attributes);

        // assume only one dataset is present
        ds_name_ = file_attributes["multiscales"][0]["datasets"][0]["path"].get<std::string>();
        const auto ds_handle = z5::filesystem::handle::Dataset(*zarr_ptr_, ds_name_);
        fs::path metadata_path;
        auto success = z5::filesystem::metadata_detail::getMetadataPath(ds_handle, metadata_path);
        z5::filesystem::metadata_detail::readMetadata(metadata_path, ds_attributes);

        full_depth_ = ds_attributes["shape"][2].get<size_t>();
        full_height_ = ds_attributes["shape"][3].get<size_t>();
        full_width_ = ds_attributes["shape"][4].get<size_t>();
        tile_depth_ = ds_attributes["chunks"][2].get<size_t>();
        tile_height_ = ds_attributes["chunks"][3].get<size_t>();
        tile_width_ = ds_attributes["chunks"][4].get<size_t>();
        std::string dtype_str = ds_attributes["dtype"].get<std::string>();
        if (dtype_str == "<u1") { data_format_ = 1; } //uint8_t
        else if (dtype_str == "<u2") { data_format_ = 2; } //uint16_t
        else if (dtype_str == "<u4") { data_format_ = 3; } //uint32_t
        else if (dtype_str == "<u8") { data_format_ = 4; } //uint16_t
        else if (dtype_str == "<i1") { data_format_ = 5; } //int8_t
        else if (dtype_str == "<i2") { data_format_ = 6; } //int16_t
        else if (dtype_str == "<i4") { data_format_ = 7; } //int32_t
        else if (dtype_str == "<i8") { data_format_ = 8; } //int64_t
        else if (dtype_str == "<f2") { data_format_ = 9; fp_pixels_ = true; } //float
        else if (dtype_str == "<f4") { data_format_ = 9; fp_pixels_ = true; } //float
        else if (dtype_str == "<f8") { data_format_ = 10; fp_pixels_ = true; } //double
        else { data_format_ = 2; } //uint16_t

        // allocate the buffer
        dest = std::vector<uint32_t> (tile_height_ * tile_width_);

        // Open the dataset once and cache the handle. The dataset metadata is
        // immutable for the lifetime of this loader, so there is no need to
        // re-open (and re-parse the .zarray metadata) on every tile read.
        ds_ = z5::openDataset(*zarr_ptr_, ds_name_);
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

        // Create a buffer to hold the read data
        std::vector<FileType> buffer(data_height * data_width);

        // Create an ArrayView into the buffer (z5 3.0.1 uses ArrayView instead of xtensor)
        // Axis order is still assumed [T,C,Z,Y,X] here; the OME 'axes' metadata is honored later.
        z5::types::ShapeType shape = {1, 1, 1, data_height, data_width};
        auto view = z5::multiarray::makeView(buffer.data(), shape);
        z5::types::ShapeType offset = {pixel_timeframe_index, pixel_channel_index, pixel_layer_index, pixel_row_index, pixel_col_index};
        
        // Read subarray from the cached z5 dataset
        z5::multiarray::readSubarray<FileType>(*ds_, view, offset.begin());
        
        // zero-fill the buffer foreseeing its partial filling at incomplete (tail) tiles
        std::fill(dest.begin(), dest.end(), 0);
        
        // Copy from buffer to destination tile, handling partial tiles and type conversion
        for (size_t k = 0; k < data_height; ++k) {
            for (size_t j = 0; j < data_width; ++j) {
                dest[k * tile_width_ + j] = static_cast<uint32_t>(buffer[k * data_width + j]);
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

    /// @brief Tiff bits per sample
    /// @return Size of a sample in bits
    [[nodiscard]] short bitsPerSample() const override { return 1; }
    /// @brief Level accessor
    /// @return 1
    [[nodiscard]] size_t numberPyramidLevels() const override { return 1; }

private:

    size_t
        full_height_ = 0,          ///< Full height in pixel
        full_width_ = 0,           ///< Full width in pixel
        full_depth_ = 0,           ///< Full depth in pixel
        tile_width_ = 0,           ///< Tile width
        tile_height_ = 0,          ///< Tile height
        tile_depth_ = 0;           ///< Tile depth

    short data_format_ = 0;
    std::unique_ptr<z5::filesystem::handle::File> zarr_ptr_;
    std::string ds_name_;
    std::unique_ptr<z5::Dataset> ds_;   ///< Cached dataset handle (opened once)

    std::vector<uint32_t> dest;
};
#endif //OMEZARR_SUPPORT
