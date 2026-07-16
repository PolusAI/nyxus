#pragma once

#ifdef OMEZARR_SUPPORT

#include <algorithm>
#include <vector>
#include <stdexcept>
#include "abs_tile_loader.h"
#include "nlohmann/json.hpp"
#include "ome/ome_zarr_meta.h"   // parse_ome_zarr -> OmeAxes (axis-role resolution)

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

/// @brief Tile Loader for OMEZarr
/// @tparam DataType AbstractView's internal type
template<class DataType>
class NyxusOmeZarrLoader : public AbstractTileLoader<DataType> 
{
public:

    /// @brief NyxusOmeZarrLoader constructor
    /// @param numberThreads Number of threads associated
    /// @param filePath Path of zarr file
    NyxusOmeZarrLoader(
        size_t numberThreads,
        std::string const& filePath)
        : AbstractTileLoader<DataType>("NyxusOmeZarrLoader", numberThreads, filePath)
    {
        // Open the file
        zarr_ptr_ = std::make_unique<z5::filesystem::handle::File>(filePath.c_str());
        nlohmann::json file_attributes, ds_attributes;
        z5::readAttributes(*zarr_ptr_, file_attributes);

        // open the highest-resolution dataset (level 0)
        ds_name_ = file_attributes["multiscales"][0]["datasets"][0]["path"].get<std::string>();
        const auto ds_handle = z5::filesystem::handle::Dataset(*zarr_ptr_, ds_name_);
        fs::path metadata_path;
        auto success = z5::filesystem::metadata_detail::getMetadataPath(ds_handle, metadata_path);
        z5::filesystem::metadata_detail::readMetadata(metadata_path, ds_attributes);

        std::vector<size_t> level0Shape, chunkShape;
        for (const auto& d : ds_attributes["shape"])  level0Shape.push_back(d.get<size_t>());
        for (const auto& d : ds_attributes["chunks"]) chunkShape.push_back(d.get<size_t>());
        std::string dtype_str = ds_attributes["dtype"].get<std::string>();

        // Resolve axis roles from the NGFF 'axes' metadata instead of assuming a
        // fixed [T,C,Z,Y,X] order. Falls back to legacy 5D TCZYX if 'axes' is absent.
        Nyxus::OmeAxes axes = Nyxus::parse_ome_zarr(file_attributes, level0Shape, dtype_str);
        if (axes.valid)
        {
            // Reject self-inconsistent metadata rather than guess: if the 'axes'
            // count disagrees with the array rank, indexing the shape by axis role
            // would read out of bounds.
            if (axes.storageAxes.size() != level0Shape.size())
                throw std::runtime_error("OME-Zarr: 'axes' count " + std::to_string(axes.storageAxes.size())
                    + " does not match array rank " + std::to_string(level0Shape.size()));
            ndim_ = axes.storageAxes.size();
            ix_ = axes.storageIndexOf('X'); iy_ = axes.storageIndexOf('Y');
            iz_ = axes.storageIndexOf('Z'); ic_ = axes.storageIndexOf('C');
            it_ = axes.storageIndexOf('T');
            n_levels_ = axes.numberPyramidLevels();
        }
        else
        {
            // No usable 'axes': map by position (X,Y last; Z,C,T before) — rank-safe.
            ndim_ = level0Shape.size();
            int n = (int)ndim_;
            ix_ = n - 1; iy_ = n - 2;
            iz_ = (n >= 3) ? n - 3 : -1;
            ic_ = (n >= 4) ? n - 4 : -1;
            it_ = (n >= 5) ? n - 5 : -1;
            n_levels_ = 1;
        }
        // X and Y must resolve to real dimensions, else the read would index OOB.
        if (ix_ < 0 || iy_ < 0 || (size_t)ix_ >= level0Shape.size() || (size_t)iy_ >= level0Shape.size())
            throw std::runtime_error("OME-Zarr: cannot resolve X/Y axes from metadata");

        full_width_  = level0Shape[ix_];
        full_height_ = level0Shape[iy_];
        full_depth_  = (iz_ >= 0) ? level0Shape[iz_] : 1;
        tile_width_  = chunkShape[ix_];
        tile_height_ = chunkShape[iy_];
        tile_depth_  = (iz_ >= 0) ? chunkShape[iz_] : 1;
        bits_per_sample_ = Nyxus::bits_of(Nyxus::pixel_type_from_zarr_dtype(dtype_str));

        // dtype -> internal format code for the read-template dispatch
        if      (dtype_str == "<u1") {data_format_=1;} //uint8_t
        else if (dtype_str == "<u2") {data_format_=2;} //uint16_t
        else if (dtype_str == "<u4") {data_format_=3;} //uint32_t
        else if (dtype_str == "<u8") {data_format_=4;} //uint16_t
        else if (dtype_str == "<i1") {data_format_=5;} //int8_t
        else if (dtype_str == "<i2") {data_format_=6;} //int16_t
        else if (dtype_str == "<i4") {data_format_=7;} //int32_t
        else if (dtype_str == "<i8") {data_format_=8;} //int64_t
        else if (dtype_str == "<f2") {data_format_=9;} //float
        else if (dtype_str == "<f4") {data_format_=9;} //float
        else if (dtype_str == "<f8") {data_format_=10;} //double
        else {data_format_=2;} //uint16_t

        // Open the dataset once and cache the handle. The dataset metadata is
        // immutable for the lifetime of this loader, so there is no need to
        // re-open (and re-parse the .zarray metadata) on every tile read.
        ds_ = z5::openDataset(*zarr_ptr_, ds_name_);
    }

    /// @brief NyxusOmeZarrLoader destructor
    ~NyxusOmeZarrLoader() override
    {
        ds_ = nullptr;
        zarr_ptr_ = nullptr;
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
        size_t indexChannel,        // C plane to read (offset into the channel axis)
        size_t indexTimeframe,      // T plane to read (offset into the time axis)
        [[maybe_unused]] size_t level) override
    {
        size_t pixel_row_index = indexRowGlobalTile*tile_height_;
        size_t pixel_col_index = indexColGlobalTile*tile_width_;
        size_t pixel_layer_index = indexLayerGlobalTile*tile_depth_;


        switch (data_format_)
        {
        case 1:
            loadTile<uint8_t>(tile, pixel_row_index, pixel_col_index, pixel_layer_index, indexChannel, indexTimeframe);
            break;
        case 2:
            loadTile<uint16_t>(tile, pixel_row_index, pixel_col_index, pixel_layer_index, indexChannel, indexTimeframe);
            break;
        case 3:
            loadTile<uint32_t>(tile, pixel_row_index, pixel_col_index, pixel_layer_index, indexChannel, indexTimeframe);
            break;
        case 4:
            loadTile<uint64_t>(tile, pixel_row_index, pixel_col_index, pixel_layer_index, indexChannel, indexTimeframe);
            break;
        case 5:
            loadTile<int8_t>(tile, pixel_row_index, pixel_col_index, pixel_layer_index, indexChannel, indexTimeframe);
            break;
        case 6:
            loadTile<int16_t>(tile, pixel_row_index, pixel_col_index, pixel_layer_index, indexChannel, indexTimeframe);
            break;
        case 7:
            loadTile<int32_t>(tile, pixel_row_index, pixel_col_index, pixel_layer_index, indexChannel, indexTimeframe);
            break;
        case 8:
            loadTile<int64_t>(tile, pixel_row_index, pixel_col_index, pixel_layer_index, indexChannel, indexTimeframe);
            break;
        case 9:
            loadTile<float>(tile, pixel_row_index, pixel_col_index, pixel_layer_index, indexChannel, indexTimeframe);
            break;
        case 10:
            loadTile<double>(tile, pixel_row_index, pixel_col_index, pixel_layer_index, indexChannel, indexTimeframe);
            break;
        default:
            loadTile<uint16_t>(tile, pixel_row_index, pixel_col_index, pixel_layer_index, indexChannel, indexTimeframe);
            break;
        }
    }

    template<typename FileType>
    void loadTile(std::shared_ptr<std::vector<DataType>> &dest, size_t pixel_row_index,
                  size_t pixel_col_index, size_t pixel_layer_index,
                  size_t pixel_channel_index, size_t pixel_timeframe_index) {
        size_t data_height = tile_height_, data_width = tile_width_;
        if (pixel_row_index + data_height > full_height_) {
            data_height = full_height_ - pixel_row_index;
        }
        if (pixel_col_index + data_width > full_width_) {
            data_width = full_width_ - pixel_col_index;
        }

        // Create a buffer to hold the read data
        std::vector<FileType> buffer(data_height * data_width);

        // Build the read window by axis ROLE (honoring the resolved 'axes' order),
        // reading one Y*X plane at the requested Z/C/T. z5 3.0.1 uses ArrayView.
        z5::types::ShapeType shape(ndim_, 1), offset(ndim_, 0);
        shape[iy_] = data_height; offset[iy_] = pixel_row_index;
        shape[ix_] = data_width;  offset[ix_] = pixel_col_index;
        if (iz_ >= 0) offset[iz_] = pixel_layer_index;
        if (ic_ >= 0) offset[ic_] = pixel_channel_index;
        if (it_ >= 0) offset[it_] = pixel_timeframe_index;
        auto view = z5::multiarray::makeView(buffer.data(), shape);
        
        // Read subarray from the cached z5 dataset
        z5::multiarray::readSubarray<FileType>(*ds_, view, offset.begin());
        
        // Copy from buffer to destination tile, handling partial tiles
        for (size_t k = 0; k < data_height; ++k) {
            std::copy(buffer.begin() + k * data_width, 
                     buffer.begin() + (k + 1) * data_width, 
                     dest->begin() + k * tile_width_);
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

    short data_format_ = 0;
    std::unique_ptr<z5::filesystem::handle::File> zarr_ptr_;
    std::string ds_name_;
    std::unique_ptr<z5::Dataset> ds_;   ///< Cached dataset handle (opened once)
};
#endif //OMEZARR_SUPPORT
