#pragma once

#ifdef OMEZARR_SUPPORT

#include <algorithm>
#include <vector>
#include <stdexcept>
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

#include "raw_format.h"

// FIX (IO): z5 Datatype -> the "<u2"-style dtype string this loader's dispatch expects
// (see omezarr.h). Guarded so a TU including both loaders doesn't redefine it.
#ifndef NYXUS_ZARR_DTYPE_STRING_OF
#define NYXUS_ZARR_DTYPE_STRING_OF
inline std::string zarr_dtype_string_of (z5::types::Datatype dt)
{
    switch (dt)
    {
        case z5::types::uint8:   return "<u1";
        case z5::types::uint16:  return "<u2";
        case z5::types::uint32:  return "<u4";
        case z5::types::uint64:  return "<u8";
        case z5::types::int8:    return "<i1";
        case z5::types::int16:   return "<i2";
        case z5::types::int32:   return "<i4";
        case z5::types::int64:   return "<i8";
        case z5::types::float32: return "<f4";
        case z5::types::float64: return "<f8";
        default:                 return "<u2";
    }
}
#endif // NYXUS_ZARR_DTYPE_STRING_OF

class RawOmezarrLoader: public RawFormatLoader
{
public:

    RawOmezarrLoader (std::string const& filePath): RawFormatLoader("RawOmezarrLoader", filePath)
    {
        // Open the file
        zarr_ptr_ = std::make_unique<z5::filesystem::handle::File>(filePath.c_str());
        nlohmann::json file_attributes;
        z5::readAttributes(*zarr_ptr_, file_attributes);

        // Resolve the level-0 dataset path (NGFF 0.5 nests 'multiscales' under "ome"; 0.4 has
        // it at the attributes root).
        const nlohmann::json& ms_root = (file_attributes.contains("ome") && file_attributes["ome"].contains("multiscales"))
            ? file_attributes["ome"] : file_attributes;
        ds_name_ = ms_root["multiscales"][0]["datasets"][0]["path"].get<std::string>();

        // FIX (IO): open via z5 (auto-detects Zarr v2 .zarray vs v3 zarr.json, handles v3
        // chunk-key encoding, codecs and sharding) and query shape/chunking/dtype from the
        // Dataset object -- format-agnostic, so this reads BOTH OME-Zarr 0.4 and 0.5.
        ds_ = z5::openDataset(*zarr_ptr_, ds_name_);
        std::vector<size_t> level0Shape(ds_->shape().begin(), ds_->shape().end());
        std::vector<size_t> chunkShape(ds_->defaultChunkShape().begin(), ds_->defaultChunkShape().end());
        std::string dtype_str = zarr_dtype_string_of(ds_->getDtype());

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
            // FIX: advertise the real C/T extents so the pipeline iterates channels and
            // timeframes; without this the base class default of 1 kept it on plane (c=0,t=0).
            n_channels_ = axes.sizeC;
            n_timeframes_ = axes.sizeT;
            // FIX (IO): keep the parsed physical voxel spacing for opt-in calibration.
            phys_x_ = axes.physX; phys_y_ = axes.physY; phys_z_ = axes.physZ;
            phys_unit_ = axes.unitXY;
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
            // FIX: derive C/T extents from the positional axes so the fallback path also
            // reports multi-channel / time-series counts (1 when the axis is absent).
            n_channels_ = (ic_ >= 0) ? level0Shape[ic_] : 1;
            n_timeframes_ = (it_ >= 0) ? level0Shape[it_] : 1;
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
