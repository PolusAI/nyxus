#pragma once

#ifdef OMEZARR_SUPPORT

#include <algorithm>
#include "abs_tile_loader.h"
#include "nlohmann/json.hpp"
#include "xtensor/xarray.hpp"

// factory functions to create files, groups and datasets
#include "z5/factory.hxx"
// handles for z5 filesystem objects
#include "z5/filesystem/handle.hxx"
// io for xtensor multi-arrays
#include "z5/multiarray/xtensor_access.hxx"
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

         // assume only one dataset is present
        std::string ds_name = file_attributes["multiscales"][0]["datasets"][0]["path"].get<std::string>();
        const auto ds_handle = z5::filesystem::handle::Dataset(*zarr_ptr_, ds_name);
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
    }

    /// @brief NyxusOmeZarrLoader destructor
    ~NyxusOmeZarrLoader() override
    {
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
        [[maybe_unused]] size_t level) override
    {
        size_t pixel_row_index = indexRowGlobalTile*tile_height_;
        size_t pixel_col_index = indexColGlobalTile*tile_width_;
        size_t pixel_layer_index = indexLayerGlobalTile*tile_depth_;


        switch (data_format_)
        {
        case 1:
            loadTile<uint8_t>(tile, pixel_row_index, pixel_col_index, pixel_layer_index);
            break;
        case 2:
            loadTile<uint16_t>(tile, pixel_row_index, pixel_col_index, pixel_layer_index);
            break;
        case 3:
            loadTile<uint32_t>(tile, pixel_row_index, pixel_col_index, pixel_layer_index);
            break;
        case 4:
            loadTile<uint64_t>(tile, pixel_row_index, pixel_col_index, pixel_layer_index);
            break;
        case 5:
            loadTile<int8_t>(tile, pixel_row_index, pixel_col_index, pixel_layer_index);
            break;
        case 6:
            loadTile<int16_t>(tile, pixel_row_index, pixel_col_index, pixel_layer_index);
            break;
        case 7:
            loadTile<int32_t>(tile, pixel_row_index, pixel_col_index, pixel_layer_index);
            break;
        case 8:
            loadTile<int64_t>(tile, pixel_row_index, pixel_col_index, pixel_layer_index);
            break;
        case 9:
            loadTile<float>(tile, pixel_row_index, pixel_col_index, pixel_layer_index);
            break;
        case 10:
            loadTile<double>(tile, pixel_row_index, pixel_col_index, pixel_layer_index);
            break;
        default:
            loadTile<uint16_t>(tile, pixel_row_index, pixel_col_index, pixel_layer_index);
            break;
        }
    }

    template<typename FileType>
    void loadTile(std::shared_ptr<std::vector<DataType>> &dest, size_t pixel_row_index, size_t pixel_col_index, size_t pixel_layer_index){
        std::vector<std::string> datasets;
        zarr_ptr_->keys(datasets);
        auto ds = z5::openDataset(*zarr_ptr_, datasets[0]);

        size_t data_height = tile_height_, data_width = tile_width_;
        if (pixel_row_index + data_height > full_height_) {data_height = full_height_ - pixel_row_index;}
        if (pixel_col_index + data_width > full_width_) {data_width = full_width_ - pixel_col_index;}

        typename xt::xarray<FileType>::shape_type shape = {1,1,1,data_height,data_width };
        z5::types::ShapeType offset = { 0,0,pixel_layer_index, pixel_row_index, pixel_col_index };
        xt::xarray<FileType> array(shape);
        z5::multiarray::readSubarray<FileType>(ds, array, offset.begin());
        std::vector<DataType> tmp = std::vector<DataType> (array.begin(), array.end());


        for (size_t k=0;k<data_height;++k)
        {
            std::copy(tmp.begin()+ k*data_width, tmp.begin()+(k+1)*data_width, dest->begin()+k*tile_width_);
        }
        //*dest = std::vector<DataType> (array.begin(), array.end());
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
};
#endif //OMEZARR_SUPPORT
