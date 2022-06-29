#include <algorithm>
#include <regex>

#include <fast_loader/fast_loader.h> 
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
class NyxusOmeZarrLoader : public fl::AbstractTileLoader<fl::DefaultView<DataType>> 
{
public:

    /// @brief NyxusOmeZarrLoader constructor
    /// @param numberThreads Number of threads associated
    /// @param filePath Path of zar file
    /// @param tileWidth Tile width requested
    /// @param tileHeight Tile height requested
    /// @param tileDepth Tile depth requested
    NyxusOmeZarrLoader(
        size_t numberThreads,
        std::string const& filePath)
        : fl::AbstractTileLoader<fl::DefaultView<DataType>>("NyxusOmeZarrLoader", numberThreads, filePath)
    {
        short samplesPerPixel = 1;
        // Open the file
        zarr_ptr_ = std::make_unique<z5::filesystem::handle::File>(filePath.c_str());
        nlohmann::json attributes;
        z5::readAttributes(*zarr_ptr_, attributes);
        
        std::string metadata = attributes["metadata"].dump();
        std::regex type_regex("Type=\\\\\"(\\w+)");
        std::smatch matches;
        if(std::regex_search(metadata, matches, type_regex)) {
            if (matches[1].str() == "uint8") {data_format_=1;}
            else if (matches[1].str() == "uint16") {data_format_=2;}
            else if (matches[1].str() == "uint32") {data_format_=3;}
            else if (matches[1].str() == "uint64") {data_format_=4;}
            else if (matches[1].str() == "int8") {data_format_=5;}
            else if (matches[1].str() == "int16") {data_format_=6;}
            else if (matches[1].str() == "int32") {data_format_=7;}
            else if (matches[1].str() == "int64") {data_format_=8;}
            else if (matches[1].str() == "float") {data_format_=9;}
            else if (matches[1].str() == "double") {data_format_=10;}
            else {data_format_=2;}
        }
    

        std::regex size_regex("Size([X|Y|Z])=\\\\\"(\\d+)");
        
        while(std::regex_search(metadata, matches, size_regex)) {
            if (matches[1].str() == "X") {fullWidth_ = stoi(matches[2].str());}
            else if (matches[1].str() == "Y") {fullHeight_ = stoi(matches[2].str());}
            else if (matches[1].str() == "Z") {fullDepth_ = stoi(matches[2].str());}
            else {continue;}
            metadata = matches.suffix().str();
        }

        tileWidth_ = std::min(fullWidth_, size_t(1024));
        tileHeight_ = std::min(fullHeight_, size_t(1024));
        tileDepth_ = std::min(fullDepth_, size_t(1));
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
        size_t pixel_row_index = indexRowGlobalTile*tileHeight_;
        size_t pixel_col_index = indexColGlobalTile*tileWidth_;
        size_t pixel_layer_index = indexLayerGlobalTile*tileDepth_;

        
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
        
        size_t data_height = tileHeight_, data_width = tileWidth_;
        if (pixel_row_index + data_height > fullHeight_) {data_height = fullHeight_ - pixel_row_index;}
        if (pixel_col_index + data_width > fullWidth_) {data_width = fullWidth_ - pixel_col_index;}

        typename xt::xarray<FileType>::shape_type shape = {1,1,1,data_height,data_width };
        z5::types::ShapeType offset = { 0,0,pixel_layer_index, pixel_row_index, pixel_col_index };
        xt::xarray<FileType> array(shape);
        z5::multiarray::readSubarray<FileType>(ds, array, offset.begin());     
        std::vector<DataType> tmp = std::vector<DataType> (array.begin(), array.end());

        
        for (size_t k=0;k<data_height;++k)
        {
            std::copy(tmp.begin()+ k*data_width, tmp.begin()+(k+1)*data_width, dest->begin()+k*tileWidth_);
        }
        //*dest = std::vector<DataType> (array.begin(), array.end());
    }

    /// @brief Copy Method for the NyxusOmeZarrLoader
    /// @return Return a copy of the current NyxusOmeZarrLoader
    std::shared_ptr<fl::AbstractTileLoader<fl::DefaultView<DataType>>> copyTileLoader() override 
    {
        return std::make_shared<NyxusOmeZarrLoader<DataType>>(this->numberThreads(),this->filePath());
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
    [[nodiscard]] short bitsPerSample() const override { return 1; }
    /// @brief Level accessor
    /// @return 1
    [[nodiscard]] size_t numberPyramidLevels() const override { return 1; }

private:

    size_t
        fullHeight_ = 0,          ///< Full height in pixel
        fullWidth_ = 0,           ///< Full width in pixel
        fullDepth_ = 0,           ///< Full depth in pixel
        tileWidth_ = 0,           ///< Tile width
        tileHeight_ = 0,          ///< Tile height
        tileDepth_ = 0;           ///< Tile depth

    short data_format_ = 0;
    std::unique_ptr<z5::filesystem::handle::File> zarr_ptr_;
};
