#pragma once

#ifdef OMEZARR_SUPPORT

#include <algorithm>
#include <cmath>
#include <vector>
#include <stdexcept>
#include "abs_tile_loader.h"
#include "grey_level_cast.h"
#include "ome/ome_zarr_layout.h"   // open_zarr_level0() / ZarrLayout -- shared with raw_omezarr.h

// z5 multiarray API (ArrayView-based, no xtensor)
#include "z5/multiarray/array_view.hxx"
#include "z5/multiarray/array_access.hxx"

/// @brief Tile Loader for OMEZarr
/// @tparam DataType AbstractView's internal type
template<class DataType>
class NyxusOmeZarrLoader : public AbstractTileLoader<DataType>
{
public:

    /// @brief NyxusOmeZarrLoader constructor
    /// @param numberThreads Number of threads associated
    /// @param filePath Path of zarr file
    /// @param _inten_offset Offset of the load-time map (SlideProps::inten_offset)
    /// @param _inten_max Upper end of the clamp, used only when _quantize is set
    /// @param _target_dyn_range Grey levels the rescale spans, used only when _quantize is set
    /// @param _quantize Min-max rescale a real-valued dataset, instead of the offset map
    NyxusOmeZarrLoader(
        size_t numberThreads,
        std::string const& filePath,
        double _inten_offset = 0.0,
        double _inten_max = 1.0,
        double _target_dyn_range = 1e4,
        bool _quantize = false,
        bool _round_offset = false)
        : AbstractTileLoader<DataType>("NyxusOmeZarrLoader", numberThreads, filePath),
        inten_offset_(_inten_offset),
        inten_max_(_inten_max),
        target_dyn_range_(_target_dyn_range),
        quantize_(_quantize),
        round_offset_(_round_offset)
    {
        // Open the level-0 array once and cache the handle: its metadata is immutable for the
        // lifetime of this loader. Axis roles, extents, chunking and pixel type come from the
        // NGFF 'axes' metadata (shared with RawOmezarrLoader -- see ome/ome_zarr_layout.h).
        zarr_ptr_ = std::make_unique<z5::filesystem::handle::File>(filePath.c_str());
        ds_ = Nyxus::open_zarr_level0 (*zarr_ptr_, layout_);
    }

    /// @brief NyxusOmeZarrLoader destructor
    ~NyxusOmeZarrLoader() override
    {
        ds_ = nullptr;
        zarr_ptr_ = nullptr;
    }

    /// @brief Load one chunk of plane (c,t)
    /// @param tile Tile to copy into, tileDepth() planes of tileHeight() x tileWidth()
    /// @param indexRowGlobalTile Tile row index
    /// @param indexColGlobalTile Tile column index
    /// @param indexLayerGlobalTile Tile layer index; a chunk may span several Z-planes
    /// @param level Tile's level
    void loadTileFromFile(std::shared_ptr<std::vector<DataType>> tile,
        size_t indexRowGlobalTile,
        size_t indexColGlobalTile,
        size_t indexLayerGlobalTile,
        size_t indexChannel,        // C plane to read (offset into the channel axis)
        size_t indexTimeframe,      // T plane to read (offset into the time axis)
        [[maybe_unused]] size_t level) override
    {
        const Nyxus::ZarrBlock b = layout_.block_at (indexRowGlobalTile, indexColGlobalTile, indexLayerGlobalTile, indexChannel, indexTimeframe);
        Nyxus::with_zarr_sample_type (layout_.dtype, [&] (auto sample)
            {
                this->template loadTile<decltype(sample)> (*tile, b);
            });
    }

    template<typename FileType>
    void loadTile (std::vector<DataType>& dest, const Nyxus::ZarrBlock& b)
    {
        std::vector<FileType> buffer (b.depth * b.height * b.width);
        auto view = z5::multiarray::makeView (buffer.data(), b.shape);
        z5::multiarray::readSubarray<FileType> (*ds_, view, b.offset.begin());

        // dest is plane-major: plane p, row k at (p*tile_height + k)*tile_width, matching the
        // stride ImageLoader::assemble_tile_layer reads. The sample goes through the same load-time
        // map every other backend applies, so a signed dataset keeps its negatives and a
        // real-valued one its fraction.
        const size_t th = layout_.tile_height, tw = layout_.tile_width;
        for (size_t p = 0; p < b.depth; ++p)
            for (size_t k = 0; k < b.height; ++k)
                for (size_t j = 0; j < b.width; ++j)
                    dest[(p * th + k) * tw + j] = map_intensity ((double) buffer[(p * b.height + k) * b.width + j]);
    }

    /// @brief Tiff file height
    /// @param level Tiff level [not used]
    /// @return Full height
    [[nodiscard]] size_t fullHeight([[maybe_unused]] size_t level) const override { return layout_.full_height; }
    /// @brief Tiff full width
    /// @param level Tiff level [not used]
    /// @return Full width
    [[nodiscard]] size_t fullWidth([[maybe_unused]] size_t level) const override { return layout_.full_width; }
    /// @brief Tiff full depth
    /// @param level Tiff level [not used]
    /// @return Full Depth
    [[nodiscard]] size_t fullDepth([[maybe_unused]] size_t level) const override { return layout_.full_depth; }

    /// @brief Tiff tile width
    /// @param level Tiff level [not used]
    /// @return Tile width
    [[nodiscard]] size_t tileWidth([[maybe_unused]] size_t level) const override { return layout_.tile_width; }
    /// @brief Tiff tile height
    /// @param level Tiff level [not used]
    /// @return Tile height
    [[nodiscard]] size_t tileHeight([[maybe_unused]] size_t level) const override { return layout_.tile_height; }
    /// @brief Tiff tile depth
    /// @param level Tiff level [not used]
    /// @return Tile depth
    [[nodiscard]] size_t tileDepth([[maybe_unused]] size_t level) const override { return layout_.tile_depth; }

    /// @brief Bits per sample (resolved from the dataset dtype)
    [[nodiscard]] short bitsPerSample() const override { return layout_.bits_per_sample; }
    /// @brief Number of resolution (pyramid) levels declared in multiscales
    [[nodiscard]] size_t numberPyramidLevels() const override { return layout_.n_levels; }
    /// @brief Channel (C) extent resolved from the NGFF axes (1 if no channel axis)
    [[nodiscard]] size_t numberChannels() const override { return layout_.n_channels; }
    /// @brief Time (T) extent resolved from the NGFF axes (1 if no time axis)
    [[nodiscard]] size_t fullTimestamps([[maybe_unused]] size_t level) const override { return layout_.n_timeframes; }
    /// @brief Physical voxel spacing from the NGFF coordinateTransformations (1.0 if uncalibrated)
    [[nodiscard]] double physicalSizeX() const override { return layout_.phys_x; }
    [[nodiscard]] double physicalSizeY() const override { return layout_.phys_y; }
    [[nodiscard]] double physicalSizeZ() const override { return layout_.phys_z; }
    [[nodiscard]] std::string physicalSizeUnit() const override { return layout_.phys_unit; }

private:

    Nyxus::ZarrLayout layout_;          ///< Axis roles, extents, chunking, pixel type and spacing of the level-0 array
    std::unique_ptr<z5::filesystem::handle::File> zarr_ptr_;
    std::unique_ptr<z5::Dataset> ds_;   ///< Cached dataset handle (opened once)

    double inten_offset_ = 0.0,
        inten_max_ = 1.0,
        target_dyn_range_ = 1e4;

    // Whether this dataset is min-max rescaled into [0, target_dyn_range] (a real-valued dataset
    // left in its default mode) or carried on the offset map. SlideProps::inten_map is where the
    // choice is made and recorded; ImageLoader::open passes it here, exactly as it does for TIFF.
    bool quantize_ = false;
    bool round_offset_ = false;		// offset map: round to nearest (--preserve-hu) instead of truncating

    // The offset map, shared by the real-valued and native-integer paths, keeping 1 grey level ==
    // 1 intensity unit. The intensity families add the offset back, so reported statistics are in
    // the dataset's own domain. A mask is opened with offset 0 and no quantize, which leaves its
    // labels untouched. Both branches narrow through the Nyxus::grey_level* the TIFF loaders use,
    // so no backend can drift from another.
    DataType map_intensity (double x) const
    {
        if (! quantize_)
            return Nyxus::grey_level<DataType> (x - inten_offset_, round_offset_);
        // Ahead of the clamp, as in the TIFF loaders: the clamp would store +Inf as the top level.
        if (! std::isfinite (x)) return (DataType) 0;
        double t = x < inten_offset_ ? inten_offset_ : x;
        t = t > inten_max_ ? inten_max_ : t;
        return Nyxus::grey_level_truncated<DataType> (target_dyn_range_ * (t - inten_offset_) / (inten_max_ - inten_offset_));
    }
};
#endif //OMEZARR_SUPPORT
