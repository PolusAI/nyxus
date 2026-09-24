#pragma once

#ifdef OMEZARR_SUPPORT

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>
#include <stdexcept>
#include "grey_level_cast.h"
#include "ome/ome_zarr_layout.h"   // open_zarr_level0() / ZarrLayout -- shared with omezarr.h

// z5 multiarray API (ArrayView-based, no xtensor)
#include "z5/multiarray/array_view.hxx"
#include "z5/multiarray/array_access.hxx"

#include "raw_format.h"

class RawOmezarrLoader: public RawFormatLoader
{
public:

    RawOmezarrLoader (std::string const& filePath): RawFormatLoader("RawOmezarrLoader", filePath)
    {
        // Open the level-0 array once and cache the handle: its metadata is immutable for the
        // lifetime of this loader. Axis roles, extents, chunking and pixel type come from the
        // NGFF 'axes' metadata (shared with NyxusOmeZarrLoader -- see ome/ome_zarr_layout.h).
        zarr_ptr_ = std::make_unique<z5::filesystem::handle::File>(filePath.c_str());
        ds_ = Nyxus::open_zarr_level0 (*zarr_ptr_, layout_);
        fp_pixels_ = layout_.fp_pixels;

        // The buffer holds the sample as the file states it, widened to double, so the prescan
        // measures a signed or real-valued dataset's extrema on the values the file holds. It
        // spans a whole chunk, tileDepth() planes deep.
        dest = std::vector<double> (layout_.tile_height * layout_.tile_width * layout_.tile_depth);
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
        const Nyxus::ZarrBlock b = layout_.block_at (indexRowGlobalTile, indexColGlobalTile, indexLayerGlobalTile, indexChannel, indexTimeframe);
        Nyxus::with_zarr_sample_type (layout_.dtype, [&] (auto sample)
            {
                loadTile<decltype(sample)> (b);
            });
    }

    void free_tile() override
    {
    }

    // The buffer holds the sample as the file states it, so a negative, non-finite or
    // above-UINT32_MAX value reaches this accessor, and converting any of them to an unsigned
    // integer is undefined. Nyxus::grey_level_truncated() zeroes the first two and saturates the
    // third -- the same narrowing every load-time map now shares, rather than a clamp this accessor
    // carries alone. That, not the fact that only a mask is read through it today, is what makes
    // it safe. It truncates because it reads integer labels, not intensities.
    uint32_t get_uint32_pixel (size_t idx) const
    {
        return Nyxus::grey_level_truncated<uint32_t> (dest[idx]);
    }

    double get_dpequiv_pixel (size_t idx) const
    {
        double rv = dest[idx];
        return rv;
    }

    template<typename FileType>
    void loadTile (const Nyxus::ZarrBlock& b)
    {
        std::vector<FileType> buffer (b.depth * b.height * b.width);
        auto view = z5::multiarray::makeView (buffer.data(), b.shape);
        z5::multiarray::readSubarray<FileType> (*ds_, view, b.offset.begin());

        // zero-fill the buffer foreseeing its partial filling at incomplete (tail) tiles
        std::fill (dest.begin(), dest.end(), 0);

        // dest is plane-major: plane p, row k at (p*tile_height + k)*tile_width. The sample is
        // widened to double rather than narrowed to an unsigned grey level, so a negative or
        // fractional dataset reaches the prescan as it is written.
        const size_t th = layout_.tile_height, tw = layout_.tile_width;
        for (size_t p = 0; p < b.depth; ++p)
            for (size_t k = 0; k < b.height; ++k)
                for (size_t j = 0; j < b.width; ++j)
                    dest[(p * th + k) * tw + j] = static_cast<double> (buffer[(p * b.height + k) * b.width + j]);
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

    std::vector<double> dest;
};
#endif //OMEZARR_SUPPORT
