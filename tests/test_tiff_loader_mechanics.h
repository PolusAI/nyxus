#pragma once
//
// Regression test for the TIFF strip loader fix in src/nyx/grayscale_tiff.h:
// the 32-bit unsigned sample case used copyRow<size_t> (8 bytes on LP64/Win64)
// for a 4-byte sample, so copyRow read ((size_t*)buf)[col] -- 2x past the libtiff
// scanline buffer -> corrupted/garbage pixels or a heap over-read (crash). The
// fix is copyRow<uint32_t>.
//
// This drives the exact loader that carried the bug: it writes a uint32 strip
// TIFF to disk with libtiff and reads it back through
// NyxusGrayscaleTiffStripLoader<uint32_t>, asserting every pixel round-trips.
// The intensities span past the 16-bit range and past 2^31 (up to UINT32_MAX) so
// a wrong element size / truncation / column shift is detectable.
//
#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <tiffio.h>

#include "../src/nyx/grayscale_tiff.h"

// Write a strip-based (non-tiled), single-channel uint32 grayscale TIFF.
static inline void nyxus_ut_write_uint32_strip_tiff(
    const std::string& path, const std::vector<uint32_t>& px, uint32_t w, uint32_t h)
{
    TIFF* t = TIFFOpen(path.c_str(), "w");
    if (t == nullptr)
        throw std::runtime_error("could not create TIFF: " + path);

    TIFFSetField(t, TIFFTAG_IMAGEWIDTH, w);
    TIFFSetField(t, TIFFTAG_IMAGELENGTH, h);
    TIFFSetField(t, TIFFTAG_BITSPERSAMPLE, 32);
    TIFFSetField(t, TIFFTAG_SAMPLESPERPIXEL, 1);
    TIFFSetField(t, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_UINT);   // -> loader sampleFormat_ == 1
    TIFFSetField(t, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
    TIFFSetField(t, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
    TIFFSetField(t, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);
    TIFFSetField(t, TIFFTAG_ROWSPERSTRIP, 1);                   // strip (scanline) layout, not tiled

    std::vector<uint32_t> row(w);
    for (uint32_t y = 0; y < h; ++y)
    {
        std::copy(px.begin() + static_cast<size_t>(y) * w,
                  px.begin() + static_cast<size_t>(y + 1) * w, row.begin());
        if (TIFFWriteScanline(t, row.data(), y, 0) < 0)
        {
            TIFFClose(t);
            throw std::runtime_error("TIFFWriteScanline failed");
        }
    }
    TIFFClose(t);
}

inline void test_uint32_strip_tiff_loader()
{
    const uint32_t w = 5, h = 3;
    // Values past 16-bit and past 2^31 (incl. UINT32_MAX) so a mis-typed read shows up.
    const std::vector<uint32_t> expected = {
        100u,         70000u,       300000u,       16777216u,   4000000000u,
        5u,           123456789u,   3000000000u,   65535u,      65536u,
        2147483648u,  1u,           4294967295u,   2u,          268435456u
    };

    const auto path =
        (std::filesystem::temp_directory_path() / "nyxus_ut_uint32_strip.tif").string();
    nyxus_ut_write_uint32_strip_tiff(path, expected, w, h);

    try
    {
        NyxusGrayscaleTiffStripLoader<uint32_t> loader(1, path);

        if (loader.fullWidth(0) != w || loader.fullHeight(0) != h)
            throw std::runtime_error("TIFF header dimensions mismatch");

        const size_t tw = loader.tileWidth(0),
                     th = loader.tileHeight(0),
                     td = loader.tileDepth(0);
        auto tile = std::make_shared<std::vector<uint32_t>>(tw * th * td);

        // Single tile at (row,col,layer,level) = (0,0,0,0) covers this small image.
        loader.loadTileFromFile(tile, 0, 0, 0, 0);

        for (uint32_t y = 0; y < h; ++y)
            for (uint32_t x = 0; x < w; ++x)
            {
                const uint32_t got = (*tile)[static_cast<size_t>(y) * tw + x];
                const uint32_t exp = expected[static_cast<size_t>(y) * w + x];
                if (got != exp)
                    throw std::runtime_error(
                        "pixel (" + std::to_string(x) + "," + std::to_string(y) +
                        ") = " + std::to_string(got) + ", expected " + std::to_string(exp));
            }
    }
    catch (...)
    {
        std::error_code ec;
        std::filesystem::remove(path, ec);   // best-effort cleanup, then rethrow
        throw;
    }

    std::error_code ec;
    std::filesystem::remove(path, ec);
}
