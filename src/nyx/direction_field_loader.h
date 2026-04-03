#pragma once

#ifdef __APPLE__
    #define uint64 uint64_hack_
    #define int64 int64_hack_
    #include <tiffio.h>
    #undef uint64
    #undef int64
#else
    #include <tiffio.h>
#endif

#include <string>
#include <memory>
#include <cmath>
#include <iostream>
#include <algorithm>
#include "features/image_cube.h"
#include "direction_field_loader_nifti.h"

/// @brief Loader for multi-channel TIFF/NIfTI direction field images
/// 
/// PURPOSE: This class provides a unified interface to load direction fields from different file formats.
/// Direction fields tell GLCM which direction to look when computing texture at each pixel.
/// 
/// WHAT IS A DIRECTION FIELD?
/// - Traditional GLCM uses fixed angles (0°, 45°, 90°, 135°) everywhere
/// - Direction field uses DIFFERENT angles at EACH pixel location
/// - Stored as vectors: [dx, dy] for 2D or [dx, dy, dz] for 3D
/// - Example: In a fiber image, direction field follows fiber orientation
///
/// FILE FORMAT REQUIREMENTS:
/// - TIFF: Multi-channel image with 2-3 channels (dx, dy[, dz])
/// - NIfTI: 4D image where 4th dimension holds vector components
/// - Values: float32 or float64
class DirectionFieldLoader
{
public:
    /// @brief Main entry point - Load direction field from file
    /// 
    /// PURPOSE: Auto-detect file format and dispatch to appropriate loader
    /// 
    /// @param filePath Full path to the direction field file
    /// 
    /// @return SimpleCube<float> containing direction vectors
    ///         Shape: (width, height, num_channels)
    ///         where num_channels = 2 for 2D (dx,dy) or 3 for 3D (dx,dy,dz)
    /// 
    /// EXAMPLE USAGE:
    ///   auto dirField = DirectionFieldLoader::load("myfield.tif", false);
    ///   float dx = dirField->zyx(0, row, col);  // x-component at (row, col)
    ///   float dy = dirField->zyx(1, row, col);  // y-component at (row, col)
    static std::unique_ptr<SimpleCube<float>> load(const std::string& filePath)
    {
        // Auto-detect file format based on extension
        std::string ext = getFileExtension(filePath);
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        
        if (ext == ".nii" || ext == ".gz")
        {
            // NIfTI format - common in medical imaging
            return DirectionFieldLoaderNifti::load(filePath);
        }
        else if (ext == ".tif" || ext == ".tiff")
        {
            // TIFF format - common in microscopy
            return loadTiff(filePath);
        }
        else
        {
            throw std::runtime_error("DirectionFieldLoader: Unsupported file format '" + ext + "'. Expected .tif, .tiff, .nii, or .nii.gz");
        }
    }

private:
    
    /// @brief Extract file extension from path
    /// 
    /// PURPOSE: Determine file type so we know which loader to use
    /// 
    /// @param filePath Full path like "/path/to/file.tif"
    /// @return Extension including dot, like ".tif" or ".gz"
    /// 
    /// WHY PRIVATE? This is just a helper function - users don't need to call it directly
    static std::string getFileExtension(const std::string& filePath)
    {
        size_t dotPos = filePath.find_last_of('.');
        if (dotPos == std::string::npos)
            return "";
        return filePath.substr(dotPos);
    }
    
    /// @brief Load direction field from a multi-channel TIFF file
    /// 
    /// PURPOSE: Read TIFF files that store direction vectors
    /// 
    /// TIFF FORMAT DETAILS:
    /// - Must have 2-3 "samples per pixel" (channels)
    /// - Channel 0 = dx (x-component of direction)
    /// - Channel 1 = dy (y-component of direction)  
    /// - Channel 2 = dz (z-component, optional, for 3D)
    /// - Data type: float32 or float64
    /// - Can be tiled or strip-based (we handle both)
    /// 
    /// @param filePath Path to TIFF file
    /// @return SimpleCube with loaded
    /// 
    /// WHY PRIVATE? Implementation detail - users call load(), not this directly
    static std::unique_ptr<SimpleCube<float>> loadTiff(const std::string& filePath)
    {
        // Open TIFF file using libtiff library
        TIFF* tiff = TIFFOpen(filePath.c_str(), "r");
        if (!tiff)
        {
            throw std::runtime_error("DirectionFieldLoader: Cannot open file " + filePath);
        }

        // Read TIFF metadata (image properties)
        uint32_t width, height;              // Image dimensions
        uint16_t samplesPerPixel;            // Number of channels (should be 2 or 3)
        uint16_t bitsPerSample;              // Bits per value (should be 32 or 64 for float)
        uint16_t sampleFormat;               // Data type (should be IEEE float)
        
        TIFFGetField(tiff, TIFFTAG_IMAGEWIDTH, &width);
        TIFFGetField(tiff, TIFFTAG_IMAGELENGTH, &height);
        TIFFGetField(tiff, TIFFTAG_SAMPLESPERPIXEL, &samplesPerPixel);
        TIFFGetField(tiff, TIFFTAG_BITSPERSAMPLE, &bitsPerSample);
        TIFFGetField(tiff, TIFFTAG_SAMPLEFORMAT, &sampleFormat);

        // Validate: Must have 2 channels (dx,dy) or 3 channels (dx,dy,dz)
        if (samplesPerPixel < 2 || samplesPerPixel > 3)
        {
            TIFFClose(tiff);
            throw std::runtime_error(
                "DirectionFieldLoader: Expected 2 or 3 channels, got " + 
                std::to_string(samplesPerPixel) + " channels"
            );
        }

        // Validate: Data must be floating-point (not integer)
        if (sampleFormat != SAMPLEFORMAT_IEEEFP)
        {
            TIFFClose(tiff);
            throw std::runtime_error(
                "DirectionFieldLoader: Expected floating-point data (SAMPLEFORMAT_IEEEFP), got format " + 
                std::to_string(sampleFormat)
            );
        }

        // Validate: Must be 32-bit or 64-bit float
        if (bitsPerSample != 32 && bitsPerSample != 64)
        {
            TIFFClose(tiff);
            throw std::runtime_error(
                "DirectionFieldLoader: Expected 32 or 64 bits per sample, got " + 
                std::to_string(bitsPerSample)
            );
        }

        // Allocate output cube to hold the direction vectors
        auto directionField = std::make_unique<SimpleCube<float>>(width, height, samplesPerPixel);

        // TIFF images can be organized two ways:
        // 1. Tiled - image divided into rectangular tiles (faster random access)
        // 2. Strip - image stored as horizontal strips (simpler, sequential)
        bool isTiled = TIFFIsTiled(tiff);
        
        if (isTiled)
        {
            // Read tiled image
            loadTiledImage(tiff, width, height, samplesPerPixel, bitsPerSample, *directionField);
        }
        else
        {
            // Read strip-based image (most common)
            loadStripImage(tiff, width, height, samplesPerPixel, bitsPerSample, *directionField);
        }

        TIFFClose(tiff);

        return directionField;
    }

    /// @brief Load a tiled TIFF image
    /// 
    /// PURPOSE: Handle TIFF files organized as rectangular tiles
    /// 
    /// WHAT ARE TILES?
    /// - Image is divided into rectangular blocks (e.g., 256x256 pixels each)
    /// - Allows efficient random access to image regions
    /// - Common in large microscopy images
    /// 
    /// HOW IT WORKS:
    /// 1. Read each tile from file
    /// 2. Copy relevant pixels from tile to output cube
    /// 3. Handle edge tiles that may be partially outside image bounds
    /// 
    /// @param tiff Open TIFF file handle
    /// @param width Image width in pixels
    /// @param height Image height in pixels
    /// @param numChannels Number of vector components (2 or 3)
    /// @param bitsPerSample Bits per channel value (32 or 64)
    /// @param directionField Output cube to fill with data
    static void loadTiledImage(
        TIFF* tiff,
        uint32_t width,
        uint32_t height,
        uint16_t numChannels,
        uint16_t bitsPerSample,
        SimpleCube<float>& directionField)
    {
        // Get tile dimensions (how big each tile is)
        uint32_t tileWidth, tileHeight;
        TIFFGetField(tiff, TIFFTAG_TILEWIDTH, &tileWidth);
        TIFFGetField(tiff, TIFFTAG_TILELENGTH, &tileHeight);

        // Allocate buffer to hold one tile
        size_t tileSize = TIFFTileSize(tiff);
        tdata_t buf = _TIFFmalloc(tileSize);
        
        if (!buf)
        {
            throw std::runtime_error("DirectionFieldLoader: Failed to allocate tile buffer");
        }

        // Iterate over all tiles, row by row
        for (uint32_t row = 0; row < height; row += tileHeight)
        {
            for (uint32_t col = 0; col < width; col += tileWidth)
            {
                // Read one tile from file into buffer
                if (TIFFReadTile(tiff, buf, col, row, 0, 0) < 0)
                {
                    _TIFFfree(buf);
                    throw std::runtime_error("DirectionFieldLoader: Error reading tile");
                }

                // Copy data from tile buffer to output cube
                copyTileData(
                    buf, 
                    bitsPerSample,
                    directionField,
                    row, col, 
                    tileHeight, tileWidth,
                    height, width,
                    numChannels
                );
            }
        }

        _TIFFfree(buf);
    }

    /// @brief Load a strip-based TIFF image
    /// 
    /// PURPOSE: Handle TIFF files organized as horizontal strips (scanlines)
    /// 
    /// WHAT ARE STRIPS?
    /// - Image stored as horizontal rows (scanlines)
    /// - Simple, sequential storage
    /// - Most common TIFF organization
    /// 
    /// HOW IT WORKS:
    /// 1. Read one row at a time
    /// 2. Copy pixels from row buffer to output cube
    /// 
    /// @param tiff Open TIFF file handle
    /// @param width Image width in pixels
    /// @param height Image height in pixels
    /// @param numChannels Number of vector components (2 or 3)
    /// @param bitsPerSample Bits per channel value (32 or 64)
    /// @param directionField Output cube to fill with data
    static void loadStripImage(
        TIFF* tiff,
        uint32_t width,
        uint32_t height,
        uint16_t numChannels,
        uint16_t bitsPerSample,
        SimpleCube<float>& directionField)
    {
        // Allocate buffer to hold one row of pixels
        size_t scanlineSize = TIFFScanlineSize(tiff);
        tdata_t buf = _TIFFmalloc(scanlineSize);
        
        if (!buf)
        {
            throw std::runtime_error("DirectionFieldLoader: Failed to allocate scanline buffer");
        }

        // Read image one row at a time
        for (uint32_t row = 0; row < height; row++)
        {
            // Read one row into buffer
            if (TIFFReadScanline(tiff, buf, row) < 0)
            {
                _TIFFfree(buf);
                throw std::runtime_error("DirectionFieldLoader: Error reading scanline");
            }

            // Copy row data to output cube
            copyScanlineData(
                buf,
                bitsPerSample,
                directionField,
                row,
                width,
                numChannels
            );
        }

        _TIFFfree(buf);
    }

    /// @brief Copy one tile's data to the output cube
    /// 
    /// PURPOSE: Transfer pixels from tile buffer to SimpleCube storage
    /// 
    /// WHY NEEDED?
    /// - Tile buffer is raw bytes in TIFF format
    /// - SimpleCube uses specific indexing (zyx)
    /// - Need to handle edge tiles that extend beyond image bounds
    /// - Need to convert between float32/float64
    /// 
    /// MEMORY LAYOUT:
    /// - Tile buffer: [pixel0_ch0, pixel0_ch1, ..., pixel1_ch0, pixel1_ch1, ...]
    /// - SimpleCube: Accessed via zyx(channel, row, col)
    /// 
    /// @param buf Raw tile data buffer
    /// @param bitsPerSample 32 or 64 (determines if buf is float* or double*)
    /// @param directionField Output cube
    /// @param startRow Starting row of this tile in the image
    /// @param startCol Starting column of this tile in the image
    /// @param tileHeight Height of tile
    /// @param tileWidth Width of tile
    /// @param imageHeight Total image height (for bounds checking)
    /// @param imageWidth Total image width (for bounds checking)
    /// @param numChannels Number of components per pixel
    static void copyTileData(
        tdata_t buf,
        uint16_t bitsPerSample,
        SimpleCube<float>& directionField,
        uint32_t startRow,
        uint32_t startCol,
        uint32_t tileHeight,
        uint32_t tileWidth,
        uint32_t imageHeight,
        uint32_t imageWidth,
        uint16_t numChannels)
    {
        // Calculate actual bounds (edge tiles may extend past image)
        uint32_t endRow = std::min(startRow + tileHeight, imageHeight);
        uint32_t endCol = std::min(startCol + tileWidth, imageWidth);

        // Iterate over pixels in this tile
        for (uint32_t row = startRow; row < endRow; row++)
        {
            for (uint32_t col = startCol; col < endCol; col++)
            {
                // Calculate index in tile buffer
                // Tile stores pixels sequentially: row0, row1, ...
                size_t tileIdx = ((row - startRow) * tileWidth + (col - startCol)) * numChannels;
                
                // Copy each channel (dx, dy, [dz])
                for (uint16_t ch = 0; ch < numChannels; ch++)
                {
                    float value;
                    if (bitsPerSample == 32)
                    {
                        // Buffer contains float32 values
                        value = static_cast<float*>(buf)[tileIdx + ch];
                    }
                    else // 64-bit
                    {
                        // Buffer contains float64 values - convert to float32
                        value = static_cast<float>(static_cast<double*>(buf)[tileIdx + ch]);
                    }
                    
                    // Store in SimpleCube
                    // zyx indexing: z=channel, y=row, x=col
                    directionField.zyx(ch, row, col) = value;
                }
            }
        }
    }

    /// @brief Copy one scanline's data to the output cube
    /// 
    /// PURPOSE: Transfer pixels from scanline buffer to SimpleCube storage
    /// 
    /// Similar to copyTileData but simpler because we're copying a full row
    /// 
    /// @param buf Raw scanline data buffer
    /// @param bitsPerSample 32 or 64
    /// @param directionField Output cube
    /// @param row Which row we're copying
    /// @param width Image width
    /// @param numChannels Number of components per pixel
    static void copyScanlineData(
        tdata_t buf,
        uint16_t bitsPerSample,
        SimpleCube<float>& directionField,
        uint32_t row,
        uint32_t width,
        uint16_t numChannels)
    {
        // Iterate over all columns in this row
        for (uint32_t col = 0; col < width; col++)
        {
            // Calculate index in scanline buffer
            size_t idx = col * numChannels;
            
            // Copy each channel
            for (uint16_t ch = 0; ch < numChannels; ch++)
            {
                float value;
                if (bitsPerSample == 32)
                {
                    value = static_cast<float*>(buf)[idx + ch];
                }
                else // 64-bit
                {
                    value = static_cast<float>(static_cast<double*>(buf)[idx + ch]);
                }
                
                // Store in SimpleCube with zyx indexing
                directionField.zyx(ch, row, col) = value;
            }
        }
    }


};