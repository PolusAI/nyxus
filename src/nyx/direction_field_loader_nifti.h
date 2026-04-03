#pragma once

#include <string>
#include <memory>
#include <cmath>
#include <iostream>
#include "features/image_cube.h"
#include "io/nifti/nifti2_io.h"

/// @brief Loader for NIfTI direction field images
/// 
/// PURPOSE: Load direction fields from NIfTI medical imaging format
/// 
/// NIfTI FORMAT DETAILS:
/// - NIfTI uses 4 dimensions: (nx, ny, nz, nt)
///   - nx, ny = width, height (spatial dimensions)
///   - nz = depth (1 for 2D images)
///   - nt = time/volumes (we use this for vector components)
/// - For direction fields:
///   - nt should be 2 (dx, dy) for 2D
///   - nt should be 3 (dx, dy, dz) for 3D
///   - nt can be 4 if magnitude is also stored (we ignore 4th component)
/// 
/// MEMORY LAYOUT:
/// - NIfTI stores data as: data[x + nx*(y + ny*(z + nz*t))]
/// - Component 0 (t=0) = dx values for all pixels
/// - Component 1 (t=1) = dy values for all pixels
/// - Component 2 (t=2) = dz values for all pixels (if 3D)
/// 
/// Direction vectors are stored as float values 
class DirectionFieldLoaderNifti
{
public:
    /// @brief Load a direction field from a NIfTI file
    /// 
    /// PURPOSE: Main entry point for loading NIfTI direction fields
    /// 
    /// PROCESS:
    /// 1. Open and read NIfTI file
    /// 2. Validate dimensions (must be 2D or 3D with proper component count)
    /// 3. Copy data from NIfTI memory layout to SimpleCube
    /// 
    /// @param filePath Path to the NIfTI file (.nii or .nii.gz)

    /// @return SimpleCube containing the direction field with shape (width, height, num_components)
    /// 
    /// EXAMPLE:
    ///   auto field = DirectionFieldLoaderNifti::load("directions.nii.gz", false);
    ///   // field->zyx(0, y, x) = dx at position (x, y)
    ///   // field->zyx(1, y, x) = dy at position (x, y)
    static std::unique_ptr<SimpleCube<float>> load(const std::string& filePath)
    {
        // Read the NIfTI file
        nifti_image* nii = nifti_image_read(filePath.c_str(), 1);
        if (!nii)
        {
            throw std::runtime_error("DirectionFieldLoaderNifti: Cannot open file " + filePath);
        }

        // Get dimensions from NIfTI header
        size_t width = nii->nx;          // Image width (x-dimension)
        size_t height = nii->ny;         // Image height (y-dimension)
        size_t depth = nii->nz;          // Image depth (z-dimension, should be 1 for 2D)
        size_t numComponents = nii->nt;  // Number of volumes (4th dimension = vector components)

        std::cout << "NIfTI dimensions: " << width << "x" << height << "x" << depth << "x" << numComponents << std::endl;

        // Validate dimensions
        // For 2D images: depth must be 1, components should be 2 (dx, dy)
        // For 3D images: depth > 1, components should be 3 (dx, dy, dz)
        // Currently only 2D is fully supported
        if (depth != 1)
        {
            nifti_image_free(nii);
            throw std::runtime_error(
                "DirectionFieldLoaderNifti: Expected 2D image (depth=1), got depth=" + 
                std::to_string(depth)
            );
        }

        if (numComponents < 2 || numComponents > 4)
        {
            nifti_image_free(nii);
            throw std::runtime_error(
                "DirectionFieldLoaderNifti: Expected 2-4 components, got " + 
                std::to_string(numComponents) + " components"
            );
        }

        // Determine how many components to use
        // - If 2D (depth=1): use 2 components (dx, dy)
        // - If 3D (depth>1): use 3 components (dx, dy, dz)
        // - If nt=4, the 4th component might be magnitude - we ignore it
        size_t useComponents = std::min(size_t(3), numComponents);
        if (depth == 1)
            useComponents = std::min(size_t(2), numComponents);  // 2D case

        // Allocate output cube
        auto directionField = std::make_unique<SimpleCube<float>>(width, height, useComponents);

        // Copy data from NIfTI buffer to SimpleCube
        // NIfTI supports multiple data types - handle the most common ones
        switch (nii->datatype)
        {
        case 16: // NIFTI_TYPE_FLOAT32
            copyData<float>(nii, *directionField, width, height, useComponents, numComponents);
            break;
        case 64: // NIFTI_TYPE_FLOAT64
            copyData<double>(nii, *directionField, width, height, useComponents, numComponents);
            break;
        case 8: // NIFTI_TYPE_INT32
            copyData<int32_t>(nii, *directionField, width, height, useComponents, numComponents);
            break;
        case 512: // NIFTI_TYPE_UINT16
            copyData<uint16_t>(nii, *directionField, width, height, useComponents, numComponents);
            break;
        default:
            nifti_image_free(nii);
            throw std::runtime_error(
                "DirectionFieldLoaderNifti: Unsupported data type " + 
                std::to_string(nii->datatype)
            );
        }

        nifti_image_free(nii);


        return directionField;
    }

private:
    /// @brief Copy data from NIfTI to SimpleCube (template for different data types)
    /// 
    /// PURPOSE: Transfer data from NIfTI's memory layout to SimpleCube's layout
    /// 
    /// WHY TEMPLATE?
    /// - NIfTI can store data as float, double, int32, uint16, etc.
    /// - Template allows handling all types with same code
    /// - Converts everything to float32 for SimpleCube
    /// 
    /// MEMORY LAYOUT CONVERSION:
    /// - NIfTI layout: data[x + width*(y + height*(z + depth*component))]
    ///   All dx values are stored together, then all dy values, etc.
    /// - SimpleCube layout: directionField.zyx(component, y, x)
    ///   Each pixel has all its components together
    /// 
    /// @tparam T Source data type (float, double, int32_t, etc.)
    /// @param nii NIfTI image structure
    /// @param directionField Output SimpleCube to fill
    /// @param width Image width
    /// @param height Image height
    /// @param useComponents How many components to copy (2 or 3)
    /// @param totalComponents Total components in NIfTI (might be more than useComponents)
    template<typename T>
    static void copyData(
        nifti_image* nii,
        SimpleCube<float>& directionField,
        size_t width,
        size_t height,
        size_t useComponents,
        size_t totalComponents)
    {
        // Get pointer to NIfTI data buffer (cast to appropriate type)
        T* data = static_cast<T*>(nii->data);
        
        // NIfTI memory organization:
        // - All pixels for component 0 (dx) come first
        // - Then all pixels for component 1 (dy)
        // - Then all pixels for component 2 (dz) if present
        // We need to reorganize this into SimpleCube format
        
        for (size_t comp = 0; comp < useComponents; comp++)
        {
            for (size_t y = 0; y < height; y++)
            {
                for (size_t x = 0; x < width; x++)
                {
                    // Calculate NIfTI linear index
                    // Formula: x + width * (y + height * (z + depth * component))
                    // For 2D where z=0: x + width * (y + height * component)
                    // This means: component 0 occupies indices [0, width*height)
                    //             component 1 occupies indices [width*height, 2*width*height)
                    //             etc.
                    size_t niftiIdx = x + width * (y + height * comp);
                    float value = static_cast<float>(data[niftiIdx]);
                    
                    // Store in SimpleCube with zyx indexing: z=component, y=row, x=col
                    directionField.zyx(comp, y, x) = value;
                }
            }
        }
    }


};
