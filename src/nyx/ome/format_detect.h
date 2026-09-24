#pragma once

// Single source of truth for "which loader backend reads this file?", used by all three
// loader dispatch sites (image_loader, raw_image_loader, image_loader1x) so they classify
// every path alike, including the double extensions .ome.zarr and .nii.gz.
// STL + <filesystem> only.

#include <string>

namespace Nyxus
{
	// Tiff covers every TIFF flavor (plain, multi-page, OME-TIFF): a single TIFF loader serves
	// them all and reads the OME-XML itself.
	enum class ContainerKind { Tiff, OmeZarr, Dicom, Nifti };

	// Classify by big-extension alone. It opens nothing, so it stays cheap even though
	// ImageLoader::open() runs once per oversized ROI. Never throws.
	ContainerKind detect_container_family (const std::string& path);
}
