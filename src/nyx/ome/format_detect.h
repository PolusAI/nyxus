#pragma once

// Single source of truth for "which loader backend reads this file?", used by all three
// loader dispatch sites (image_loader, raw_image_loader, image_loader1x) so they classify
// every path alike, including the double extensions .ome.zarr and .nii.gz.
// STL + <filesystem> only.

#include <string>

namespace Nyxus
{
	// Tiff covers every TIFF flavor (plain, multi-page, OME-TIFF) named .tif or .tiff, which
	// includes .ome.tif and .ome.tiff: a single TIFF loader serves them all and reads the OME-XML
	// itself. A name with no recognized extension is Unsupported, and is refused rather than
	// handed to the TIFF loader.
	enum class ContainerKind { Tiff, OmeZarr, Dicom, Nifti, Unsupported };

	// Classify by extension alone, case-insensitively. It opens nothing, so it stays cheap even
	// though ImageLoader::open() runs once per oversized ROI. Never throws.
	ContainerKind detect_container_family (const std::string& path);

	// detect_container_family for a file about to be opened: throws std::runtime_error, naming
	// the extensions Nyxus reads, when the file is Unsupported.
	ContainerKind supported_container_family (const std::string& path);
}
