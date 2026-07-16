#pragma once

// Single source of truth for "what kind of file is this?" used by all three
// loader dispatch sites (image_loader, raw_image_loader, image_loader1x). Adds
// the OME content-sniff that is absent today and unifies the extension handling
// (image_loader1x currently uses single-extension matching and cannot see
// .ome.zarr / .nii.gz). STL + libtiff + <filesystem> only.

#include <string>

namespace Nyxus
{
	enum class ContainerKind { TiffPlain, OmeTiff, OmeZarr, Dicom, Nifti, Unknown };

	struct InputFormat
	{
		ContainerKind kind = ContainerKind::Unknown;
		bool is_ome = false;    // OME metadata actually present (OME-XML / NGFF multiscales)
	};

	// Classify by big-extension, then content-sniff: OME-XML in TIFF IFD-0
	// ImageDescription, or NGFF `multiscales` in a Zarr group. Never throws.
	InputFormat detect_input_format(const std::string& path);

	const char* to_string(ContainerKind k);
}
