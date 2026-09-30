#define NOMINMAX

#include "format_detect.h"

#include <algorithm>
#include <cctype>
#include <stdexcept>

#include "../helpers/fsystem.h"

namespace Nyxus
{
	static std::string lowercase (std::string s)
	{
		std::transform(s.begin(), s.end(), s.begin(),
			[](unsigned char c) { return (char)std::tolower(c); });
		return s;
	}

	ContainerKind detect_container_family (const std::string& path)
	{
		// The last extension decides, so "scan.001.tif" is a TIFF like "scan.tif" and
		// "scan.ome.tif" is one too; only a compressed NIfTI needs the extension before it.
		const std::string ext = lowercase (fs::path(path).extension().string()),
			big = lowercase (Nyxus::get_big_extension(path));

		if (ext == ".tif" || ext == ".tiff")
			return ContainerKind::Tiff;
		if (ext == ".zarr")
			return ContainerKind::OmeZarr;
		if (ext == ".dcm" || ext == ".dicom")
			return ContainerKind::Dicom;
		if (ext == ".nii" || big == ".nii.gz")
			return ContainerKind::Nifti;

		return ContainerKind::Unsupported;
	}

	ContainerKind supported_container_family (const std::string& path)
	{
		ContainerKind kind = detect_container_family (path);
		if (kind == ContainerKind::Unsupported)
			throw std::runtime_error ("unsupported file type: " + path + " -- Nyxus reads .tif, .tiff, .ome.tif, .ome.tiff, "
				".zarr, .ome.zarr, .dcm, .dicom, .nii and .nii.gz");
		return kind;
	}
}
