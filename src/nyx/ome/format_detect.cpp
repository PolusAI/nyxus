#define NOMINMAX

#include "format_detect.h"

#include <algorithm>
#include <cctype>

#include "../helpers/fsystem.h"

namespace Nyxus
{
	ContainerKind detect_container_family (const std::string& path)
	{
		std::string ext = Nyxus::get_big_extension(path);
		std::transform(ext.begin(), ext.end(), ext.begin(),
			[](unsigned char c) { return (char)std::tolower(c); });

		if (ext == ".zarr" || ext == ".ome.zarr")
			return ContainerKind::OmeZarr;
		if (ext == ".dcm" || ext == ".dicom")
			return ContainerKind::Dicom;
		if (ext == ".nii" || ext == ".nii.gz")
			return ContainerKind::Nifti;

		// Everything else is a TIFF flavor (.tif/.tiff/.ome.tif/.ome.tiff/.tf2/.tf8/.btf).
		return ContainerKind::Tiff;
	}
}
