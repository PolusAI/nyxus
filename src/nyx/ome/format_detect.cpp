#define NOMINMAX

#include "format_detect.h"

#include <algorithm>
#include <fstream>
#include <cctype>

#include <tiffio.h>

#include "../helpers/fsystem.h"

namespace Nyxus
{
	const char* to_string(ContainerKind k)
	{
		switch (k)
		{
			case ContainerKind::TiffPlain: return "TiffPlain";
			case ContainerKind::OmeTiff:   return "OmeTiff";
			case ContainerKind::OmeZarr:   return "OmeZarr";
			case ContainerKind::Dicom:     return "Dicom";
			case ContainerKind::Nifti:     return "Nifti";
			default:                       return "Unknown";
		}
	}

	// Does this TIFF carry an OME-XML block in IFD-0's ImageDescription?
	static bool tiff_is_ome(const std::string& path)
	{
		// silence libtiff's stderr chatter on non-OME / odd TIFFs during the sniff
		TIFFErrorHandler oldE = TIFFSetErrorHandler(nullptr);
		TIFFErrorHandler oldW = TIFFSetWarningHandler(nullptr);
		bool ome = false;
		TIFF* t = TIFFOpen(path.c_str(), "r");
		if (t)
		{
			char* desc = nullptr;   // libtiff-owned, do not free
			if (TIFFGetField(t, TIFFTAG_IMAGEDESCRIPTION, &desc) == 1 && desc)
			{
				std::string d(desc);
				ome = d.find("<OME") != std::string::npos;
			}
			TIFFClose(t);
		}
		TIFFSetErrorHandler(oldE);
		TIFFSetWarningHandler(oldW);
		return ome;
	}

	// Cheap text check for NGFF `multiscales` in a Zarr group's metadata file,
	// without pulling in a JSON parser (keeps this unit z5/nlohmann-free).
	static bool zarr_has_multiscales(const std::string& dirpath)
	{
		const char* candidates[] = { ".zattrs", "zarr.json" };
		for (const char* c : candidates)
		{
			fs::path mp = fs::path(dirpath) / c;
			std::error_code ec;
			if (!fs::exists(mp, ec)) continue;
			std::ifstream f(mp);
			if (!f) continue;
			std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
			if (content.find("multiscales") != std::string::npos)
				return true;
		}
		return false;
	}

	InputFormat detect_input_format(const std::string& path)
	{
		std::string ext = Nyxus::get_big_extension(path);
		std::transform(ext.begin(), ext.end(), ext.begin(),
			[](unsigned char c) { return (char)std::tolower(c); });

		if (ext == ".zarr" || ext == ".ome.zarr")
			return { ContainerKind::OmeZarr, zarr_has_multiscales(path) };
		if (ext == ".dcm" || ext == ".dicom")
			return { ContainerKind::Dicom, false };
		if (ext == ".nii" || ext == ".nii.gz")
			return { ContainerKind::Nifti, false };

		// Everything else is a TIFF flavor (.tif/.tiff/.ome.tif/.ome.tiff/.tf2/.tf8/.btf).
		if (tiff_is_ome(path))
			return { ContainerKind::OmeTiff, true };
		return { ContainerKind::TiffPlain, false };
	}
}
