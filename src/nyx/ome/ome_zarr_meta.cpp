#ifdef OMEZARR_SUPPORT

#include "ome_zarr_meta.h"
#include <cctype>
#include <stdexcept>

namespace Nyxus
{
	// Pull the 'scale' vector out of a dataset's coordinateTransformations (the
	// entry whose type == "scale"); empty if the key/entry is absent or malformed.
	// Safe on datasets that lack coordinateTransformations entirely (else a const
	// operator[] on a missing key throws).
	static std::vector<double> scale_of(const nlohmann::json& dset)
	{
		std::vector<double> s;
		if (!dset.is_object() || !dset.contains("coordinateTransformations"))
			return s;
		const auto& cts = dset["coordinateTransformations"];
		if (cts.is_array())
			for (const auto& ct : cts)
				if (ct.is_object() && ct.value("type", std::string()) == "scale" &&
					ct.contains("scale") && ct["scale"].is_array())
				{
					for (const auto& v : ct["scale"]) s.push_back(v.is_number() ? v.get<double>() : 1.0);
					break;
				}
		return s;
	}

	const nlohmann::json* ome_zarr_multiscale (const nlohmann::json& groupAttrs)
	{
		// 0.5 nests the model under "ome"; 0.4 puts it at the group root.
		const nlohmann::json* root = nullptr;
		if (groupAttrs.contains("ome") && groupAttrs["ome"].contains("multiscales"))
			root = &groupAttrs["ome"];
		else if (groupAttrs.contains("multiscales"))
			root = &groupAttrs;
		if (!root)
			return nullptr;

		const auto& multiscales = (*root)["multiscales"];
		if (!multiscales.is_array() || multiscales.empty())
			return nullptr;
		return &multiscales[0];
	}

	std::string ome_zarr_level_path (const nlohmann::json& ms, std::size_t level)
	{
		if (!ms.is_object() || !ms.contains("datasets") || !ms["datasets"].is_array() || level >= ms["datasets"].size())
			throw std::runtime_error ("OME-Zarr: multiscales declares no dataset for pyramid level " + std::to_string (level));

		const auto& dset = ms["datasets"][level];
		return dset.is_object() ? dset.value("path", std::to_string(level)) : std::to_string(level);
	}

	OmeAxes parse_ome_zarr(const nlohmann::json& groupAttrs,
		const std::vector<std::size_t>& level0Shape,
		const std::string& dtypeStr)
	{
		OmeAxes ax;

		const nlohmann::json* multiscale = ome_zarr_multiscale (groupAttrs);
		if (!multiscale)
			return ax;
		const auto& ms = *multiscale;
		if (!ms.contains("axes") || !ms.contains("datasets"))
			return ax;

		const auto& axes = ms["axes"];
		const auto& datasets = ms["datasets"];
		if (!axes.is_array() || !datasets.is_array() || datasets.empty())
			return ax;

		// Level-0 scale (per-axis), from the first dataset's transformations.
		std::vector<double> scale0 = scale_of(datasets[0]);
		if (scale0.size() != axes.size())
			scale0.assign(axes.size(), 1.0);

		ax.dtype = pixel_type_from_zarr_dtype(dtypeStr);
		if (ax.dtype == PixelType::Unknown) ax.dtype = PixelType::UInt16;
		ax.bitsPerSample = bits_of(ax.dtype);

		for (std::size_t k = 0; k < axes.size(); ++k)
		{
			const auto& a = axes[k];
			if (!a.is_object())    // a malformed axis entry -> skip (keeps role sizes at default)
				continue;
			std::string name = a.value("name", std::string());
			std::string type = a.value("type", std::string());
			std::string unit = a.value("unit", std::string());
			char label = name.empty() ? '?' : (char)std::toupper((unsigned char)name[0]);

			OmeAxis oa;
			oa.label = label;
			if (type == "channel") oa.kind = AxisKind::Channel;
			else if (type == "time") oa.kind = AxisKind::Time;
			else oa.kind = AxisKind::Space;
			oa.size = (k < level0Shape.size()) ? level0Shape[k] : 1;
			oa.physical = (k < scale0.size()) ? scale0[k] : 1.0;
			// Canonicalize space axes (X/Y/Z) to micrometer using their OWN declared unit
			// (a Z axis calibrated in a different unit than X/Y would otherwise report its raw,
			// uncomparable value under whatever label X/Y happened to carry). Time/channel
			// axes are left alone -- a time unit like "second" isn't a length to convert.
			if (label == 'X' || label == 'Y' || label == 'Z')
				canonicalize_to_micrometer(oa.physical, unit);
			oa.unit = unit;
			ax.storageAxes.push_back(oa);
			ax.storageOrder.push_back(label);

			switch (label)
			{
				case 'X': ax.sizeX = oa.size; ax.physX = oa.physical; ax.unitXY = unit; break;
				case 'Y': ax.sizeY = oa.size; ax.physY = oa.physical; if (ax.unitXY.empty()) ax.unitXY = unit; break;
				case 'Z': ax.sizeZ = oa.size; ax.physZ = oa.physical; ax.unitZ = unit; break;
				case 'C': ax.sizeC = oa.size; break;
				case 'T': ax.sizeT = oa.size; break;
				default: break;
			}
		}

		ax.omeDimensionOrder = ome_dimension_order_from_storage(ax.storageOrder);

		// Pyramid levels: one per dataset entry; Nyxus reads level 0.
		ax.n_pyramid_levels = datasets.size();

		ax.valid = true;
		return ax;
	}
}

#endif // OMEZARR_SUPPORT
