#pragma once

// OME-Zarr (NGFF) metadata -> OmeAxes. Parses the group `multiscales`
// block: `axes` (order + type + unit), `datasets` (pyramid paths +
// coordinateTransformations), for both 0.4 (metadata at the group root) and 0.5
// (metadata under the "ome" key). The caller supplies the level-0 array shape
// and dtype (read from the dataset's .zarray/zarr.json via z5) so this parser
// stays decoupled from the storage backend.
//
// Guarded by OMEZARR_SUPPORT because it depends on nlohmann::json (vendored with
// z5). Only meaningful when Nyxus is built with OME-Zarr support.

#ifdef OMEZARR_SUPPORT

#include <string>
#include <vector>
#include <cstddef>
#include "nlohmann/json.hpp"
#include "ome_axes.h"

namespace Nyxus
{
	// The first 'multiscales' entry of NGFF group attributes: 0.5 nests the model under an
	// "ome" key, 0.4 puts it at the group root. nullptr when there is none.
	const nlohmann::json* ome_zarr_multiscale (const nlohmann::json& groupAttrs);

	// Path of pyramid level 'level' of multiscales entry 'ms': the dataset's "path", or the
	// level's index when the entry names none. Throws std::runtime_error when 'ms' has no
	// dataset at that level.
	std::string ome_zarr_level_path (const nlohmann::json& ms, std::size_t level);

	// Parse NGFF group attributes into an OmeAxes.
	//   groupAttrs  : the group's attributes JSON (.zattrs contents for v2, or the
	//                 "attributes" object of zarr.json for v3).
	//   level0Shape : shape of the level-0 dataset, in on-disk axis order.
	//   dtypeStr    : the level-0 dtype string (v2 "<u2" or v3 "uint16").
	// Returns OmeAxes with .valid==false if no multiscales/axes are found.
	OmeAxes parse_ome_zarr(const nlohmann::json& groupAttrs,
		const std::vector<std::size_t>& level0Shape,
		const std::string& dtypeStr);
}

#endif // OMEZARR_SUPPORT
