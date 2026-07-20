#pragma once

// OME-TIFF metadata -> OmeAxes. Parses the dimensional attributes of the OME-XML
// <Pixels> element (SizeX/Y/Z/C/T, DimensionOrder, Type, PhysicalSize*,
// TimeIncrement) plus <Channel> names. STL-only.
//
// Resolves dimensions + calibration + dtype + the <TiffData> plane->IFD map
// (same-file; a <UUID> multi-file block sets OmeAxes::multiFileTiff and is left
// at the canonical fallback). SubIFDs pyramids are not parsed yet; levels here is
// a single full-res level.

#include <string>
#include "ome_axes.h"

namespace Nyxus
{
	// Parse an OME-XML document (the string from a TIFF IFD-0 ImageDescription).
	// Returns an OmeAxes with .valid==false if no <Pixels> element is found.
	OmeAxes parse_ome_xml(const std::string& omeXml);
}
