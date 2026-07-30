#pragma once

// OME-TIFF metadata -> OmeAxes. Parses the dimensional attributes of the OME-XML
// <Pixels> element (SizeX/Y/Z/C/T, DimensionOrder, Type, PhysicalSize*,
// TimeIncrement) plus <Channel> names. STL-only.
//
// Currently resolves dimensions + calibration + dtype. The TiffData plane->IFD
// map and SubIFDs pyramids are not parsed yet; levels here is a single full-res
// level.

#include <string>
#include "ome_axes.h"

namespace Nyxus
{
	// Parse an OME-XML document (the string from a TIFF IFD-0 ImageDescription).
	// Returns an OmeAxes with .valid==false if no <Pixels> element is found.
	OmeAxes parse_ome_xml(const std::string& omeXml);
}
