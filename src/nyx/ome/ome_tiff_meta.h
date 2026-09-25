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
	// A <TiffData><UUID> block naming any file other than this one is a multi-file OME-TIFF
	// (OmeAxes::multiFileTiff). Which file a block names is read from its UUID text whenever the
	// root <OME> declares one to compare it against; failing that, from its FileName attribute
	// against 'self_filename', the file the XML came from, which may be empty when the caller has
	// no name to compare.
	OmeAxes parse_ome_xml(const std::string& omeXml, const std::string& self_filename = "");
}
