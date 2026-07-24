#include "ome_tiff_meta.h"

#include <cctype>
#include <charconv>
#include <algorithm>
#include <cmath>
#include <locale>
#include <sstream>

namespace Nyxus
{
	// Extract the start-tag text "<Name ... >" of the first element called `name`.
	// Returns "" if not found. Matches a name boundary so "Pixels" does not match
	// a hypothetical "PixelsFoo".
	static std::string element_start_tag(const std::string& xml, const std::string& name)
	{
		const std::string open = "<" + name;
		std::size_t p = 0;
		while ((p = xml.find(open, p)) != std::string::npos)
		{
			char after = (p + open.size() < xml.size()) ? xml[p + open.size()] : '\0';
			if (after == ' ' || after == '\t' || after == '\n' || after == '\r' || after == '>' || after == '/')
			{
				std::size_t gt = xml.find('>', p);
				if (gt == std::string::npos) return "";
				return xml.substr(p, gt - p + 1);
			}
			p += open.size();
		}
		return "";
	}

	// Read attribute `name` from a single start-tag string. Boundary-checked so
	// "SizeX" does not match inside "PhysicalSizeX". Handles ' and " quoting.
	static bool get_attr(const std::string& tag, const std::string& name, std::string& out)
	{
		std::size_t pos = 0;
		while ((pos = tag.find(name, pos)) != std::string::npos)
		{
			bool ok_before = (pos == 0) || std::isspace((unsigned char)tag[pos - 1]);
			std::size_t q = pos + name.size();
			while (q < tag.size() && std::isspace((unsigned char)tag[q])) ++q;
			if (ok_before && q < tag.size() && tag[q] == '=')
			{
				++q;
				while (q < tag.size() && std::isspace((unsigned char)tag[q])) ++q;
				if (q < tag.size() && (tag[q] == '"' || tag[q] == '\''))
				{
					char quote = tag[q];
					std::size_t start = q + 1, end = tag.find(quote, start);
					if (end != std::string::npos) { out = tag.substr(start, end - start); return true; }
				}
			}
			pos += name.size();
		}
		return false;
	}

	OmeAxes parse_ome_xml(const std::string& xml)
	{
		OmeAxes ax;
		const std::string pix = element_start_tag(xml, "Pixels");
		if (pix.empty())
			return ax;   // valid stays false

		auto s = [&](const char* n, const std::string& def) { std::string v; return get_attr(pix, n, v) ? v : def; };
		// Size reader (std::from_chars): absent / non-numeric / negative / overflow
		// -> default; "0" -> 1. OME extents are >=1; from_chars rejects a leading
		// '-' and overflow via errc, so no strtoull wrap or divide-by-zero leaks out.
		auto i = [&](const char* n, std::size_t def) -> std::size_t {
			std::string v; if (!get_attr(pix, n, v)) return def;
			std::size_t r = 0;
			auto [ptr, ec] = std::from_chars(v.data(), v.data() + v.size(), r);
			if (ec != std::errc() || ptr == v.data()) return def;
			return r < 1 ? std::size_t(1) : r;
		};
		// Double reader: absent / non-numeric / non-finite -> default. libc++ provides
		// std::from_chars for integral types only (its floating-point overloads are
		// deleted or constrained to is_integral), so the double form does not compile
		// there; a stream imbued with the classic locale reads the same value on every
		// standard library and, unlike strtod, ignores the host's LC_NUMERIC, which a
		// process embedding the library may have set to a comma-decimal locale.
		// "nan"/"inf" prefixes (e.g. "NaNsense") still parse, so the isfinite check
		// keeps those out of physical-calibration math.
		auto d = [&](const char* n, double def) -> double {
			std::string v; if (!get_attr(pix, n, v)) return def;
			std::istringstream is(v);
			is.imbue(std::locale::classic());
			double r = 0;
			is >> r;
			if (is.fail() || !std::isfinite(r)) return def;
			return r;
		};

		ax.sizeX = i("SizeX", 1);
		ax.sizeY = i("SizeY", 1);
		ax.sizeZ = i("SizeZ", 1);
		ax.sizeC = i("SizeC", 1);
		ax.sizeT = i("SizeT", 1);

		// DimensionOrder must be a permutation of the 5 canonical axes; a missing
		// or malformed value falls back to the OME default so storageOrder stays sane.
		std::string dord = s("DimensionOrder", "XYZCT");
		bool perm = dord.size() == 5;
		for (char c : std::string("XYZCT"))
			if (std::count(dord.begin(), dord.end(), c) != 1) perm = false;
		ax.omeDimensionOrder = perm ? dord : "XYZCT";

		// Explicit plane->IFD mapping from <TiffData> (OME-XML). Absent it, planes are stored
		// contiguously from IFD 0 in DimensionOrder (the ifdForPlane default), which is what
		// tifffile's single "<TiffData IFD=0 PlaneCount=N/>" also means -- so this only changes
		// behavior for writers that start at a non-zero IFD or reorder planes (e.g. bioformats
		// per-plane blocks, or a multi-image container). Each block maps PlaneCount consecutive
		// DimensionOrder planes starting at (FirstZ,FirstC,FirstT) to consecutive IFDs from IFD.
		{
			const std::size_t totalPlanes = ax.sizeZ * ax.sizeC * ax.sizeT;
			// integer attribute of a start-tag, with default (same guards as `i` above)
			auto iat = [](const std::string& tag, const char* n, std::size_t def) -> std::size_t {
				std::string v; if (!get_attr(tag, n, v)) return def;
				std::size_t r = 0;
				auto [ptr, ec] = std::from_chars(v.data(), v.data() + v.size(), r);
				return (ec != std::errc() || ptr == v.data()) ? def : r;
			};

			bool anyTiffData = false, sawMultiFile = false;
			std::vector<std::size_t> map;   // identity-initialized on first same-file block
			std::size_t tp = 0;
			while ((tp = xml.find("<TiffData", tp)) != std::string::npos)
			{
				char after = (tp + 9 < xml.size()) ? xml[tp + 9] : '\0';
				if (!(after == ' ' || after == '\t' || after == '\n' || after == '\r' || after == '>' || after == '/'))
				{ tp += 9; continue; }
				std::size_t gt = xml.find('>', tp);
				if (gt == std::string::npos) break;
				const std::string tag = xml.substr(tp, gt - tp + 1);
				anyTiffData = true;

				// A <TiffData> with a <UUID> child names a plane in ANOTHER file (companion /
				// multi-file OME-TIFF) -- unsupported here. Detect it (tag not self-closed AND a
				// <UUID follows before </TiffData>) and skip mapping it, leaving those planes at
				// the canonical fallback rather than pointing them at a wrong local IFD.
				bool selfClosed = (gt > 0 && xml[gt - 1] == '/');
				if (!selfClosed)
				{
					std::size_t close = xml.find("</TiffData>", gt);
					std::size_t uuid = xml.find("<UUID", gt);
					if (uuid != std::string::npos && (close == std::string::npos || uuid < close))
					{ sawMultiFile = true; tp = gt + 1; continue; }
				}

				const std::size_t fz = iat(tag, "FirstZ", 0), fc = iat(tag, "FirstC", 0), ft = iat(tag, "FirstT", 0);
				const std::size_t ifd0 = iat(tag, "IFD", 0);
				const std::size_t startOrd = ax.canonicalPlaneOrdinal(fz, fc, ft);
				// PlaneCount default per OME: the remaining planes from the start plane.
				std::size_t count = iat(tag, "PlaneCount", (startOrd < totalPlanes) ? totalPlanes - startOrd : 0);

				if (map.empty() && totalPlanes > 0)
					{ map.resize(totalPlanes); for (std::size_t k = 0; k < totalPlanes; ++k) map[k] = k; }
				for (std::size_t k = 0; k < count && (startOrd + k) < totalPlanes; ++k)
					map[startOrd + k] = ifd0 + k;
				tp = gt + 1;
			}
			ax.multiFileTiff = sawMultiFile;
			// Only keep a non-identity map (a plain identity means canonical -> leave empty so
			// the common no-/canonical-TiffData path stays allocation-free and obviously canonical).
			if (anyTiffData && !map.empty())
			{
				bool identity = true;
				for (std::size_t k = 0; k < map.size(); ++k) if (map[k] != k) { identity = false; break; }
				if (!identity) ax.planeToIfd = std::move(map);
			}
		}

		ax.dtype = pixel_type_from_ome_type(s("Type", "uint16"));
		if (ax.dtype == PixelType::Unknown) ax.dtype = PixelType::UInt16;
		ax.bitsPerSample = bits_of(ax.dtype);

		ax.physX = d("PhysicalSizeX", 1.0);
		ax.physY = d("PhysicalSizeY", 1.0);
		ax.physZ = d("PhysicalSizeZ", 1.0);
		ax.timeIncrement = d("TimeIncrement", 1.0);
		ax.unitXY = s("PhysicalSizeXUnit", "");
		ax.unitZ = s("PhysicalSizeZUnit", "");
		ax.unitT = s("TimeIncrementUnit", "");

		// Canonicalize each axis to micrometer using ITS OWN declared unit (X/Y share
		// unitXY, Z has its own unitZ) -- a file that declares Z in a different unit than
		// X/Y (or either in nm/mm/etc.) previously reported raw, uncomparable values under
		// a unit label that only ever reflected X/Y. No-op for an already-micrometer or
		// unrecognized/uncalibrated unit.
		std::string unitY = ax.unitXY;
		canonicalize_to_micrometer(ax.physX, ax.unitXY);
		canonicalize_to_micrometer(ax.physY, unitY);
		canonicalize_to_micrometer(ax.physZ, ax.unitZ);

		// On-disk order = reverse(DimensionOrder) with singleton axes dropped
		// (planes are XY, so the result always ends in "YX").
		auto size_of = [&](char c) -> std::size_t {
			switch (c) { case 'X': return ax.sizeX; case 'Y': return ax.sizeY;
				case 'Z': return ax.sizeZ; case 'C': return ax.sizeC; case 'T': return ax.sizeT; }
			return 1;
		};
		std::string rev(ax.omeDimensionOrder.rbegin(), ax.omeDimensionOrder.rend());
		for (char c : rev)
			if (c == 'X' || c == 'Y' || size_of(c) > 1)
				ax.storageOrder.push_back(c);

		for (char c : ax.storageOrder)
		{
			OmeAxis a;
			a.label = c;
			a.kind = axis_kind_of(c);
			a.size = size_of(c);
			if (c == 'X') { a.physical = ax.physX; a.unit = ax.unitXY; }
			else if (c == 'Y') { a.physical = ax.physY; a.unit = ax.unitXY; }
			else if (c == 'Z') { a.physical = ax.physZ; a.unit = ax.unitZ; }
			else if (c == 'T') { a.physical = ax.timeIncrement; a.unit = ax.unitT; }
			else /* C */ { a.physical = 1.0; }
			ax.storageAxes.push_back(a);
		}

		// Single full-res level (SubIFDs pyramid parsing not implemented yet).
		OmePyramidLevel lv;
		lv.index = 0;
		lv.sizeX = ax.sizeX; lv.sizeY = ax.sizeY; lv.sizeZ = ax.sizeZ;
		lv.scaleX = ax.physX; lv.scaleY = ax.physY; lv.scaleZ = ax.physZ;
		lv.path = "0";
		ax.levels.push_back(lv);

		// Optional channel names.
		std::size_t cp = 0;
		while ((cp = xml.find("<Channel", cp)) != std::string::npos)
		{
			char after = (cp + 8 < xml.size()) ? xml[cp + 8] : '\0';
			if (after == ' ' || after == '\t' || after == '\n' || after == '\r' || after == '>')
			{
				std::size_t gt = xml.find('>', cp);
				if (gt == std::string::npos) break;
				std::string ctag = xml.substr(cp, gt - cp + 1);
				std::string nm;
				if (get_attr(ctag, "Name", nm)) ax.channelNames.push_back(nm);
				cp = gt + 1;
			}
			else cp += 8;
		}

		ax.valid = true;
		return ax;
	}
}
