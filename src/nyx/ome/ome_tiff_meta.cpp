#include "ome_tiff_meta.h"

#include <cctype>
#include <charconv>
#include <algorithm>
#include <cmath>
#include <locale>
#include <sstream>

namespace Nyxus
{
	// Position of the '<' of element `name`'s start tag (closing == false) or end tag
	// (closing == true), at or after `from`. An OME-XML document may carry a namespace prefix on
	// every element and not just the root, so both spellings match ("<Pixels", "<ome:Pixels"), and
	// the name is boundary-checked so "Pixels" does not match a hypothetical "PixelsFoo".
	// npos when there is no such tag.
	static std::size_t find_tag (const std::string& xml, const std::string& name, std::size_t from, bool closing)
	{
		const std::string head = closing ? "</" : "<";
		std::size_t p = from;
		while ((p = xml.find (name, p)) != std::string::npos)
		{
			char after = (p + name.size() < xml.size()) ? xml[p + name.size()] : '\0';
			bool ok_after = after == '>' || after == ' ' || after == '\t' || after == '\n' || after == '\r'
				|| (! closing && after == '/');
			if (ok_after && p >= head.size())
			{
				// unprefixed: "<Name" / "</Name"
				if (xml.compare (p - head.size(), head.size(), head) == 0)
					return p - head.size();

				// prefixed: "<pfx:Name" / "</pfx:Name", where pfx is a single name -- none of the
				// characters that delimit a tag, an attribute or another prefix may appear in it
				if (xml[p - 1] == ':')
				{
					const std::size_t colon = p - 1, lt = xml.rfind ('<', colon);
					if (lt != std::string::npos && colon > lt + head.size()
						&& xml.compare (lt, head.size(), head) == 0
						&& xml.find_first_of (" \t\r\n<>/\"'=:", lt + head.size()) == colon)
						return lt;
				}
			}
			p += name.size();
		}
		return std::string::npos;
	}

	// Extract the start-tag text "<Name ... >" of the first element called `name`, prefixed or
	// not. Returns "" if not found.
	static std::string element_start_tag(const std::string& xml, const std::string& name)
	{
		std::size_t p = find_tag (xml, name, 0, false);
		if (p == std::string::npos)
			return "";
		std::size_t gt = xml.find('>', p);
		return (gt == std::string::npos) ? "" : xml.substr(p, gt - p + 1);
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

	// 's' without leading and trailing whitespace
	static std::string trimmed (const std::string& s)
	{
		std::size_t b = 0, e = s.size();
		while (b < e && std::isspace((unsigned char)s[b])) ++b;
		while (e > b && std::isspace((unsigned char)s[e - 1])) --e;
		return s.substr(b, e - b);
	}

	// The file name part of a path, lowercased: the OME <UUID FileName="..."> attribute names a
	// file, not a path, and the two sides may differ in case on Windows.
	static std::string base_name_lower (const std::string& path)
	{
		std::size_t slash = path.find_last_of ("/\\");
		std::string base = (slash == std::string::npos) ? path : path.substr (slash + 1);
		for (auto& c : base)
			c = (char) std::tolower ((unsigned char) c);
		return base;
	}

	// The root <OME ...> start tag, whatever namespace prefix it carries ("<OME", "<ome:OME").
	static std::string root_ome_tag (const std::string& xml)
	{
		return element_start_tag (xml, "OME");
	}

	OmeAxes parse_ome_xml(const std::string& xml, const std::string& self_filename)
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
			std::string fileUuid;	// this file's own UUID; empty when the root declares none
			get_attr (root_ome_tag (xml), "UUID", fileUuid);
			fileUuid = trimmed (fileUuid);
			const std::string self_base = base_name_lower (self_filename);
			std::vector<std::size_t> map;   // identity-initialized on first same-file block
			std::size_t tp = 0;
			while ((tp = find_tag (xml, "TiffData", tp, false)) != std::string::npos)
			{
				std::size_t gt = xml.find('>', tp);
				if (gt == std::string::npos) break;
				const std::string tag = xml.substr(tp, gt - tp + 1);
				anyTiffData = true;

				// A <TiffData> with a <UUID> child names the file that holds its planes; a block
				// naming another file marks a multi-file OME-TIFF, which this reader does not open.
				// The block's UUID decides whenever there is one to compare against the file's own
				// (the root <OME UUID=...>, which Bio-Formats repeats in every block of a
				// single-file OME-TIFF): it identifies the file no matter what the file is called,
				// whereas the FileName attribute records the name at write time and goes stale the
				// moment the file is renamed or copied. So the name decides only when there is no
				// UUID pair to compare, and a block naming no file at all is the current file by the
				// OME schema. Only a block this reader cannot confirm is its own is flagged and left
				// unmapped rather than pointed at a wrong local IFD.
				bool selfClosed = (gt > 0 && xml[gt - 1] == '/');
				if (!selfClosed)
				{
					std::size_t close = find_tag (xml, "TiffData", gt, true);
					std::size_t uuid = find_tag (xml, "UUID", gt, false);
					if (uuid != std::string::npos && (close == std::string::npos || uuid < close))
					{
						// the <UUID ...> start tag, its FileName, and its text -- the latter only
						// up to this block's own </TiffData>, so a self-closed <UUID/> cannot pick
						// up a later block's text
						std::size_t utagEnd = xml.find('>', uuid);
						std::string utag = (utagEnd == std::string::npos) ? std::string() : xml.substr(uuid, utagEnd - uuid + 1);
						std::string fname;
						get_attr (utag, "FileName", fname);

						std::string planeUuid;
						if (utagEnd != std::string::npos && !(utagEnd > 0 && xml[utagEnd - 1] == '/'))
						{
							std::size_t textEnd = find_tag (xml, "UUID", utagEnd, true);
							if (textEnd != std::string::npos && (close == std::string::npos || textEnd < close))
								planeUuid = trimmed (xml.substr(utagEnd + 1, textEnd - utagEnd - 1));
						}

						bool same_file;
						if (! planeUuid.empty() && ! fileUuid.empty())
							same_file = planeUuid == fileUuid;
						else
							if (! fname.empty() && ! self_base.empty())
								same_file = base_name_lower (fname) == self_base;
							else
								same_file = fname.empty();	// names no file: this one

						if (! same_file)
						{ sawMultiFile = true; tp = gt + 1; continue; }
					}
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
		ax.unitXY = s("PhysicalSizeXUnit", "");
		ax.unitZ = s("PhysicalSizeZUnit", "");

		// Canonicalize each axis to micrometer using ITS OWN declared unit (X/Y share
		// unitXY, Z has its own unitZ) -- a file that declares Z in a different unit than
		// X/Y (or either in nm/mm/etc.) would otherwise report raw, uncomparable values under
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
			else /* C */ { a.physical = 1.0; }
			ax.storageAxes.push_back(a);
		}

		// One full-res level: a SubIFDs pyramid is not read.
		ax.n_pyramid_levels = 1;

		ax.valid = true;
		return ax;
	}
}
