#include "ome_tiff_meta.h"

#include <cctype>
#include <cstdlib>
#include <algorithm>
#include <cmath>

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
		// Size reader: absent OR non-numeric OR negative OR "0" -> at least 1. OME
		// extents are >=1; a 0 divides-by-zero the tile grid and a negative would
		// wrap through strtoull into a huge (allocation-bombing) extent.
		auto i = [&](const char* n, std::size_t def) -> std::size_t {
			std::string v; if (!get_attr(pix, n, v)) return def;
			const char* p = v.c_str();
			while (*p == ' ' || *p == '\t') ++p;
			if (*p == '-') return def;          // reject negative before strtoull wraps it
			char* end = nullptr; unsigned long long r = std::strtoull(p, &end, 10);
			if (end == p) return def;           // no digits parsed -> keep default
			return r < 1 ? std::size_t(1) : (std::size_t)r;
		};
		// Double reader: absent OR non-numeric OR non-finite -> keep the default.
		// Guards against a failed parse (0.0) and against strtod swallowing "NaN"/
		// "inf" prefixes (e.g. "NaNsense" -> NaN), any of which would corrupt
		// physical-calibration math downstream.
		auto d = [&](const char* n, double def) -> double {
			std::string v; if (!get_attr(pix, n, v)) return def;
			char* end = nullptr; double r = std::strtod(v.c_str(), &end);
			if (end == v.c_str() || !std::isfinite(r)) return def;
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
