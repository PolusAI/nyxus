#pragma once

// Plane addressing shared by the four TIFF loaders (the strip and tile loaders of both the
// AbstractTileLoader and RawFormatLoader stacks). An OME-TIFF stores one (z,c,t) plane per
// directory in the order its OME-XML DimensionOrder gives; a plain multi-page TIFF stores
// one Z-plane per directory. Reading the OME-XML, counting a plain TIFF's planes,
// range-checking a requested plane and selecting its directory all live here once, so no
// loader can check less than another.

#ifdef __APPLE__
    #define uint64 uint64_hack_
    #define int64 int64_hack_
    #include <tiffio.h>
    #undef uint64
    #undef int64
#else
    #include <tiffio.h>
#endif

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

#include "ome_tiff_meta.h"

namespace Nyxus
{
	/// @brief Read the OME-XML an OME-TIFF carries in the current (first) directory's
	/// ImageDescription into 'axes'.
	/// @return true for an OME-TIFF; false (and 'axes' left untouched) for a plain TIFF, which
	/// has no description or a description that is not OME-XML. Throws for a multi-file OME-TIFF.
	inline bool read_ome_tiff_axes (TIFF* tiff, OmeAxes& axes)
	{
		char* desc = nullptr;	// owned by libtiff
		if (TIFFGetField (tiff, TIFFTAG_IMAGEDESCRIPTION, &desc) != 1 || desc == nullptr)
			return false;

		// the root element, with or without a namespace prefix ("<OME ...", "<ome:OME ...")
		std::string xml (desc);
		if (xml.find ("<OME") == std::string::npos && xml.find (":OME") == std::string::npos)
			return false;

		// the file's own name, so a <TiffData><UUID FileName=...> block naming it is read as this
		// file's rather than a companion's
		const char* self = TIFFFileName (tiff);
		OmeAxes parsed = parse_ome_xml (xml, self ? std::string (self) : std::string());
		if (! parsed.valid)
			return false;

		// Planes kept in companion files cannot be read from this one: a (z,c,t) that lives
		// elsewhere would be read from whatever local directory its ordinal lands on, or fail
		// partway through a run
		if (parsed.multiFileTiff)
			throw std::runtime_error ("multi-file OME-TIFF: some planes are stored in companion files, which Nyxus does not read; "
				"combine the planes into a single OME-TIFF");

		axes = parsed;
		return true;
	}

	/// @brief Z extent of a plain (non-OME) TIFF: the run of directories, from the first, that
	/// share its width, height and tiling and are not flagged as a reduced-resolution or mask
	/// image. A Z-stack is such a run, tiled or not. A pyramid kept in top-level directories
	/// ends the run at its first smaller level, and a thumbnail or mask page ends it too.
	/// Leaves the first directory current.
	inline std::size_t plain_tiff_depth (TIFF* tiff)
	{
		auto shape_of = [tiff] (std::uint32_t& w, std::uint32_t& h)
		{
			w = h = 0;
			TIFFGetField (tiff, TIFFTAG_IMAGEWIDTH, &w);
			TIFFGetField (tiff, TIFFTAG_IMAGELENGTH, &h);
		};

		if (TIFFCurrentDirectory (tiff) != 0)
			TIFFSetDirectory (tiff, 0);
		std::uint32_t w0, h0;
		shape_of (w0, h0);
		const int tiled0 = TIFFIsTiled (tiff);

		// directories are read in sequence, which costs one pass however many there are
		std::size_t n = 1;
		while (TIFFReadDirectory (tiff) == 1)
		{
			std::uint32_t w, h, subfile = 0;
			shape_of (w, h);
			TIFFGetField (tiff, TIFFTAG_SUBFILETYPE, &subfile);
			if (w != w0 || h != h0 || TIFFIsTiled (tiff) != tiled0 || (subfile & (FILETYPE_REDUCEDIMAGE | FILETYPE_MASK)) != 0)
				break;
			++n;
		}

		TIFFSetDirectory (tiff, 0);
		return n;
	}

	/// @brief Make directory 'ifd' current. tdir_t is 16 bits wide before libtiff 4.5 and 32
	/// bits since, so the index is checked against the type rather than cast into it. A
	/// directory that is already current is not re-read, so a 2D scan that reads one plane
	/// tile after tile selects it once.
	/// @param who Loader name for the error message
	inline void select_tiff_directory (TIFF* tiff, std::size_t ifd, const char* who)
	{
		if (ifd > (std::size_t) (std::numeric_limits<tdir_t>::max)())
			throw std::runtime_error (std::string (who) + ": TIFF directory " + std::to_string (ifd)
				+ " exceeds the " + std::to_string ((std::numeric_limits<tdir_t>::max)()) + " this libtiff can address");

		if (TIFFCurrentDirectory (tiff) == (tdir_t) ifd)
			return;

		if (TIFFSetDirectory (tiff, (tdir_t) ifd) != 1)
			throw std::runtime_error (std::string (who) + ": cannot select TIFF directory " + std::to_string (ifd));
	}

	/// @brief Select the directory that holds plane (z,c,t).
	/// @param ome The parsed OME-XML for an OME-TIFF, nullptr for a plain TIFF
	/// @param n_planes_z Z extent of a plain TIFF (see plain_tiff_depth); ignored for an OME-TIFF
	/// @param who Loader name for the error message
	/// Throws when the plane lies outside the file. The OME check matters most: the directory
	/// index is computed from (z,c,t), so an out-of-range coordinate can land on a directory
	/// that exists but belongs to a different plane. A plain TIFF has no C or T axis, so only
	/// its Z is checked.
	inline void select_tiff_plane (TIFF* tiff, const OmeAxes* ome, std::size_t n_planes_z,
		std::size_t z, std::size_t c, std::size_t t, const char* who)
	{
		if (ome != nullptr)
		{
			if (z >= ome->sizeZ || c >= ome->sizeC || t >= ome->sizeT)
				throw std::runtime_error (std::string (who) + ": plane (z,c,t)=(" + std::to_string (z) + "," + std::to_string (c)
					+ "," + std::to_string (t) + ") is outside the OME-TIFF's (" + std::to_string (ome->sizeZ) + ","
					+ std::to_string (ome->sizeC) + "," + std::to_string (ome->sizeT) + ")");

			select_tiff_directory (tiff, ome->ifdForPlane (z, c, t), who);
		}
		else
		{
			if (z >= n_planes_z)
				throw std::runtime_error (std::string (who) + ": plane z=" + std::to_string (z)
					+ " is outside the TIFF's " + std::to_string (n_planes_z) + " directories");

			select_tiff_directory (tiff, z, who);
		}
	}
}
