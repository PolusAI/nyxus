#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#ifdef __APPLE__
    #define uint64 uint64_hack_
    #define int64 int64_hack_
    #include <tiffio.h>
    #undef uint64
    #undef int64
#else
    #include <tiffio.h>
#endif

namespace Nyxus
{
	/// @brief The image rows of one tile row of a striped TIFF directory, decoded once and kept for
	/// every tile of that row.
	///
	/// A strip loader addresses its image in tiles, so the tiles of one tile row all need the same
	/// image rows, each a different column segment of them. The band holds those rows whole: the
	/// first tile of a tile row decodes them and every other tile of the row copies its columns out
	/// of the band without touching libtiff.
	///
	/// Its footprint is one tile row of scanlines, tile height x image width x bytes per sample --
	/// 1024 x 60000 x 2 bytes, about 123 MB, for a 60,000-pixel-wide uint16 slide -- held for the
	/// loader's lifetime. Every strip loader holds its own band, so a featurize pass that reads an
	/// intensity and a mask image holds two, and ram_limit does not count either of them.
	///
	/// libtiff reaches an arbitrary row of a strip only in an uncompressed file; a codec decodes a
	/// strip from its first row forward and cannot skip rows it has not decoded, although it can
	/// restart a strip from its first row. So in a compressed directory a band is decoded from where
	/// the decoder stands when that is inside the band's first strip at or before the band's first
	/// row -- the case for the next tile row of a top-to-bottom scan, which therefore decodes every
	/// row of the image once -- and otherwise from the first row of that strip, the rows above the
	/// band decoded and dropped. Selecting a directory leaves no strip decoded, so the first band of
	/// a directory always starts at a strip boundary.
	class TiffStripBand
	{
	public:
		/// @brief Make rows [startRow, endRow) of the current directory available through row().
		/// Decodes nothing when the band already holds exactly those rows of that directory.
		/// @param failedRow, errcode Set to the row and the TIFFReadScanline result when a read fails
		/// @return false when a scanline does not read; the band then holds no rows
		bool load (TIFF* tiff, std::uint32_t startRow, std::uint32_t endRow, std::uint32_t& failedRow, int& errcode)
		{
			const tdir_t dir = TIFFCurrentDirectory (tiff);
			if (valid_ && dir == dir_ && startRow == start_ && endRow == end_)
				return true;

			valid_ = false;
			scanline_szb_ = (std::size_t) TIFFScanlineSize (tiff);
			band_.resize ((std::size_t) (endRow - startRow) * scanline_szb_);

			std::uint32_t firstRow = startRow;
			std::uint16_t compression = COMPRESSION_NONE;
			std::uint32_t rowsPerStrip = 0;
			TIFFGetFieldDefaulted (tiff, TIFFTAG_COMPRESSION, &compression);
			TIFFGetFieldDefaulted (tiff, TIFFTAG_ROWSPERSTRIP, &rowsPerStrip);
			if (compression != COMPRESSION_NONE && rowsPerStrip != 0)
			{
				const std::uint32_t stripFirstRow = startRow - startRow % rowsPerStrip,
					decoderRow = TIFFCurrentRow (tiff);
				const bool inStrip = TIFFCurrentStrip (tiff) == TIFFComputeStrip (tiff, startRow, 0)
					&& decoderRow >= stripFirstRow && decoderRow <= startRow;
				firstRow = inStrip ? decoderRow : stripFirstRow;
			}

			// rows above startRow only bring the codec up to startRow; they land in the slot of
			// startRow, which is read after them
			for (std::uint32_t r = firstRow; r < endRow; r++)
			{
				std::uint8_t* dst = band_.data() + (std::size_t) (r < startRow ? 0 : r - startRow) * scanline_szb_;
				errcode = TIFFReadScanline (tiff, dst, r);
				if (errcode != 1)
				{
					failedRow = r;
					return false;
				}
				scanlines_decoded_++;
			}

			dir_ = dir;
			start_ = startRow;
			end_ = endRow;
			valid_ = true;
			return true;
		}

		/// @brief Image row 'r' of the loaded band, startRow <= r < endRow
		const std::uint8_t* row (std::uint32_t r) const { return band_.data() + (std::size_t) (r - start_) * scanline_szb_; }

		std::size_t scanline_bytes() const { return scanline_szb_; }

		/// @brief Scanlines decoded since construction, the rows decoded and dropped included
		std::size_t scanlines_decoded() const { return scanlines_decoded_; }

	private:
		std::vector<std::uint8_t> band_;
		std::size_t scanline_szb_ = 0,
			scanlines_decoded_ = 0;
		tdir_t dir_ = 0;
		std::uint32_t start_ = 0,
			end_ = 0;
		bool valid_ = false;
	};
}
