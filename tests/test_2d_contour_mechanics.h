#pragma once

#include <random>
#include <stdexcept>
#include <type_traits>
#include "test_2d_contour_common.h"

// BorderedContour keeps a reference to the pixels it is built over, so it binds only a vector that
// outlives it: built over a temporary, the reference would dangle once the full expression ends.
// Binding an rvalue is refused at compile time; an lvalue, const or not, is accepted.
void test_2d_contour_bordered_contour_binds_lvalues_mechanics()
{
	ASSERT_FALSE((std::is_constructible_v<BorderedContour, std::vector<Pixel2>&&, const std::vector<size_t>&>));
	ASSERT_FALSE((std::is_constructible_v<BorderedContour, const std::vector<Pixel2>&&, const std::vector<size_t>&>));
	ASSERT_TRUE((std::is_constructible_v<BorderedContour, std::vector<Pixel2>&, const std::vector<size_t>&>));
	ASSERT_TRUE((std::is_constructible_v<BorderedContour, const std::vector<Pixel2>&, const std::vector<size_t>&>));
}

// A contour's border ends have one layout, which the CPU searches (BorderedContour) and the GPU upload
// both read: every border holds at least one pixel, so the ends rise strictly, and the last end is
// the pixel count, so the borders cover the contour. BorderedContour accepts that layout and refuses
// any other, rather than reading pixels past the last end as one more border, which the GPU, holding
// only the borders the ends list, would not search.
void test_2d_contour_bordered_contour_layout_mechanics()
{
	const std::vector<Pixel2> K { Pixel2(0, 0, 1), Pixel2(1, 0, 1), Pixel2(2, 0, 1), Pixel2(5, 5, 1), Pixel2(6, 5, 1) };

	// the layout, and the borders it gives
	const std::vector<size_t> ends { 3, 5 };
	ASSERT_TRUE(border_ends_cover(K.size(), ends));
	const BorderedContour Kb (K, ends);
	ASSERT_EQ(Kb.borders.size(), (size_t)2);
	ASSERT_EQ(Kb.borders[1].begin, (size_t)3);
	ASSERT_EQ(Kb.borders[1].xmin, 5);
	const std::vector<Pixel2> none;
	ASSERT_TRUE(border_ends_cover(0, {}));
	ASSERT_TRUE(BorderedContour(none, {}).borders.empty());

	// pixels past the last end, an empty run, ends past the contour, no ends over pixels
	for (const std::vector<size_t>& bad : { std::vector<size_t>{ 3 }, { 3, 3, 5 }, { 0, 5 }, { 3, 6 }, {} })
	{
		ASSERT_FALSE(border_ends_cover(K.size(), bad));
		ASSERT_THROW(BorderedContour(K, bad), std::invalid_argument);
	}
}

// LR::merge_multicontour lists every contour pixel once, so a border all of whose pixels an earlier
// border holds adds no pixels; it adds no end either, and the ends keep the layout BorderedContour
// and the GPU upload read. Here the second border's pixels all lie on the first.
void test_2d_contour_merge_skips_covered_border_mechanics()
{
	LR r (1);
	r.multicontour_ = { { Pixel2(4, 4, 1), Pixel2(5, 4, 1), Pixel2(5, 5, 1) }, { Pixel2(5, 4, 1), Pixel2(5, 5, 1) }, { Pixel2(9, 9, 1) } };
	r.merge_multicontour();
	ASSERT_EQ(r.contour_pixels_.size(), (size_t)4);
	ASSERT_EQ(r.contour_border_ends_, (std::vector<size_t>{ 3, 4 }));
	ASSERT_TRUE(border_ends_cover(r.contour_pixels_.size(), r.contour_border_ends_));

	// with no covered border every border adds an end
	r.multicontour_ = { { Pixel2(4, 4, 1) }, { Pixel2(9, 9, 1) } };
	r.merge_multicontour();
	ASSERT_EQ(r.contour_border_ends_, (std::vector<size_t>{ 1, 2 }));
}

// The row runs of a mask: maximal stretches of '1' along a row, a stretch cut at a multiple of
// 'tile' counted as two (tile 0: no cut).
static size_t contour_mask_row_runs (const std::vector<std::string>& rows, size_t tile = 0)
{
	size_t n = 0;
	for (const auto& row : rows)
		for (size_t x = 0; x < row.size(); x++)
			if (row[x] == '1' && (x == 0 || row[x - 1] != '1' || (tile && x % tile == 0)))
				n++;
	return n;
}

// The GPU memory plan bounds a ROI's contour borders by twice its row runs: every outer border
// starts at a run's first pixel and every hole's border at a run's last. Random masks with holes
// and fragments keep within it, borders and border ends alike; a lattice of separate pixels, every one
// a run and a border, reaches half of it.
void test_2d_contour_borders_row_runs_bound_mechanics()
{
	std::mt19937 rng (5301);
	std::uniform_real_distribution<double> u (0.0, 1.0);
	for (int i = 0; i < 200; i++)
	{
		const int w = 3 + i % 17, h = 3 + (i * 7) % 13;
		std::vector<std::string> rows (h, std::string (w, '.'));
		const double fill = 0.35 + 0.6 * u (rng);
		for (auto& row : rows)
			for (auto& c : row)
				if (u (rng) < fill)
					c = '1';
		const auto S = contour_mask_pixels (rows, 4, 6);
		if (S.empty())
			continue;
		LR r (1);
		contour_roi_from_pixels (r, S);
		contour_calculate (r);
		const size_t bound = 2 * contour_mask_row_runs (rows);
		SCOPED_TRACE ("mask #" + std::to_string(i));
		ASSERT_LE (r.multicontour_.size(), bound);
		ASSERT_LE (r.contour_border_ends_.size(), bound);
	}

	const std::vector<std::string> lattice { "1.1.1.1", ".......", "1.1.1.1", ".......", "1.1.1.1" },
		riddled { "1111111", "1.1.1.1", "1111111", "1.1.1.1", "1111111" };
	// 12 pixels, 12 runs, 12 borders; 11 runs, the outer border and 6 holes
	for (const auto& [rows, runs, borders] : { std::tuple<const std::vector<std::string>*, size_t, size_t>{ &lattice, 12, 12 }, { &riddled, 11, 7 } })
	{
		LR r (1);
		contour_roi_from_pixels (r, contour_mask_pixels (*rows, 4, 6));
		contour_calculate (r);
		ASSERT_EQ (contour_mask_row_runs (*rows), runs);
		ASSERT_EQ (r.multicontour_.size(), borders);
	}
}

// The 2D prescan counts the slide's row runs (SlideProps::n_row_runs), a run cut by a tile's left
// edge counted as two, for the GPU memory plan's bound on contour borders.
void test_2d_contour_prescan_row_runs_mechanics()
{
	const std::vector<std::string> rows = {
		"..................................",
		"..111111111111111111......1111....",
		"..1111....1111111111......1111....",
		"..1111....11111111111111111111....",
		"..111111111111111111..............",
		"...........................11111..",
	};
	const fs::path dir = fs::temp_directory_path() / "nyxus_contour_row_runs";
	std::error_code ec;
	fs::remove_all (dir, ec);
	fs::create_directories (dir);
	const uint32_t tile = 16;
	const uint32_t W = (uint32_t) rows[0].size(), H = (uint32_t) rows.size();
	write_tiled_plane_u16 (dir / "i.tif", W, H, tile, [] (uint32_t, uint32_t) { return (uint16_t) 5; });
	write_tiled_plane_u16 (dir / "m.tif", W, H, tile, [&] (uint32_t x, uint32_t y) { return (uint16_t) (y < H && x < W && rows[y][x] == '1' ? 3 : 0); });

	SlideProps p ((dir / "i.tif").string(), (dir / "m.tif").string());
	AnisotropyOptions aniso;
	FpImageOptions fpo;
	ASSERT_TRUE(Nyxus::scan_slide_props (p, 2, aniso, false, fpo, false));
	ASSERT_EQ(p.n_rois, (size_t)1);
	ASSERT_EQ(p.n_row_runs, contour_mask_row_runs (rows, tile));
	ASSERT_EQ(p.n_row_runs, (size_t)13);

	// with no mask the slide is one ROI, every row a run, cut at each tile edge
	SlideProps whole ((dir / "i.tif").string(), "");
	ASSERT_TRUE(Nyxus::scan_slide_props (whole, 2, aniso, false, fpo, false));
	ASSERT_EQ(whole.n_row_runs, rows.size() * 3);
	fs::remove_all (dir, ec);
}
