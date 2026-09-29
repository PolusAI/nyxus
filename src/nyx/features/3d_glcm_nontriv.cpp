#include <algorithm>
#include <set>
#include <vector>
#include "../environment.h"
#include "3d_glcm.h"
#include "3d_ooc_volume.h"
#include "image_matrix_nontriv.h"

using namespace Nyxus;

// The 13 direction shifts of 3d_glcm.cpp
namespace {
	struct Sh { int dx, dy, dz; };
	const Sh SHIFTS[13] = {
		{1, 1, 1}, {1, 1, 0}, {1, 1, -1}, {1, 0, 1}, {1, 0, 0}, {1, 0, -1},
		{1, -1, 1}, {1, -1, 0}, {1, -1, -1}, {0, 1, 1}, {0, 1, 0}, {0, 1, -1}, {0, 0, 1}
	};
}

// Co-occurrence counts are additive and the 13 directions reach at most 'offset' planes in Z, so
// the matrices accumulate over a window of offset+1 streamed planes.
void D3_GLCM_feature::osized_calculate (LR& r, const Fsettings& s, ImageLoader&)
{
	clear_result_buffers();

	const size_t nvox = r.raw_voxels_NT.size();

	// IBSI forces no binning
	int greyInfo = STNGS_IBSI(s) ? 0 : STNGS_GLCM_GREYDEPTH(s);
	bool ibsi = STNGS_IBSI(s);
	double soft_nan = STNGS_NAN(s);
	PixIntens mn = r.aux_min, mx = r.aux_max;

	// background cells carry the binned level of intensity 0, as they do in the in-core cube
	const PixIntens bg = TextureFeature::bin_pixel (0, mn, mx, greyInfo);
	Nyxus::OocBinnedVolume vol (r, [mn, mx, greyInfo](PixIntens v) { return TextureFeature::bin_pixel (v, mn, mx, greyInfo); },
		bg, D3_GLCM_feature::offset + 1);
	const int W = vol.width(), H = vol.height(), Dz = vol.depth();

	// --- the grey levels I. Under radiomics binning they are the unique levels of the whole binned
	// cube, so the background level counts when the bounding box has background.
	PixIntens maxbin = 0;
	std::set<PixIntens> uniq = vol.levels (/*with_background=*/ true, /*drop_zero=*/ true, maxbin);

	I.clear();
	if (radiomics_grey_binning(greyInfo))
		I.assign (uniq.begin(), uniq.end());		// std::set is already sorted ascending
	else if (matlab_grey_binning(greyInfo))
	{
		int n = greyInfo;
		I.resize (n);
		for (int i = 0; i < n; i++) I[i] = i + 1;
	}
	else
	{
		int n = (int) maxbin;					// IBSI: levels 1..max
		I.resize (n);
		for (int i = 0; i < n; i++) I[i] = i + 1;
	}
	const int Ng = (int) I.size();

	// nothing to featurize: 13 blank angles, so the angled vectors and their averages line up
	if (Ng == 0 || nvox == 0)
	{
		P_matrix.allocate (1, 1);
		std::fill (P_matrix.begin(), P_matrix.end(), 0.0);
		for (int k = 0; k < 13; k++)
		{
			sum_p = 0;
			finalize_angle (soft_nan);
		}
		return;
	}

	// --- the 13 co-occurrence matrices
	std::vector<SimpleMatrix<double>> mats (13);
	for (int k = 0; k < 13; k++)
	{
		mats[k].allocate (Ng, Ng);
		std::fill (mats[k].begin(), mats[k].end(), 0.0);
	}

	const int off = D3_GLCM_feature::offset;
	const bool sym = D3_GLCM_feature::symmetric_glcm;

	// grey level -> matrix row; only radiomics binning indexes I by value, the others use level-1
	const bool radiomics = radiomics_grey_binning(greyInfo);
	std::vector<int> rowLUT;
	if (radiomics)
		rowLUT = Nyxus::ooc_row_lut (I, maxbin);

	// one (base, neighbor) pair at M.xy(neighbor, base), and its transpose when the matrix is
	// symmetric: radiomics, no binning, or symmetric_glcm
	auto add_pair = [&](SimpleMatrix<double>& M, PixIntens lvl_base, PixIntens lvl_nbr)
	{
		if (ibsi_grey_binning(greyInfo))
			if (lvl_nbr == 0 || lvl_base == 0)
				return;
		int a = (int) lvl_nbr, b = (int) lvl_base;
		if (radiomics)
		{
			if (a == 0 || b == 0)
				return;
			a = rowLUT[a];
			b = rowLUT[b];
		}
		else { a -= 1; b -= 1; }
		M.xy (a, b) += 1.0;
		if (sym || radiomics || ibsi_grey_binning(greyInfo))
			M.xy (b, a) += 1.0;
	};

	for (int lz = 0; lz < Dz; lz++)
	{
		const std::vector<PixIntens>& cur = vol.plane (lz);
		const std::vector<PixIntens>* farp = (lz >= off) ? &vol.plane (lz - off) : nullptr;

		for (int k = 0; k < 13; k++)
		{
			const int dx = SHIFTS[k].dx * off, dy = SHIFTS[k].dy * off, dz = SHIFTS[k].dz * off;
			SimpleMatrix<double>& M = mats[k];

			if (dz == 0)
			{
				// base and neighbor both on the current plane
				for (int y = 0; y < H; y++)
					for (int x = 0; x < W; x++)
					{
						int nx = x + dx, ny = y + dy;
						if (nx >= 0 && nx < W && ny >= 0 && ny < H)
							add_pair (M, cur[(size_t) y * W + x], cur[(size_t) ny * W + nx]);
					}
			}
			else if (dz < 0)
			{
				// base on current plane, neighbor 'off' planes below (the far plane in the ring)
				if (!farp) continue;
				const std::vector<PixIntens>& far = *farp;
				for (int y = 0; y < H; y++)
					for (int x = 0; x < W; x++)
					{
						int nx = x + dx, ny = y + dy;
						if (nx >= 0 && nx < W && ny >= 0 && ny < H)
							add_pair (M, cur[(size_t) y * W + x], far[(size_t) ny * W + nx]);
					}
			}
			else
			{
				// dz>0: base on the far plane (local z-off), neighbor on the current plane
				if (!farp) continue;
				const std::vector<PixIntens>& far = *farp;
				for (int y = 0; y < H; y++)
					for (int x = 0; x < W; x++)
					{
						int nx = x + dx, ny = y + dy;
						if (nx >= 0 && nx < W && ny >= 0 && ny < H)
							add_pair (M, far[(size_t) y * W + x], cur[(size_t) ny * W + nx]);
					}
			}
		}
	}

	// --- per-direction feature values
	for (int k = 0; k < 13; k++)
	{
		P_matrix = std::move (mats[k]);
		sum_p = 0;
		for (double a : P_matrix) sum_p += a;
		finalize_angle (soft_nan);
	}
}
