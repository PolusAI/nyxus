#include <algorithm>
#include <set>
#include <vector>
#include "../environment.h"
#include "3d_glcm.h"
#include "image_matrix_nontriv.h"

using namespace Nyxus;

// The 13 direction shifts, identical to the in-core path (3d_glcm.cpp)
namespace {
	struct Sh { int dx, dy, dz; };
	const Sh SHIFTS[13] = {
		{1, 1, 1}, {1, 1, 0}, {1, 1, -1}, {1, 0, 1}, {1, 0, 0}, {1, 0, -1},
		{1, -1, 1}, {1, -1, 0}, {1, -1, -1}, {0, 1, 1}, {0, 1, 0}, {0, 1, -1}, {0, 0, 1}
	};
}

// Out-of-core 3D GLCM. Instead of scanning the in-RAM grey-binned cube, this builds the 13
// co-occurrence matrices by streaming the disk-backed voxel cloud one Z-plane at a time through a
// sliding window of (offset+1) dense grey-binned planes (co-occurrence counts are additive and the
// 13 directions only reach +/- offset in Z). The per-direction feature math is then the shared
// finalize_angle(), so the values are identical to calculate().
void D3_GLCM_feature::osized_calculate (LR& r, const Fsettings& s, ImageLoader&)
{
	clear_result_buffers();

	const size_t nvox = r.raw_voxels_NT.size();

	// grey-binning mode, mirroring calculate(): IBSI forces the "no binning" (==0) mode
	int greyInfo = STNGS_IBSI(s) ? 0 : STNGS_GLCM_GREYDEPTH(s);
	bool ibsi = STNGS_IBSI(s);
	double soft_nan = STNGS_NAN(s);
	PixIntens mn = r.aux_min, mx = r.aux_max;

	const int W = (int) r.aabb.get_width(),
		H = (int) r.aabb.get_height(),
		Dz = (int) r.aabb.get_z_depth();
	const StatsInt xmin = r.aabb.get_xmin(),
		ymin = r.aabb.get_ymin(),
		zmin = r.aabb.get_zmin();

	// Background cells of the in-core cube are grey-binned too (0 -> 1 for matlab, 0 otherwise);
	// reproduce that so a dense plane matches a Z-slice of the binned cube exactly.
	const PixIntens bg = TextureFeature::bin_pixel (0, mn, mx, greyInfo);

	// --- pass 1: the global set of grey levels (I) + matrix dimension, as in calculateCoocMatAtAngle,
	// which builds I from the unique values of the WHOLE binned cube (mask + background) for the
	// radiomics branch -- so a background bin (e.g. matlab binning maps raw-0 to bin 1) must be
	// included too when the ROI bbox actually has background voxels.
	const bool hasBackground = nvox < (size_t) r.aabb.get_width() * r.aabb.get_height() * r.aabb.get_z_depth();
	std::set<PixIntens> uniq;
	PixIntens maxbin = 0;
	if (hasBackground && bg != 0)
	{
		uniq.insert (bg);
		maxbin = bg;
	}
	for (size_t i = 0; i < nvox; i++)
	{
		Pixel3 v = r.raw_voxels_NT[i];
		PixIntens b = TextureFeature::bin_pixel (v.inten, mn, mx, greyInfo);
		if (b > maxbin) maxbin = b;
		if (b != 0) uniq.insert (b);
	}

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

	// Nothing to featurize (empty/blank): emit 13 blank angles, matching the in-core path's
	// per-direction soft-NaN pushes so the angled vectors and their averages line up.
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

	// --- pass 2: accumulate the 13 co-occurrence matrices over a sliding Z window
	std::vector<SimpleMatrix<double>> mats (13);
	for (int k = 0; k < 13; k++)
	{
		mats[k].allocate (Ng, Ng);
		std::fill (mats[k].begin(), mats[k].end(), 0.0);
	}

	const int off = D3_GLCM_feature::offset;
	const bool sym = D3_GLCM_feature::symmetric_glcm;

	// O(1) grey-level -> matrix row, replacing the per-pair binary search in the hot add_pair loop
	// (radiomics is the only binning branch that indexes I by value; the others use pi-1 directly).
	// rowLUT[v] == lower_bound(I, v) - I.begin() for every possible binned level, so the row is
	// identical to the search it replaces.
	const bool radiomics = radiomics_grey_binning(greyInfo);
	std::vector<int> rowLUT;
	if (radiomics)
	{
		rowLUT.assign ((size_t) maxbin + 1, 0);
		for (PixIntens v = 0; v <= maxbin; v++)
			rowLUT[v] = (int) (std::lower_bound (I.begin(), I.end(), v) - I.begin());
	}

	// Count one (base, neighbor) grey-level pair into matrix M, mirroring calculateCoocMatAtAngle:
	// GLCM.xy(idx(neighbor), idx(base)), plus the symmetric transpose for radiomics/ibsi/symmetric.
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

	// Ring of (off+1) dense grey-binned planes, indexed by local z modulo (off+1)
	const int ring = off + 1;
	std::vector<std::vector<PixIntens>> planes (ring);
	std::vector<Pixel3> slab;

	for (int lz = 0; lz < Dz; lz++)
	{
		std::vector<PixIntens>& cur = planes[lz % ring];
		cur.assign ((size_t) W * H, bg);
		r.raw_voxels_NT.read_slab ((size_t)(zmin + lz), slab);
		for (const auto& v : slab)
		{
			int lx = (int) v.x - (int) xmin, ly = (int) v.y - (int) ymin;
			if (lx >= 0 && lx < W && ly >= 0 && ly < H)
				cur[(size_t) ly * W + lx] = TextureFeature::bin_pixel (v.inten, mn, mx, greyInfo);
		}

		const std::vector<PixIntens>* farp = (lz >= off) ? &planes[(lz - off) % ring] : nullptr;

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

	// --- per-direction feature values from the accumulated matrices (shared finalize).
	// Move each matrix into P_matrix rather than copy it -- mats[k] is not reused afterwards.
	for (int k = 0; k < 13; k++)
	{
		P_matrix = std::move (mats[k]);
		sum_p = 0;
		for (double a : P_matrix) sum_p += a;
		finalize_angle (soft_nan);
	}
}
