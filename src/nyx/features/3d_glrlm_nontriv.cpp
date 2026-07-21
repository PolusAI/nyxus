#include <algorithm>
#include <set>
#include <utility>
#include <vector>
#include "../environment.h"
#include "3d_glrlm.h"
#include "image_matrix_nontriv.h"

using namespace Nyxus;

// Same 13 canonical direction shifts as 3d_glrlm.cpp (dz in {0,1} only: half of the 26-connected
// directions, avoiding double-counting an axis and its opposite). AngleShift order is {dz,dy,dx}.
static const AngleShift SHIFTS13[] =
{
	{1,  1,  1}, {1,  1,  0}, {1,  1, -1}, {1,  0,  1}, {1,  0,  0}, {1,  0, -1},
	{1, -1,  1}, {1, -1,  0}, {1, -1, -1}, {0,  1,  1}, {0,  1,  0}, {0,  1, -1}, {0,  0,  1}
};

// Out-of-core 3D GLRLM. gather_rl_zones() greedily walks each run forward along its direction,
// consuming voxels via an in-RAM VISITED marker -- workable in-core (whole cube available) but not
// as-is out-of-core. Two cases, driven by the fact all 13 canonical directions have dz in {0,1}:
//
//  - dz==0 (3 directions): the run never leaves its Z-plane, so gather_rl_zones() is called
//    UNCHANGED on a depth-1 SimpleCube built from one streamed dense plane -- byte-identical to the
//    in-core algorithm, just fed one plane at a time.
//  - dz==1 (10 directions): a run advances exactly one (dz,dy,dx) step per Z-plane, so it visits at
//    most one voxel per plane. This lets a "carry" array (indexed by the CURRENT plane's (y,x))
//    track each in-progress run's length across a 2-plane window, finalizing (recording into the
//    grey/length histogram) a run the moment its continuation check fails, instead of needing the
//    whole volume. Peak memory is O(plane area), not O(volume).
//
// Grey-level LUT and histogram-fill both reuse calc_SRE()/etc unchanged once a full SimpleMatrix<int>
// is assembled per direction, so values are identical to calculate().
void D3_GLRLM_feature::osized_calculate (LR& r, const Fsettings& s, ImageLoader&)
{
	n_angles_ = (int) (sizeof(SHIFTS13) / sizeof(AngleShift));
	clear_buffers();

	PixIntens minI = r.aux_min, maxI = r.aux_max;
	if (minI == maxI)
	{
		double w = STNGS_NAN(s);
		angled_SRE.assign (n_angles_, w); angled_LRE.assign (n_angles_, w);
		angled_GLN.assign (n_angles_, w); angled_GLNN.assign (n_angles_, w);
		angled_RLN.assign (n_angles_, w); angled_RLNN.assign (n_angles_, w);
		angled_RP.assign (n_angles_, w); angled_GLV.assign (n_angles_, w);
		angled_RV.assign (n_angles_, w); angled_RE.assign (n_angles_, w);
		angled_LGLRE.assign (n_angles_, w); angled_HGLRE.assign (n_angles_, w);
		angled_SRLGLE.assign (n_angles_, w); angled_SRHGLE.assign (n_angles_, w);
		angled_LRLGLE.assign (n_angles_, w); angled_LRHGLE.assign (n_angles_, w);
		return;
	}

	int greyInfo = STNGS_IBSI(s) ? 0 : STNGS_GLRLM_GREYDEPTH(s);
	const bool ibsi = STNGS_IBSI(s);
	const PixIntens bg = TextureFeature::bin_pixel (0, minI, maxI, greyInfo);

	const int W = (int) r.aabb.get_width(),
		H = (int) r.aabb.get_height(),
		Dz = (int) r.aabb.get_z_depth();
	const StatsInt xmin = r.aabb.get_xmin(), ymin = r.aabb.get_ymin(), zmin = r.aabb.get_zmin();
	const size_t nvox = r.raw_voxels_NT.size();
	const bool hasBackground = nvox < (size_t) W * H * Dz;

	// --- grey levels I (mirrors calculate(): unique values of the WHOLE binned cube, incl.
	// background, minus 0 -- for IBSI a 1..max linspace instead). Ng == I.size() in both cases.
	// NOTE: the in-core path's I-construction branches on ibsi_grey_binning(greyInfo) (the numeric
	// "no rescale" binning mode), NOT the STNGS_IBSI(s) compliance flag used below for row lookup --
	// under default settings greyInfo==0 (ibsi_grey_binning true) while STNGS_IBSI(s) is false, so
	// conflating the two picks the wrong (sparse unique-set) branch here.
	std::vector<PixIntens> I;
	if (ibsi_grey_binning (greyInfo))
	{
		PixIntens maxbin = 0;
		for (size_t i = 0; i < nvox; i++)
		{
			PixIntens b = TextureFeature::bin_pixel (r.raw_voxels_NT[i].inten, minI, maxI, greyInfo);
			if (b > maxbin) maxbin = b;
		}
		I.resize (maxbin);
		for (int i = 0; i < (int) maxbin; i++) I[i] = i + 1;
	}
	else
	{
		std::set<PixIntens> uniq;
		if (hasBackground && bg != 0)
			uniq.insert (bg);
		for (size_t i = 0; i < nvox; i++)
		{
			PixIntens b = TextureFeature::bin_pixel (r.raw_voxels_NT[i].inten, minI, maxI, greyInfo);
			if (b != 0) uniq.insert (b);
		}
		I.assign (uniq.begin(), uniq.end());
	}
	const int Ng = (int) I.size();
	const size_t Np = nvox;	// matches r.raw_pixels_3D.size() in the in-core path

	auto row_of = [&](PixIntens pi) -> int
	{
		return ibsi ? (int) pi - 1 : (int)(std::lower_bound (I.begin(), I.end(), pi) - I.begin());
	};

	// --- stream one direction at a time
	for (const AngleShift& ash : SHIFTS13)
	{
		std::vector<std::vector<int>> counts (Ng > 0 ? Ng : 1);	// counts[row][length-1] = run count

		auto finalize = [&](PixIntens pi, int length)
		{
			if (length <= 0 || Ng == 0) return;
			int row = row_of (pi);
			if (row < 0 || row >= Ng) return;
			if ((int) counts[row].size() < length)
				counts[row].resize (length, 0);
			counts[row][length - 1]++;
		};

		if (ash.dz == 0)
		{
			// In-plane run: reuse the in-core algorithm verbatim on a depth-1 sub-cube per plane.
			std::vector<PixIntens> plane;
			std::vector<Pixel3> slab;
			for (int z = 0; z < Dz; z++)
			{
				plane.assign ((size_t) W * H, bg);
				r.raw_voxels_NT.read_slab ((size_t)(zmin + z), slab);
				for (const auto& v : slab)
				{
					int lx = (int) v.x - (int) xmin, ly = (int) v.y - (int) ymin;
					if (lx >= 0 && lx < W && ly >= 0 && ly < H)
						plane[(size_t) ly * W + lx] = TextureFeature::bin_pixel (v.inten, minI, maxI, greyInfo);
				}

				SimpleCube<PixIntens> D1 (plane, W, H, 1);
				std::vector<std::pair<PixIntens, int>> zones;
				D3_GLRLM_feature::gather_rl_zones (zones, ash, D1, /*zeroI=*/ 0);
				for (auto& zo : zones)
					finalize (zo.first, zo.second);
			}
		}
		else
		{
			// Cross-plane run (dz==1): carry each in-progress run's length across a 2-plane window,
			// indexed by the CURRENT plane's (y,x). A run at (y,x) in plane z has its predecessor at
			// (y-dy, x-dx) in plane z-1; if intensities match, the run continues (length+1), else the
			// PREDECESSOR's run just ended and is finalized. Whatever remains unconsumed after the
			// last plane is finalized once more after the loop.
			std::vector<PixIntens> prevPlane, curPlane;
			std::vector<int> prevCarry, curCarry;
			std::vector<Pixel3> slab;

			for (int z = 0; z < Dz; z++)
			{
				curPlane.assign ((size_t) W * H, bg);
				r.raw_voxels_NT.read_slab ((size_t)(zmin + z), slab);
				for (const auto& v : slab)
				{
					int lx = (int) v.x - (int) xmin, ly = (int) v.y - (int) ymin;
					if (lx >= 0 && lx < W && ly >= 0 && ly < H)
						curPlane[(size_t) ly * W + lx] = TextureFeature::bin_pixel (v.inten, minI, maxI, greyInfo);
				}
				curCarry.assign ((size_t) W * H, 0);

				std::vector<char> consumedPrev;
				if (z > 0)
					consumedPrev.assign ((size_t) W * H, 0);

				for (int y = 0; y < H; y++)
					for (int x = 0; x < W; x++)
					{
						PixIntens pi = curPlane[(size_t) y * W + x];
						if (pi == 0)	// background/skip: matches gather_rl_zones's zeroI==0 check
							continue;

						int py = y - ash.dy, px = x - ash.dx;
						int length = 1;
						if (z > 0 && py >= 0 && py < H && px >= 0 && px < W
							&& prevPlane[(size_t) py * W + px] == pi)
						{
							length = prevCarry[(size_t) py * W + px] + 1;
							consumedPrev[(size_t) py * W + px] = 1;
						}
						curCarry[(size_t) y * W + x] = length;
					}

				if (z > 0)
				{
					for (int y = 0; y < H; y++)
						for (int x = 0; x < W; x++)
						{
							size_t idx = (size_t) y * W + x;
							if (prevCarry[idx] > 0 && !consumedPrev[idx])
								finalize (prevPlane[idx], prevCarry[idx]);
						}
				}

				prevPlane.swap (curPlane);
				prevCarry.swap (curCarry);
			}
			// Finalize whatever is still active after the last plane
			for (int y = 0; y < H; y++)
				for (int x = 0; x < W; x++)
				{
					size_t idx = (size_t) y * W + x;
					if (prevCarry[idx] > 0)
						finalize (prevPlane[idx], prevCarry[idx]);
				}
		}

		// --- assemble the direction's matrix from counts and reuse the shared feature math
		int Nr = 0;
		for (auto& row : counts)
			Nr = (std::max) (Nr, (int) row.size());

		P_matrix P;
		P.allocate (Nr, Ng);
		std::fill (P.begin(), P.end(), 0);
		for (int row = 0; row < Ng; row++)
			for (int col = 0; col < (int) counts[row].size(); col++)
				P.xy (col, row) = counts[row][col];

		double sum = 0;
		for (auto p : P) sum += p;

		angled_SRE.push_back (calc_SRE (P, sum));
		angled_LRE.push_back (calc_LRE (P, sum));
		angled_GLN.push_back (calc_GLN (P, sum));
		angled_GLNN.push_back (calc_GLNN (P, sum));
		angled_RLN.push_back (calc_RLN (P, sum));
		angled_RLNN.push_back (calc_RLNN (P, sum));
		angled_RP.push_back (Np > 0 ? sum / double(Np) : STNGS_NAN(s));
		angled_GLV.push_back (calc_GLV (P, I, sum));
		angled_RV.push_back (calc_RV (P, sum));
		angled_RE.push_back (calc_RE (P, sum));
		angled_LGLRE.push_back (calc_LGLRE (P, I, sum));
		angled_HGLRE.push_back (calc_HGLRE (P, I, sum));
		angled_SRLGLE.push_back (calc_SRLGLE (P, I, sum));
		angled_SRHGLE.push_back (calc_SRHGLE (P, I, sum));
		angled_LRLGLE.push_back (calc_LRLGLE (P, I, sum));
		angled_LRHGLE.push_back (calc_LRHGLE (P, I, sum));
	}
}
