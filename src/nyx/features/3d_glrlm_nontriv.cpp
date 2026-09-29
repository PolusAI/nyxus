#include <algorithm>
#include <set>
#include <utility>
#include <vector>
#include "../environment.h"
#include "3d_glrlm.h"
#include "3d_ooc_volume.h"
#include "image_matrix_nontriv.h"

using namespace Nyxus;

// The 13 direction shifts of 3d_glrlm.cpp: dz in {0,1}, one of each direction and its opposite.
// AngleShift order is {dz,dy,dx}.
static const AngleShift SHIFTS13[] =
{
	{1,  1,  1}, {1,  1,  0}, {1,  1, -1}, {1,  0,  1}, {1,  0,  0}, {1,  0, -1},
	{1, -1,  1}, {1, -1,  0}, {1, -1, -1}, {0,  1,  1}, {0,  1,  0}, {0,  1, -1}, {0,  0,  1}
};

// A dz==0 run never leaves its plane, so gather_rl_zones() runs on each streamed plane as a depth-1
// cube. A dz==1 run takes exactly one voxel per plane, so its length is carried across a 2-plane
// window, indexed by the current plane's (y,x), and recorded when the run fails to continue.
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

	// the ROI's binned cube, streamed: a cross-plane run reaches one plane back, so the window
	// keeps 2 planes
	Nyxus::OocBinnedVolume vol (r, [minI, maxI, greyInfo](PixIntens v) { return TextureFeature::bin_pixel (v, minI, maxI, greyInfo); }, bg, 2);
	const int W = vol.width(), H = vol.height(), Dz = vol.depth();
	const size_t nvox = vol.n_voxels();

	// --- grey levels I: 1..max under no binning, else the unique levels of the whole binned cube,
	// background included, minus 0. The construction follows the binning mode, ibsi_grey_binning();
	// the row lookup below follows the IBSI flag, and the two differ at GREYDEPTH=0 without IBSI.
	std::vector<PixIntens> I;
	if (ibsi_grey_binning (greyInfo))
	{
		PixIntens maxbin = vol.max_level();
		I.resize (maxbin);
		for (int i = 0; i < (int) maxbin; i++) I[i] = i + 1;
	}
	else
	{
		PixIntens maxbin = 0;
		std::set<PixIntens> uniq = vol.levels (/*with_background=*/ true, /*drop_zero=*/ true, maxbin);
		I.assign (uniq.begin(), uniq.end());
	}
	const int Ng = (int) I.size();
	const size_t Np = nvox;

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
			for (int z = 0; z < Dz; z++)
			{
				SimpleCube<PixIntens> D1 (vol.plane (z), W, H, 1);
				std::vector<std::pair<PixIntens, int>> zones;
				D3_GLRLM_feature::gather_rl_zones (zones, ash, D1, /*zeroI=*/ 0);
				for (auto& zo : zones)
					finalize (zo.first, zo.second);
			}
		}
		else
		{
			// A run at (y,x) in plane z continues the one at (y-dy, x-dx) in plane z-1 when the levels
			// match; a run in plane z-1 that nothing continued has ended.
			std::vector<int> prevCarry, curCarry;

			for (int z = 0; z < Dz; z++)
			{
				const std::vector<PixIntens>& curPlane = vol.plane (z);
				const std::vector<PixIntens>* prevPlane = (z > 0) ? &vol.plane (z - 1) : nullptr;
				curCarry.assign ((size_t) W * H, 0);

				std::vector<char> consumedPrev;
				if (z > 0)
					consumedPrev.assign ((size_t) W * H, 0);

				for (int y = 0; y < H; y++)
					for (int x = 0; x < W; x++)
					{
						PixIntens pi = curPlane[(size_t) y * W + x];
						if (pi == 0)	// gather_rl_zones's zeroI
							continue;

						int py = y - ash.dy, px = x - ash.dx;
						int length = 1;
						if (z > 0 && py >= 0 && py < H && px >= 0 && px < W
							&& (*prevPlane)[(size_t) py * W + px] == pi)
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
								finalize ((*prevPlane)[idx], prevCarry[idx]);
						}
				}

				prevCarry.swap (curCarry);
			}
			// the runs still open at the last plane
			if (Dz > 0)
			{
				const std::vector<PixIntens>& lastPlane = vol.plane (Dz - 1);
				for (int y = 0; y < H; y++)
					for (int x = 0; x < W; x++)
					{
						size_t idx = (size_t) y * W + x;
						if (prevCarry[idx] > 0)
							finalize (lastPlane[idx], prevCarry[idx]);
					}
			}
		}

		// --- the direction's matrix
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
