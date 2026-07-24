#include <algorithm>
#include <iostream>
#include <iomanip>
#include <set>
#include <sstream>
#include <unordered_set>
#include "3d_glszm.h"
#include "streaming_ccl.h"
#include "../environment.h"
#include "../helpers/timing.h"

using namespace Nyxus;

void D3_GLSZM_feature::invalidate (double soft_nan)
{
	fv_SAE =
	fv_LAE =
	fv_GLN =
	fv_GLNN =
	fv_SZN =
	fv_SZNN =
	fv_ZP =
	fv_GLV =
	fv_ZV =
	fv_ZE =
	fv_LGLZE =
	fv_HGLZE =
	fv_SALGLE =
	fv_SAHGLE =
	fv_LALGLE =
	fv_LAHGLE = soft_nan;
}

D3_GLSZM_feature::D3_GLSZM_feature() : FeatureMethod("D3_GLSZM_feature")
{
	provide_features(D3_GLSZM_feature::featureset);
}

void D3_GLSZM_feature::osized_add_online_pixel(size_t x, size_t y, uint32_t intensity) {} // Not supporting

// Out-of-core 3D GLSZM. gather_size_zones() is a full 26-connectivity flood fill that greedily
// walks a zone (with explicit backtracking via 'history') and marks voxels VISITED -- workable
// in-core but assumes the whole cube is resident, since a zone can wander in any direction and
// therefore its final size isn't known until the whole volume has been explored.
//
// This streams a standard two-pass ("raster scan") connected-component labeling: process Z-planes
// in ascending order, and within each plane voxels in raster (row,col) order, checking only
// CAUSAL neighbours -- those already fully labeled by the scan order. For 26-connectivity that is
// 4 same-plane neighbours (West, North, North-West, North-East) plus all 9 neighbours in the
// previous (fully resolved) plane -- 13 of the 26, exactly the causal half. The symmetric "future"
// neighbours are not missed: they get discovered from the OTHER voxel's perspective once the scan
// reaches it. A growable union-find (LabelUnionFind, streaming_ccl.h) merges labels that turn out
// to belong to the same zone and accumulates each zone's voxel count (summed on union), so the
// final size is known the instant the volume finishes streaming -- no second voxel pass needed.
// Peak memory is a 2-plane label window (O(area)) plus the label table (bounded by zone count,
// worst case O(volume) for pathological data -- the same class of bound already accepted for
// GLRLM's per-row run-length histograms).
void D3_GLSZM_feature::osized_calculate (LR& r, const Fsettings& s, ImageLoader&)
{
	clear_buffers();

	if (r.aux_min == r.aux_max)
	{
		invalidate (STNGS_NAN(s));
		return;
	}

	int greyInfo = STNGS_GLSZM_GREYDEPTH(s);
	if (STNGS_IBSI(s)) greyInfo = 0;
	PixIntens mn = r.aux_min, mx = r.aux_max;
	const PixIntens bg = TextureFeature::bin_pixel (0, mn, mx, greyInfo);
	// Background/skip value: matlab-aware, matching THIS feature's own convention (calculate()'s
	// zeroI) -- NOT GLRLM's hardcoded 0, which is a different feature's own quirk.
	const PixIntens zeroI = matlab_grey_binning (greyInfo) ? 1 : 0;

	const int W = (int) r.aabb.get_width(),
		H = (int) r.aabb.get_height(),
		Dz = (int) r.aabb.get_z_depth();
	const StatsInt xmin = r.aabb.get_xmin(), ymin = r.aabb.get_ymin(), zmin = r.aabb.get_zmin();
	const size_t nvox = r.raw_voxels_NT.size();
	const bool hasBackground = nvox < (size_t) W * H * Dz;

	// --- grey levels I (mirrors calculate(): ibsi_grey_binning(greyInfo) numeric check for
	// construction -- NOT the STNGS_IBSI(s) compliance flag, which calculate() reserves for row
	// lookup below; conflating the two was the exact bug found and fixed in GLRLM).
	I.clear();
	if (ibsi_grey_binning (greyInfo))
	{
		PixIntens maxbin = 0;
		for (size_t i = 0; i < nvox; i++)
		{
			PixIntens b = TextureFeature::bin_pixel (r.raw_voxels_NT[i].inten, mn, mx, greyInfo);
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
			PixIntens b = TextureFeature::bin_pixel (r.raw_voxels_NT[i].inten, mn, mx, greyInfo);
			if (b != 0) uniq.insert (b);
		}
		I.assign (uniq.begin(), uniq.end());
	}
	const bool ibsiRow = STNGS_IBSI(s);
	auto row_of = [&](PixIntens pi) -> int
	{
		return ibsiRow ? (int) pi - 1 : (int)(std::lower_bound (I.begin(), I.end(), pi) - I.begin());
	};

	// --- streaming 26-connectivity CCL over a 2-plane window
	LabelUnionFind uf;
	std::vector<PixIntens> prevPlane, curPlane;
	std::vector<int> prevLabel, curLabel;
	std::vector<Pixel3> slab;

	static const int SAME_PLANE_CAUSAL[4][2] = { {0,-1}, {-1,0}, {-1,-1}, {-1,1} };	// dy,dx

	for (int z = 0; z < Dz; z++)
	{
		curPlane.assign ((size_t) W * H, bg);
		r.raw_voxels_NT.read_slab ((size_t)(zmin + z), slab);
		for (const auto& v : slab)
		{
			int lx = (int) v.x - (int) xmin, ly = (int) v.y - (int) ymin;
			if (lx >= 0 && lx < W && ly >= 0 && ly < H)
				curPlane[(size_t) ly * W + lx] = TextureFeature::bin_pixel (v.inten, mn, mx, greyInfo);
		}
		curLabel.assign ((size_t) W * H, -1);

		for (int y = 0; y < H; y++)
			for (int x = 0; x < W; x++)
			{
				PixIntens pi = curPlane[(size_t) y * W + x];
				if (pi == zeroI)
					continue;

				int found = -1;
				for (auto& d : SAME_PLANE_CAUSAL)
				{
					int ny = y + d[0], nx = x + d[1];
					if (ny < 0 || ny >= H || nx < 0 || nx >= W) continue;
					if (curPlane[(size_t) ny * W + nx] != pi) continue;
					int lbl = curLabel[(size_t) ny * W + nx];
					if (lbl < 0) continue;
					found = (found < 0) ? lbl : uf.union_sum (found, lbl);
				}
				if (z > 0)
				{
					for (int dy = -1; dy <= 1; dy++)
						for (int dx = -1; dx <= 1; dx++)
						{
							int ny = y + dy, nx = x + dx;
							if (ny < 0 || ny >= H || nx < 0 || nx >= W) continue;
							if (prevPlane[(size_t) ny * W + nx] != pi) continue;
							int lbl = prevLabel[(size_t) ny * W + nx];
							if (lbl < 0) continue;
							found = (found < 0) ? lbl : uf.union_sum (found, lbl);
						}
				}

				if (found < 0)
					found = uf.make_label (pi, 1);
				else
					uf.add_sum (found, 1);
				curLabel[(size_t) y * W + x] = found;
			}

		prevPlane.swap (curPlane);
		prevLabel.swap (curLabel);
	}

	// --- one zone per distinct root
	std::vector<std::pair<PixIntens, int>> Zones;
	std::vector<char> seenRoot (uf.num_labels(), 0);
	for (size_t i = 0; i < uf.num_labels(); i++)
	{
		int rt = uf.find ((int) i);
		if (seenRoot[rt]) continue;
		seenRoot[rt] = 1;
		Zones.push_back ({ uf.intensity_of (rt), (int) uf.metric (rt) });
	}

	int maxZoneArea = 0;
	for (auto& zo : Zones)
		maxZoneArea = (std::max) (maxZoneArea, zo.second);

	Ng = (int) I.size();
	Ns = maxZoneArea;
	Nz = (int) Zones.size();
	Np = (int) nvox;

	P.allocate (Ns, Ng);
	P.fill (0);
	for (auto& zo : Zones)
	{
		int row = row_of (zo.first);
		int col = zo.second - 1;
		P.xy (col, row)++;
	}

	sum_p = 0;
	for (auto a : P) sum_p += a;
	if (sum_p == 0)
	{
		invalidate (STNGS_NAN(s));
		return;
	}

	calc_sums_of_P();
	fv_SAE = calc_SAE();
	fv_LAE = calc_LAE();
	fv_GLN = calc_GLN();
	fv_GLNN = calc_GLNN();
	fv_SZN = calc_SZN();
	fv_SZNN = calc_SZNN();
	fv_ZP = calc_ZP();
	fv_GLV = calc_GLV();
	fv_ZV = calc_ZV();
	fv_ZE = calc_ZE();
	fv_LGLZE = calc_LGLZE();
	fv_HGLZE = calc_HGLZE();
	fv_SALGLE = calc_SALGLE();
	fv_SAHGLE = calc_SAHGLE();
	fv_LALGLE = calc_LALGLE();
	fv_LAHGLE = calc_LAHGLE();
}

/*static*/ void D3_GLSZM_feature::gather_size_zones (std::vector<std::pair<PixIntens, int>> & Zones, SimpleCube <PixIntens> & D, PixIntens zeroI)
{
	size_t w = D.width(),
		h = D.height(),
		d = D.depth();

	// Number of zones
	const int VISITED = -1;
	for (int zslice = 0; zslice < d; zslice++)
	{
		for (int row = 0; row < h; row++)
		{
			for (int col = 0; col < w; col++)
			{
				// Find a non-blank pixel
				PixIntens pi = D.zyx (zslice, row, col);
				if (pi == zeroI || int(pi) == VISITED)
					continue;

				// Found a gray pixel. Find same-intensity neighbourhood of it.
				std::vector<std::tuple<int, int, int>> history; // dimension order: x,y,z
				int x = col, y = row, z = zslice;
				int zoneArea = 1;
				D.zyx (z,y,x) = VISITED;

				int dx, dy, dz;
				for (;;)
				{
					//***** same Z
					dz = 0;
					// 1
					dy = 0;	dx = 1;
					if (D.safe(z+dz, y+dy, x+dx) && D.zyx(z+dz, y+dy, x+dx) != VISITED && D.zyx(z+dz, y+dy, x+dx) == pi)
					{
						D.zyx(z+dz, y+dy, x+dx) = VISITED;
						zoneArea++;

						// Remember this pixel
						history.push_back({ x,y,z });
						// Advance
						z += dz;	y += dy;	x += dx;
						// Proceed
						continue;
					}
					// 2
					dy = 1;	dx = 1;
					if (D.safe(z+dz, y+dy, x+dx) && D.zyx(z+dz, y+dy, x+dx) != VISITED && D.zyx(z+dz, y+dy, x+dx) == pi)
					{
						D.zyx(z+dz, y+dy, x+dx) = VISITED;
						zoneArea++;

						// Remember this pixel
						history.push_back({ x,y,z });
						// Advance
						z += dz;	y += dy;	x += dx;
						// Proceed
						continue;
					}
					// 3
					dy = 1;	dx = 0;
					if (D.safe(z+dz, y+dy, x+dx) && D.zyx(z+dz, y+dy, x+dx) != VISITED && D.zyx(z+dz, y+dy, x+dx) == pi)
					{
						D.zyx(z+dz, y+dy, x+dx) = VISITED;
						zoneArea++;

						// Remember this pixel
						history.push_back({ x,y,z });
						// Advance
						z += dz;	y += dy;	x += dx;
						// Proceed
						continue;
					}
					// 4
					dy = 1;	dx = -1;
					if (D.safe(z+dz, y+dy, x+dx) && D.zyx(z+dz, y+dy, x+dx) != VISITED && D.zyx(z+dz, y+dy, x+dx) == pi)
					{
						D.zyx(z+dz, y+dy, x+dx) = VISITED;
						zoneArea++;

						// Remember this pixel
						history.push_back({ x,y,z });
						// Advance
						z += dz;	y += dy;	x += dx;
						// Proceed
						continue;
					}
					// 5
					dy = 0;	dx = -1;
					if (D.safe(z+dz, y+dy, x+dx) && D.zyx(z+dz, y+dy, x+dx) != VISITED && D.zyx(z+dz, y+dy, x+dx) == pi)
					{
						D.zyx(z+dz, y+dy, x+dx) = VISITED;
						zoneArea++;

						// Remember this pixel
						history.push_back({ x,y,z });
						// Advance
						z += dz;	y += dy;	x += dx;
						// Proceed
						continue;
					}
					// 6
					dy = -1;	dx = -1;
					if (D.safe(z+dz, y+dy, x+dx) && D.zyx(z+dz, y+dy, x+dx) != VISITED && D.zyx(z+dz, y+dy, x+dx) == pi)
					{
						D.zyx(z+dz, y+dy, x+dx) = VISITED;
						zoneArea++;

						// Remember this pixel
						history.push_back({ x,y,z });
						// Advance
						z += dz;	y += dy;	x += dx;
						// Proceed
						continue;
					}
					// 7
					dy = -1;	dx = 0;
					if (D.safe(z+dz, y+dy, x+dx) && D.zyx(z+dz, y+dy, x+dx) != VISITED && D.zyx(z+dz, y+dy, x+dx) == pi)
					{
						D.zyx(z+dz, y+dy, x+dx) = VISITED;
						zoneArea++;

						// Remember this pixel
						history.push_back({ x,y,z });
						// Advance
						z += dz;	y += dy;	x += dx;
						// Proceed
						continue;
					}
					// 8
					dy = -1;	dx = 1;
					if (D.safe(z+dz, y+dy, x+dx) && D.zyx(z+dz, y+dy, x+dx) != VISITED && D.zyx(z+dz, y+dy, x+dx) == pi)
					{
						D.zyx(z+dz, y+dy, x+dx) = VISITED;
						zoneArea++;

						// Remember this pixel
						history.push_back({ x,y,z });
						// Advance
						z += dz;	y += dy;	x += dx;
						// Proceed
						continue;
					}

					//***** upper Z
					dz = 1;
					// 1
					dy = 0;	dx = 1;
					if (D.safe(z+dz, y+dy, x+dx) && D.zyx(z+dz, y+dy, x+dx) != VISITED && D.zyx(z+dz, y+dy, x+dx) == pi)
					{
						D.zyx(z+dz, y+dy, x+dx) = VISITED;
						zoneArea++;

						// Remember this pixel
						history.push_back({ x,y,z });
						// Advance
						z += dz;	y += dy;	x += dx;
						// Proceed
						continue;
					}
					// 2
					dy = 1;	dx = 1;
					if (D.safe(z+dz, y+dy, x+dx) && D.zyx(z+dz, y+dy, x+dx) != VISITED && D.zyx(z+dz, y+dy, x+dx) == pi)
					{
						D.zyx(z+dz, y+dy, x+dx) = VISITED;
						zoneArea++;

						// Remember this pixel
						history.push_back({ x,y,z });
						// Advance
						z += dz;	y += dy;	x += dx;
						// Proceed
						continue;
					}
					// 3
					dy = 1;	dx = 0;
					if (D.safe(z+dz, y+dy, x+dx) && D.zyx(z+dz, y+dy, x+dx) != VISITED && D.zyx(z+dz, y+dy, x+dx) == pi)
					{
						D.zyx(z+dz, y+dy, x+dx) = VISITED;
						zoneArea++;

						// Remember this pixel
						history.push_back({ x,y,z });
						// Advance
						z += dz;	y += dy;	x += dx;
						// Proceed
						continue;
					}
					// 4
					dy = 1;	dx = -1;
					if (D.safe(z+dz, y+dy, x+dx) && D.zyx(z+dz, y+dy, x+dx) != VISITED && D.zyx(z+dz, y+dy, x+dx) == pi)
					{
						D.zyx(z+dz, y+dy, x+dx) = VISITED;
						zoneArea++;

						// Remember this pixel
						history.push_back({ x,y,z });
						// Advance
						z += dz;	y += dy;	x += dx;
						// Proceed
						continue;
					}
					// 5
					dy = 0;	dx = -1;
					if (D.safe(z+dz, y+dy, x+dx) && D.zyx(z+dz, y+dy, x+dx) != VISITED && D.zyx(z+dz, y+dy, x+dx) == pi)
					{
						D.zyx(z+dz, y+dy, x+dx) = VISITED;
						zoneArea++;

						// Remember this pixel
						history.push_back({ x,y,z });
						// Advance
						z += dz;	y += dy;	x += dx;
						// Proceed
						continue;
					}
					// 6
					dy = -1;	dx = -1;
					if (D.safe(z+dz, y+dy, x+dx) && D.zyx(z+dz, y+dy, x+dx) != VISITED && D.zyx(z+dz, y+dy, x+dx) == pi)
					{
						D.zyx(z+dz, y+dy, x+dx) = VISITED;
						zoneArea++;

						// Remember this pixel
						history.push_back({ x,y,z });
						// Advance
						z += dz;	y += dy;	x += dx;
						// Proceed
						continue;
					}
					// 7
					dy = -1;	dx = 0;
					if (D.safe(z+dz, y+dy, x+dx) && D.zyx(z+dz, y+dy, x+dx) != VISITED && D.zyx(z+dz, y+dy, x+dx) == pi)
					{
						D.zyx(z+dz, y+dy, x+dx) = VISITED;
						zoneArea++;

						// Remember this pixel
						history.push_back({ x,y,z });
						// Advance
						z += dz;	y += dy;	x += dx;
						// Proceed
						continue;
					}
					// 8
					dy = -1;	dx = 1;
					if (D.safe(z+dz, y+dy, x+dx) && D.zyx(z+dz, y+dy, x+dx) != VISITED && D.zyx(z+dz, y+dy, x+dx) == pi)
					{
						D.zyx(z+dz, y+dy, x+dx) = VISITED;
						zoneArea++;

						// Remember this pixel
						history.push_back({ x,y,z });
						// Advance
						z += dz;	y += dy;	x += dx;
						// Proceed
						continue;
					}

					//***** lower Z
					dz = -1;
					// 1
					dy = 0;	dx = 1;
					if (D.safe(z+dz, y+dy, x+dx) && D.zyx(z+dz, y+dy, x+dx) != VISITED && D.zyx(z+dz, y+dy, x+dx) == pi)
					{
						D.zyx(z+dz, y+dy, x+dx) = VISITED;
						zoneArea++;

						// Remember this pixel
						history.push_back({ x,y,z });
						// Advance
						z += dz;	y += dy;	x += dx;
						// Proceed
						continue;
					}
					// 2
					dy = 1;	dx = 1;
					if (D.safe(z+dz, y+dy, x+dx) && D.zyx(z+dz, y+dy, x+dx) != VISITED && D.zyx(z+dz, y+dy, x+dx) == pi)
					{
						D.zyx(z+dz, y+dy, x+dx) = VISITED;
						zoneArea++;

						// Remember this pixel
						history.push_back({ x,y,z });
						// Advance
						z += dz;	y += dy;	x += dx;
						// Proceed
						continue;
					}
					// 3
					dy = 1;	dx = 0;
					if (D.safe(z+dz, y+dy, x+dx) && D.zyx(z+dz, y+dy, x+dx) != VISITED && D.zyx(z+dz, y+dy, x+dx) == pi)
					{
						D.zyx(z+dz, y+dy, x+dx) = VISITED;
						zoneArea++;

						// Remember this pixel
						history.push_back({ x,y,z });
						// Advance
						z += dz;	y += dy;	x += dx;
						// Proceed
						continue;
					}
					// 4
					dy = 1;	dx = -1;
					if (D.safe(z+dz, y+dy, x+dx) && D.zyx(z+dz, y+dy, x+dx) != VISITED && D.zyx(z+dz, y+dy, x+dx) == pi)
					{
						D.zyx(z+dz, y+dy, x+dx) = VISITED;
						zoneArea++;

						// Remember this pixel
						history.push_back({ x,y,z });
						// Advance
						z += dz;	y += dy;	x += dx;
						// Proceed
						continue;
					}
					// 5
					dy = 0;	dx = -1;
					if (D.safe(z+dz, y+dy, x+dx) && D.zyx(z+dz, y+dy, x+dx) != VISITED && D.zyx(z+dz, y+dy, x+dx) == pi)
					{
						D.zyx(z+dz, y+dy, x+dx) = VISITED;
						zoneArea++;

						// Remember this pixel
						history.push_back({ x,y,z });
						// Advance
						z += dz;	y += dy;	x += dx;
						// Proceed
						continue;
					}
					// 6
					dy = -1;	dx = -1;
					if (D.safe(z+dz, y+dy, x+dx) && D.zyx(z+dz, y+dy, x+dx) != VISITED && D.zyx(z+dz, y+dy, x+dx) == pi)
					{
						D.zyx(z+dz, y+dy, x+dx) = VISITED;
						zoneArea++;

						// Remember this pixel
						history.push_back({ x,y,z });
						// Advance
						z += dz;	y += dy;	x += dx;
						// Proceed
						continue;
					}
					// 7
					dy = -1;	dx = 0;
					if (D.safe(z+dz, y+dy, x+dx) && D.zyx(z+dz, y+dy, x+dx) != VISITED && D.zyx(z+dz, y+dy, x+dx) == pi)
					{
						D.zyx(z+dz, y+dy, x+dx) = VISITED;
						zoneArea++;

						// Remember this pixel
						history.push_back({ x,y,z });
						// Advance
						z += dz;	y += dy;	x += dx;
						// Proceed
						continue;
					}
					// 8
					dy = -1;	dx = 1;
					if (D.safe(z+dz, y+dy, x+dx) && D.zyx(z+dz, y+dy, x+dx) != VISITED && D.zyx(z+dz, y+dy, x+dx) == pi)
					{
						D.zyx(z+dz, y+dy, x+dx) = VISITED;
						zoneArea++;

						// Remember this pixel
						history.push_back({ x,y,z });
						// Advance
						z += dz;	y += dy;	x += dx;
						// Proceed
						continue;
					}

					//***** strictly upper Z
					dz = 1;	dy = 0;	dx = 0;
					if (D.safe(z+dz, y+dy, x+dx) && D.zyx(z+dz, y+dy, x+dx) != VISITED && D.zyx(z+dz, y+dy, x+dx) == pi)
					{
						D.zyx(z+dz, y+dy, x+dx) = VISITED;
						zoneArea++;

						// Remember this pixel
						history.push_back({ x,y,z });
						// Advance
						z += dz;	y += dy;	x += dx;
						// Proceed
						continue;
					}

					//***** strictly lower Z
					dz = -1;	dy = 0;	dx = 0;
					if (D.safe(z+dz, y+dy, x+dx) && D.zyx(z+dz, y+dy, x+dx) != VISITED && D.zyx(z+dz, y+dy, x+dx) == pi)
					{
						D.zyx(z+dz, y+dy, x+dx) = VISITED;
						zoneArea++;

						// Remember this pixel
						history.push_back({ x,y,z });
						// Advance
						z += dz;	y += dy;	x += dx;
						// Proceed
						continue;
					}

					// Return from the branch
					if (history.size() > 0)
					{
						// Recollect the coordinate where we diverted from
						std::tuple<int, int, int> prev = history[history.size() - 1];
						x = std::get<0>(prev);
						y = std::get<1>(prev);
						z = std::get<2>(prev);
						history.pop_back();
						continue;
					}

					// done exploring this cluster
					break;
				}

				// save this intensity cluster
				std::pair <PixIntens, int> zo = { pi, zoneArea };
				Zones.push_back (zo);
			} // columns
		} // rows
	} // z-slices
}

void D3_GLSZM_feature::calculate (LR& r, const Fsettings& s)
{
	clear_buffers();

	// intercept blank ROIs (equal intensity)
	if (r.aux_min == r.aux_max)
	{
		invalidate (STNGS_NAN(s));
		return;
	}

	// bin intensities
	int w = r.aux_image_cube.width(),
		h = r.aux_image_cube.height(),
		d = r.aux_image_cube.depth();

	SimpleCube <PixIntens> D;
	D.allocate(w, h, d);

	auto greyInfo = STNGS_GLSZM_GREYDEPTH(s); // former Nyxus::theEnvironment.get_coarse_gray_depth()
	if (STNGS_IBSI(s)) // former Nyxus::theEnvironment.ibsi_compliance
		greyInfo = 0;

	bin_intensities_3d (D, r.aux_image_cube, r.aux_min, r.aux_max, greyInfo);

	// gather unique intensities
	std::unordered_set <PixIntens> U;

	if (ibsi_grey_binning(greyInfo))
	{
		// ibsi approach to intensities
		auto n_ibsi_levels = *std::max_element(D.begin(), D.end());
		I.resize(n_ibsi_levels);
		for (int i = 0; i < n_ibsi_levels; i++)
			I[i] = i + 1;
	}
	else
	{
		// radiomics and matlab approach to intensities
		std::unordered_set <PixIntens> U(D.begin(), D.end());
		U.erase(0);	// discard intensity '0'
		I.assign(U.begin(), U.end());
		std::sort(I.begin(), I.end());
	}

	// zero (backround) intensity at given grey binning approach
	PixIntens zeroI = matlab_grey_binning (greyInfo) ? 1 : 0;
		
	// gather intensity zones
	std::vector <std::pair<PixIntens, int>> Zones;
	D3_GLSZM_feature::gather_size_zones (Zones, D, zeroI);

	// width of GLSZM matrix 
	int maxZoneArea = 0;
	for (const std::pair<PixIntens, int> & zo : Zones)
		maxZoneArea = (std::max) (maxZoneArea, zo.second);

	// non-zero pixels
	size_t nnzVoxels = r.raw_pixels_3D.size();

	//==== Fill the SZ-matrix

	Ng = STNGS_IBSI(s) ? *std::max_element (I.begin(), I.end()) : I.size();	// former Environment::ibsi_compliance
	Ns = maxZoneArea;
	Nz = (int) Zones.size();
	Np = nnzVoxels;

	// --allocate GLSZ-matrix
	P.allocate (Ns, Ng);
	P.fill (0);

	// --iterate zones and fill the matrix
	int i = 0;
	for (const auto & zone : Zones)
	{
		// row of P-matrix
		auto iter = std::find (I.begin(), I.end(), zone.first);
		int row = STNGS_IBSI(s) ? zone.first - 1 : int(iter - I.begin());

		// column of P-matrix
		int col = zone.second - 1;	// 0-based => -1
		auto & k = P.xy (col, row);
		k++;
	}

	// normalizing coefficient (must be ==Nz)
	sum_p = 0;
	for (auto a : P)
		sum_p += a;

	// check if the P-matrix is informative
	if (sum_p == 0)
	{
		invalidate (STNGS_NAN(s));
		return;
	}

	// Precalculate sums of P
	calc_sums_of_P();

	// Calculate features
	fv_SAE = calc_SAE();
	fv_LAE = calc_LAE();
	fv_GLN = calc_GLN();
	fv_GLNN = calc_GLNN();
	fv_SZN = calc_SZN();
	fv_SZNN = calc_SZNN();
	fv_ZP = calc_ZP();
	fv_GLV = calc_GLV();
	fv_ZV = calc_ZV();
	fv_ZE = calc_ZE();
	fv_LGLZE = calc_LGLZE();
	fv_HGLZE = calc_HGLZE();
	fv_SALGLE = calc_SALGLE();
	fv_SAHGLE = calc_SAHGLE();
	fv_LALGLE = calc_LALGLE();
	fv_LAHGLE = calc_LAHGLE();
}

void D3_GLSZM_feature::calc_sums_of_P()
{
	// Zero specialized sums
	f_LAHGLE = 0;
	f_LALGLE = 0;
	f_SAHGLE = 0;
	f_SALGLE = 0;
	f_ZE = 0;
	mu_GLV = 0;
	mu_ZV = 0;

	// Reset by-level counters
	si.clear();
	si.resize(Ng + 1);
	std::fill(si.begin(), si.end(), 0.0);

	// Aggregate by grayscale level
	for (int i = 1; i <= Ng; ++i)
	{
		double inten = (double)I[i - 1];
		double sum = 0;
		for (int j = 1; j <= /*Ns*/P.width(); ++j)
		{
			double p = P.matlab(i, j);
			sum += p;

			// Once we're iterating matrix P, let's compute specialized sums
			double i2 = inten * inten,
				j2 = double(j) * double(j);

			f_LAHGLE += p * i2 * j2;
			f_LALGLE += p * j2 / i2;
			f_SAHGLE += p * i2 / j2;
			f_SALGLE += p / (i2 * j2);

			double entrTerm = fast_log10(p / sum_p + EPS) / LOG10_2;
			f_ZE += p / sum_p * entrTerm;

			mu_ZV += p / sum_p * double(j);
			mu_GLV += p / sum_p * double(inten);
		}
		si[i] = sum;
	}

	// Reset by-position counters
	sj.clear();
	sj.resize(Ns + 1);
	std::fill(sj.begin(), sj.end(), 0.0);
	for (int j = 1; j <= /*Ns*/P.width(); ++j)
	{
		double sum = 0;
		for (int i = 1; i <= Ng; ++i)
			sum += P.matlab(i, j);
		sj[j] = sum;
	}
}

void D3_GLSZM_feature::save_value(std::vector<std::vector<double>>& fvals)
{
	fvals[(int)Feature3D::GLSZM_SAE][0] = fv_SAE;
	fvals[(int)Feature3D::GLSZM_LAE][0] = fv_LAE;
	fvals[(int)Feature3D::GLSZM_GLN][0] = fv_GLN;
	fvals[(int)Feature3D::GLSZM_GLNN][0] = fv_GLNN;
	fvals[(int)Feature3D::GLSZM_SZN][0] = fv_SZN;
	fvals[(int)Feature3D::GLSZM_SZNN][0] = fv_SZNN;
	fvals[(int)Feature3D::GLSZM_ZP][0] = fv_ZP;
	fvals[(int)Feature3D::GLSZM_GLV][0] = fv_GLV;
	fvals[(int)Feature3D::GLSZM_ZV][0] = fv_ZV;
	fvals[(int)Feature3D::GLSZM_ZE][0] = fv_ZE;
	fvals[(int)Feature3D::GLSZM_LGLZE][0] = fv_LGLZE;
	fvals[(int)Feature3D::GLSZM_HGLZE][0] = fv_HGLZE;
	fvals[(int)Feature3D::GLSZM_SALGLE][0] = fv_SALGLE;
	fvals[(int)Feature3D::GLSZM_SAHGLE][0] = fv_SAHGLE;
	fvals[(int)Feature3D::GLSZM_LALGLE][0] = fv_LALGLE;
	fvals[(int)Feature3D::GLSZM_LAHGLE][0] = fv_LAHGLE;
}

// 1. Small Area Emphasis
double D3_GLSZM_feature::calc_SAE()
{
#if 0
	// Calculate feature. 'sj' is expected to have been initialized in calc_sums_of_P()
	double f = 0.0;
	for (int j = 1; j <= /*Ns*/P.width(); j++)
	{
		f += sj[j] / (j * j);
	}
	double retval = f / sum_p;
	return retval;
#endif

	double tot = 0;
	for (int i=1; i<=Ng; i++)
	{
		for (int j=1; j<=P.width(); j++)
			tot += double(P.matlab(i,j)) / double(j*j);
	}
	double retval = tot / Nz;
	return retval;
}

// 2. Large Area Emphasis
double D3_GLSZM_feature::calc_LAE()
{
#if 0
	// Calculate feature. 'sj' is expected to have been initialized in calc_sums_of_P()
	double f = 0.0;
	for (int j = 1; j <= /*Ns*/P.width(); j++)
	{
		f += sj[j] * (j * j);
	}
	double retval = f / sum_p;
	return retval;
#endif

	double tot = 0;
	for (int i = 1; i <= Ng; i++)
	{
		for (int j = 1; j <= P.width(); j++)
			tot += double(P.matlab(i,j)) * double(j*j);
	}
	double retval = tot / Nz;
	return retval;
}

// 3. Gray Level Non - Uniformity
double D3_GLSZM_feature::calc_GLN()
{
	// Calculate feature. 'si' is expected to have been initialized in calc_sums_of_P()
	double f = 0.0;
	for (int i = 1; i <= Ng; i++)
	{
		double x = si[i];
		f += x * x;
	}

	double retval = f / sum_p;
	return retval;
}

// 4. Gray Level Non - Uniformity Normalized
double D3_GLSZM_feature::calc_GLNN()
{
	// Calculate feature. 'si' is expected to have been initialized in calc_sums_of_P()
	double f = 0.0;

	for (int i = 1; i <= Ng; i++)
	{
		double x = si[i];
		f += x * x;
	}

	double retval = f / double(sum_p * sum_p);
	return retval;
}

// 5. Size - Zone Non - Uniformity
double D3_GLSZM_feature::calc_SZN()
{
	// Calculate feature. 'sj' is expected to have been initialized in calc_sums_of_P()
	double f = 0.0;
	for (int j = 1; j <= Ns; j++)
	{
		double x = sj[j];
		f += x * x;
	}

	double retval = f / sum_p;
	return retval;
}

// 6. Size - Zone Non - Uniformity Normalized
double D3_GLSZM_feature::calc_SZNN()
{
	// Calculate feature. 'sj' is expected to have been initialized in calc_sums_of_P()
	double f = 0.0;
	for (int j = 1; j <= Ns; j++)
	{
		double x = sj[j];
		f += x * x;
	}

	double retval = f / double(sum_p * sum_p);
	return retval;
}

// 7. Zone Percentage
double D3_GLSZM_feature::calc_ZP()
{
	double retval = sum_p / double(Np);
	return retval;
}

// 8. Gray Level Variance
double D3_GLSZM_feature::calc_GLV()
{
	// Calculate feature. 'mu_GLV' is expected to have been initialized in calc_sums_of_P()
	double f = 0.0;
	for (int i = 1; i <= Ng; i++)
	{
		double inten = (double)I[i - 1];
		for (int j = 1; j <= Ns; j++)
		{
			double d2 = (inten - mu_GLV) * (inten - mu_GLV);
			f += P.matlab(i, j) / sum_p * d2;
		}
	}
	return f;
}

// 9. Zone Variance
double D3_GLSZM_feature::calc_ZV()
{
	// Calculate feature. 'mu_ZV' is expected to have been initialized in calc_sums_of_P()
	double f = 0.0;
	for (int i = 1; i <= Ng; i++)
	{
		for (int j = 1; j <= Ns; j++)
		{
			double mu2 = (double(j) - mu_ZV) * (double(j) - mu_ZV);
			f += P.matlab(i, j) / sum_p * mu2;
		}
	}
	return f;
}

// 10. Zone Entropy
double D3_GLSZM_feature::calc_ZE()
{
	// Calculate feature. 'f_ZE' is expected to have been initialized in calc_sums_of_P()
	double retval = -f_ZE;
	return retval;
}

// 11. Low Gray Level Zone Emphasis
double D3_GLSZM_feature::calc_LGLZE()
{
	// Calculate feature. 'si' is expected to have been initialized in calc_sums_of_P()
	double f = 0.0;
	for (int i = 1; i <= Ng; i++)
	{
		double inten = (double)I[i - 1];
		f += si[i] / (inten * inten);
	}

	double retval = f / sum_p;
	return retval;
}

// 12. High Gray Level Zone Emphasis
double D3_GLSZM_feature::calc_HGLZE()
{
	// Calculate feature. 'si' is expected to have been initialized in calc_sums_of_P()
	double f = 0.0;
	for (int i = 1; i <= Ng; i++)
	{
		double inten = (double)I[i - 1];
		f += si[i] * (inten * inten);
	}

	double retval = f / sum_p;
	return retval;
}

// 13. Small Area Low Gray Level Emphasis
double D3_GLSZM_feature::calc_SALGLE()
{
	// Calculate feature. 'f_SALGLE' is expected to have been initialized in calc_sums_of_P()
	double retval = f_SALGLE / sum_p;
	return retval;
}

// 14. Small Area High Gray Level Emphasis
double D3_GLSZM_feature::calc_SAHGLE()
{
	// Calculate feature. 'f_SAHGLE' is expected to have been initialized in calc_sums_of_P()
	double retval = f_SAHGLE / sum_p;
	return retval;
}

// 15. Large Area Low Gray Level Emphasis
double D3_GLSZM_feature::calc_LALGLE()
{
	// Calculate feature. 'f_LALGLE' is expected to have been initialized in calc_sums_of_P()
	double retval = f_LALGLE / sum_p;
	return retval;
}

// 16. Large Area High Gray Level Emphasis
double D3_GLSZM_feature::calc_LAHGLE()
{
	// Calculate feature. 'f_LAHGLE' is expected to have been initialized in calc_sums_of_P()
	double retval = f_LAHGLE / sum_p;
	return retval;
}

void D3_GLSZM_feature::reduce (size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData, const Fsettings & s, const Dataset & _)
{
	for (auto i = start; i < end; i++)
	{
		int lab = (*ptrLabels)[i];
		LR& r = (*ptrLabelData)[lab];
		D3_GLSZM_feature f;
		f.calculate (r, s);
		f.save_value (r.fvals);
	}
}

/*static*/ void D3_GLSZM_feature::extract (LR& r, const Fsettings& s)
{
	D3_GLSZM_feature f;
	f.calculate (r, s);
	f.save_value (r.fvals);
}

