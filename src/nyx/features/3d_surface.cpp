#define _USE_MATH_DEFINES	// for M_PI, etc.
#include <limits>
#include <regex>
#include "../featureset.h"
#include "../environment.h"	
#include "../3rdparty/quickhull.hpp"
#include "../3rdparty/dsyevj3.h"
#include "3d_mesh.h"
#include "3d_surface.h"

namespace
{
	// The covariance of the physical coordinates from that of the lattice ones
	void scale_covariance (double K[3][3], double sx, double sy, double sz)
	{
		const double s[3] = { sx, sy, sz };
		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 3; j++)
				K[i][j] *= s[i] * s[j];
	}
}

bool D3_SurfaceFeature::required (const FeatureSet & fs)
{
	return fs.anyEnabled (D3_SurfaceFeature::featureset);
}

D3_SurfaceFeature::D3_SurfaceFeature() : FeatureMethod("D3_SurfaceFeature")
{
	provide_features (D3_SurfaceFeature::featureset);
}

// Volume of the convex hull of P, in the units of P cubed. P holds integer lattice points (the doubled
// mesh vertices), so every decision quick_hull makes, and every term of the volume sum, is exact.
double D3_SurfaceFeature::convex_hull_volume (const std::vector<std::array<double, 3>>& P)
{
	constexpr std::size_t dim = 3;
	using Points = std::vector<std::array<double, dim>>;

	// Fewer than four points span no volume. An empty P also cannot be handed to quick_hull, whose
	// affine basis reads the first point.
	if (P.size() < dim + 1)
		return 0.0;

	double maxcoord = 1.0;
	for (const auto& p : P)
		maxcoord = std::max ({ maxcoord, std::abs(p[0]), std::abs(p[1]), std::abs(p[2]) });

	// eps is the tolerance on "is this point outside the facet plane", so it has to sit above the
	// rounding error of the distances it judges and below the smallest real gap between a lattice point
	// and a facet. That rounding error scales with the coordinate magnitude, so eps is derived from the
	// cloud's own extent rather than fixed: a constant tolerance is only valid at one ROI size, and
	// below the arithmetic's resolution the predicate decides on noise and the facet set -- with it
	// the hull volume -- stops being reproducible across toolchains.
	const double eps = 16.0 * std::numeric_limits<double>::epsilon() * maxcoord;
	quick_hull<typename Points::const_iterator> qh{ dim, eps };
	qh.add_points(std::cbegin(P), std::cend(P));
	auto initial_simplex = qh.get_affine_basis();

	// Points confined to a plane or a line span no volume
	if (initial_simplex.size() < dim + 1)
		return 0.0;

	qh.create_initial_simplex(std::cbegin(initial_simplex), std::prev(std::cend(initial_simplex)));
	qh.create_convex_hull();

	// Signed tetrahedra against a vertex of the hull. Every coordinate difference is an integer, so
	// each triple product is exact while it stays under 2^53, and so is their sum, whatever order the
	// facets come in and wherever the ROI sits in the image.
	const double* o = nullptr;
	double v6 = 0.0;
	for (const auto& f : qh.facets_)
	{
		const auto& V = f.vertices_;
		if (! o)
			o = (*V[0]).data();
		double a[3], b[3], c[3];
		for (int i = 0; i < 3; i++)
		{
			a[i] = (*V[0])[i] - o[i];
			b[i] = (*V[1])[i] - o[i];
			c[i] = (*V[2])[i] - o[i];
		}
		v6 += a[0] * (b[1] * c[2] - b[2] * c[1])
			- a[1] * (b[0] * c[2] - b[2] * c[0])
			+ a[2] * (b[0] * c[1] - b[1] * c[0]);
	}
	return std::abs (v6) / 6.0;
}

void D3_SurfaceFeature::calculate (LR& r, const Fsettings& s)
{
	// is shape data non-informative ?
	if (r.raw_pixels_3D.size() == 0)
	{
		cleanup_instance();
		return;
	}

	const double sx = r.spacing_x, sy = r.spacing_y, sz = r.spacing_z;

	if (STNGS_SINGLEROI(s))	// former Nyxus::theEnvironment.singleROI
	{
		set_whole_box (r.aabb.get_width(), r.aabb.get_height(), r.aabb.get_z_depth(), sx, sy, sz);
		return;
	}

	// volume

	// (fast approximation based on cubic lattice packaging of balls). The per-voxel packed
	// volume is a constant, so the whole sum is voxel_count * that constant (the out-of-core
	// path already computes it closed-form; match it here instead of an O(n) accumulation). The
	// packing is that of the cubic lattice, and a voxel of sx*sy*sz is its linear image.
	double ball_r3 = 1. / 8.;	// r^3 of the ball inscribed in a unit voxel
	double sumPackedV = double(r.raw_pixels_3D.size()) * (4. / 3. * M_PI * ball_r3);
	fval_VOXEL_VOLUME = sumPackedV / 0.5236 * sx * sy * sz;		// packaging density at kissing number = 4 (cubic lattice)

	// surface area: count exposed voxel faces in the 6-neighborhood
	struct VoxelKey
	{
		StatsInt x, y, z;
		bool operator==(const VoxelKey& other) const
		{
			return x == other.x && y == other.y && z == other.z;
		}
	};
	struct VoxelKeyHash
	{
		std::size_t operator()(const VoxelKey& key) const noexcept
		{
			std::size_t h = std::hash<StatsInt>{}(key.x);
			h ^= std::hash<StatsInt>{}(key.y) + 0x9e3779b9 + (h << 6) + (h >> 2);
			h ^= std::hash<StatsInt>{}(key.z) + 0x9e3779b9 + (h << 6) + (h >> 2);
			return h;
		}
	};
	std::unordered_set<VoxelKey, VoxelKeyHash> voxels;
	voxels.reserve(r.raw_pixels_3D.size() * 2);
	for (const auto& vox : r.raw_pixels_3D)
		voxels.insert({ vox.x, vox.y, vox.z });

	static constexpr StatsInt nbr[6][3] = {
		{ 1, 0, 0 }, { -1, 0, 0 },
		{ 0, 1, 0 }, { 0, -1, 0 },
		{ 0, 0, 1 }, { 0, 0, -1 }
	};
	// a face across x is a voxel's y-z side, and so on
	const double faceArea[6] = { sy * sz, sy * sz, sx * sz, sx * sz, sx * sy, sx * sy };

	fval_AREA = 0.0;
	for (const auto& vox : r.raw_pixels_3D)
	{
		for (int k = 0; k < 6; k++)
		{
			const auto& d = nbr[k];
			if (voxels.find({ vox.x + d[0], vox.y + d[1], vox.z + d[2] }) == voxels.end())
				fval_AREA += faceArea[k];
		}
	}

	// mesh volume: the volume enclosed by the ROI's marching-cubes surface, which is the surface IBSI
	// section 3.1 defines volume (mesh) over and the one MIRP and pyradiomics integrate. The convex hull
	// is the hull of that surface: of its vertices, which the same walk collects in doubled coordinates.
	// Both are taken on the lattice, where they are exact, and a grid of sx*sy*sz voxels is its linear
	// image: the surface and its hull scale with it, so each volume is the lattice one times a voxel's.
	std::vector<std::array<double, 3>> hullPoints;
	fval_MESH_VOLUME = Nyxus::roi_mesh_volume (r.raw_pixels_3D, &hullPoints) * sx * sy * sz;
	fval_VOLUME_CONVEXHULL = convex_hull_volume (hullPoints) / 8.0 * sx * sy * sz;

	// volume-area ratio features: 3AREA (the exposed-face count) against the voxel-counting volume.
	// 3MESH_VOLUME comes from the surface mesh above and does not enter them.
	fval_AREA_2_VOLUME = fval_AREA / fval_VOXEL_VOLUME;
	fval_COMPACTNESS1 = fval_VOXEL_VOLUME / std::sqrt(M_PI * fval_AREA * fval_AREA * fval_AREA);
	fval_COMPACTNESS2 = 36. * M_PI * fval_VOXEL_VOLUME * fval_VOXEL_VOLUME / (fval_AREA * fval_AREA * fval_AREA);
	fval_SPHERICAL_DISPROPORTION = fval_AREA / std::pow(36. * M_PI * fval_VOXEL_VOLUME * fval_VOXEL_VOLUME, 1. / 3.);
	fval_SPHERICITY = std::pow(36. * M_PI * fval_VOXEL_VOLUME * fval_VOXEL_VOLUME, 1. / 3.) / fval_AREA;

	// pca features
	double K[3][3];
	Pixel3::calc_cov_matrix (K, r.raw_pixels_3D);
	scale_covariance (K, sx, sy, sz);
	double L[3];
	if (Nyxus::calc_eigvals(L, K))
	{
		// calc_eigvals returns L sorted DESCENDING, so L[0] belongs to the major axis and L[2] to the
		// least. Indexing them in that order is what makes MAJOR >= MINOR >= LEAST > 0 hold, and with
		// it ELONGATION = MINOR/MAJOR and FLATNESS = LEAST/MAJOR in [0,1]. Same definitions as IBSI
		// and MIRP; the pins are in tests/test_3d_morphology_mirp.h.
		fval_MAJOR_AXIS_LEN = 4.0 * sqrt(L[0]);
		fval_MINOR_AXIS_LEN = 4.0 * sqrt(L[1]);
		fval_LEAST_AXIS_LEN = 4.0 * sqrt(L[2]);
		fval_ELONGATION = sqrt(L[1] / L[0]);
		fval_FLATNESS = sqrt(L[2] / L[0]);
	}
	else
	{
		fval_MAJOR_AXIS_LEN = 
		fval_MINOR_AXIS_LEN = 
		fval_LEAST_AXIS_LEN = 
		fval_ELONGATION = 
		fval_FLATNESS = 0.0;
	}
}

void D3_SurfaceFeature::set_whole_box (StatsInt w, StatsInt h, StatsInt d, double sx, double sy, double sz)
{
	// 3AREA counts the box's exposed voxel faces, each the area of a voxel's side across its axis,
	// which is what the general path counts for a solid box, and the five ratios take it against the
	// voxel volume as there.
	fval_AREA = 2. * (double(w) * h * sx * sy + double(h) * d * sy * sz + double(w) * d * sx * sz);

	// The ROI is the whole box, so its marching-cubes surface is a bevelled box and the mesh volume
	// has a closed form: each of the 4*(w+h+d-3) cells along an interior edge and each of the 8
	// corner cells cuts a fixed amount off the staircase. For 3MESH_VOLUME this reproduces the
	// general path's roi_mesh_volume() exactly, down to a one-voxel box, and a voxel of sx*sy*sz
	// scales it as it scales the general path's.
	const double edgecells = double(w) + double(h) + double(d) - 3.;
	fval_MESH_VOLUME = (double(w) * h * d - edgecells / 2. - 5. / 6.) * sx * sy * sz;

	// The bevelled box is convex, so its hull is itself
	fval_VOLUME_CONVEXHULL = fval_MESH_VOLUME;

	// 3VOXEL_VOLUME reports the box volume, the volume of the union of the voxels. The general path's
	// voxel volume carries the packing constant instead.
	fval_VOXEL_VOLUME = double(w) * h * d * sx * sy * sz;

	fval_AREA_2_VOLUME = fval_AREA / fval_VOXEL_VOLUME;
	fval_COMPACTNESS1 = fval_VOXEL_VOLUME / std::sqrt(M_PI * fval_AREA * fval_AREA * fval_AREA);
	fval_COMPACTNESS2 = 36. * M_PI * fval_VOXEL_VOLUME * fval_VOXEL_VOLUME / (fval_AREA * fval_AREA * fval_AREA);
	fval_SPHERICAL_DISPROPORTION = fval_AREA / std::pow(36. * M_PI * fval_VOXEL_VOLUME * fval_VOXEL_VOLUME, 1. / 3.);
	fval_SPHERICITY = std::pow(36. * M_PI * fval_VOXEL_VOLUME * fval_VOXEL_VOLUME, 1. / 3.) / fval_AREA;

	fval_MAJOR_AXIS_LEN =
	fval_MINOR_AXIS_LEN =
	fval_LEAST_AXIS_LEN =
	fval_ELONGATION =
	fval_FLATNESS = 0;
}

void D3_SurfaceFeature::osized_add_online_pixel(size_t x, size_t y, uint32_t intensity) {}

void D3_SurfaceFeature::osized_calculate (LR& r, const Fsettings& s, ImageLoader& ldr)
{
	// Out-of-core surface: stream the disk-backed voxel cloud (raw_voxels_NT) one Z-plane at a
	// time instead of holding the whole cube (raw_pixels_3D). Per plane we tally exposed-face
	// adjacencies, and covariance is accumulated online; the mesh walk re-reads the planes two at a
	// time. Peak memory is two Z-plane occupancy bitmaps plus the hull candidates, two per lattice row
	// -- bounded by area, not volume. Values mirror the in-core calculate().
	size_t n = r.raw_voxels_NT.size();
	if (n == 0)
	{
		cleanup_instance();
		return;
	}

	const double spx = r.spacing_x, spy = r.spacing_y, spz = r.spacing_z;

	if (STNGS_SINGLEROI(s))
	{
		set_whole_box (r.aabb.get_width(), r.aabb.get_height(), r.aabb.get_z_depth(), spx, spy, spz);
		return;
	}

	// -- VOXEL_VOLUME: same cubic-lattice ball packing, a function of voxel count and voxel size only
	double sumPackedV = double(n) * (4. / 3. * M_PI * (1. / 8.));
	fval_VOXEL_VOLUME = sumPackedV / 0.5236 * spx * spy * spz;

	const int W = (int) r.aabb.get_width(),
		H = (int) r.aabb.get_height(),
		minx = (int) r.aabb.get_xmin(),
		miny = (int) r.aabb.get_ymin();

	// Surface area as exposed faces: along each axis 2*N faces less 2 per adjacent voxel pair along
	// it, each face the area of a voxel's side across that axis. x/y adjacencies are tallied within
	// a plane; z adjacencies between a plane and the previous one -- a 2-plane window.
	double adjx = 0.0, adjy = 0.0, adjz = 0.0;
	std::vector<char> prevOcc;   // occupancy bitmap of plane (z-1) over the [W x H] ROI bbox
	long long prevZ = -2;        // z index of prevOcc; -2 = none

	// Online covariance sums over all voxels
	double sx = 0, sy = 0, sz = 0, sxx = 0, syy = 0, szz = 0, sxy = 0, sxz = 0, syz = 0;

	const size_t depth = r.raw_voxels_NT.depth();
	std::vector<Pixel3> slab;
	for (size_t z = 0; z < depth; z++)
	{
		r.raw_voxels_NT.read_slab (z, slab);
		if (slab.empty())
			continue;

		// -- this plane's occupancy bitmap + online covariance sums
		std::vector<char> curOcc ((size_t) W * H, 0);
		for (const auto& v : slab)
		{
			int lx = (int) v.x - minx, ly = (int) v.y - miny;
			if (lx >= 0 && lx < W && ly >= 0 && ly < H)
				curOcc[(size_t) ly * W + lx] = 1;

			double dx = (double) v.x, dy = (double) v.y, dz = (double) v.z;
			sx += dx; sy += dy; sz += dz;
			sxx += dx * dx; syy += dy * dy; szz += dz * dz;
			sxy += dx * dy; sxz += dx * dz; syz += dy * dz;
		}

		// -- adjacencies. +x / +y within this plane (each unordered in-plane pair once)
		for (const auto& v : slab)
		{
			int lx = (int) v.x - minx, ly = (int) v.y - miny;
			if (lx + 1 < W && ly >= 0 && ly < H && curOcc[(size_t) ly * W + (lx + 1)])
				adjx += 1.0;
			if (ly + 1 < H && lx >= 0 && lx < W && curOcc[(size_t) (ly + 1) * W + lx])
				adjy += 1.0;
			// +z: pair with the previous plane at the same (x,y) (each unordered z pair once)
			if (prevZ == (long long) z - 1 && lx >= 0 && lx < W && ly >= 0 && ly < H
				&& prevOcc[(size_t) ly * W + lx])
				adjz += 1.0;
		}

		prevOcc.swap (curOcc);
		prevZ = (long long) z;
	}

	fval_AREA = (2.0 * double(n) - 2.0 * adjx) * spy * spz
		+ (2.0 * double(n) - 2.0 * adjy) * spx * spz
		+ (2.0 * double(n) - 2.0 * adjz) * spx * spy;

	// -- mesh volume and convex hull: the in-core integral and hull, fed one Z-plane at a time from the
	//    disk-backed cloud, both on the lattice and scaled by a voxel's volume as in calculate()
	const Nyxus::LatticeBounds bounds = { r.aabb.get_xmin(), r.aabb.get_xmax(), r.aabb.get_ymin(), r.aabb.get_ymax(), r.aabb.get_zmin(), r.aabb.get_zmax() };
	std::vector<std::array<double, 3>> hullPoints;
	fval_MESH_VOLUME = Nyxus::roi_mesh_volume (bounds,
		[&r](StatsInt z, std::vector<Pixel3>& voxels)
		{
			if (z >= 0)
				r.raw_voxels_NT.read_slab ((size_t) z, voxels);
		},
		&hullPoints) * spx * spy * spz;
	fval_VOLUME_CONVEXHULL = convex_hull_volume (hullPoints) / 8.0 * spx * spy * spz;

	// -- volume-area ratio features, as in calculate()
	fval_AREA_2_VOLUME = fval_AREA / fval_VOXEL_VOLUME;
	fval_COMPACTNESS1 = fval_VOXEL_VOLUME / std::sqrt(M_PI * fval_AREA * fval_AREA * fval_AREA);
	fval_COMPACTNESS2 = 36. * M_PI * fval_VOXEL_VOLUME * fval_VOXEL_VOLUME / (fval_AREA * fval_AREA * fval_AREA);
	fval_SPHERICAL_DISPROPORTION = fval_AREA / std::pow(36. * M_PI * fval_VOXEL_VOLUME * fval_VOXEL_VOLUME, 1. / 3.);
	fval_SPHERICITY = std::pow(36. * M_PI * fval_VOXEL_VOLUME * fval_VOXEL_VOLUME, 1. / 3.) / fval_AREA;

	// -- PCA axis lengths from the covariance matrix (sample covariance, matching calc_covariance:
	//    Cov(a,b) = (Sum a*b - N*mean_a*mean_b) / (N-1))
	double K[3][3];
	if (n > 1)
	{
		double mx = sx / double(n), my = sy / double(n), mz = sz / double(n);
		double nf = double(n) - 1.0;
		K[0][0] = (sxx - double(n) * mx * mx) / nf;
		K[1][1] = (syy - double(n) * my * my) / nf;
		K[2][2] = (szz - double(n) * mz * mz) / nf;
		K[0][1] = K[1][0] = (sxy - double(n) * mx * my) / nf;
		K[0][2] = K[2][0] = (sxz - double(n) * mx * mz) / nf;
		K[1][2] = K[2][1] = (syz - double(n) * my * mz) / nf;
	}
	else
		K[0][0] = K[0][1] = K[0][2] = K[1][0] = K[1][1] = K[1][2] = K[2][0] = K[2][1] = K[2][2] = 0;
	scale_covariance (K, spx, spy, spz);

	double L[3];
	if (Nyxus::calc_eigvals(L, K))
	{
		// calc_eigvals returns L sorted DESCENDING (L[0] largest): MAJOR<-L[0], MINOR<-L[1],
		// LEAST<-L[2]; ELONGATION=MINOR/MAJOR=sqrt(L[1]/L[0]), FLATNESS=LEAST/MAJOR=sqrt(L[2]/L[0]).
		// Matches MIRP/IBSI and the in-RAM calculate() path.
		fval_MAJOR_AXIS_LEN = 4.0 * sqrt(L[0]);
		fval_MINOR_AXIS_LEN = 4.0 * sqrt(L[1]);
		fval_LEAST_AXIS_LEN = 4.0 * sqrt(L[2]);
		fval_ELONGATION = sqrt(L[1] / L[0]);
		fval_FLATNESS = sqrt(L[2] / L[0]);
	}
	else
		fval_MAJOR_AXIS_LEN = fval_MINOR_AXIS_LEN = fval_LEAST_AXIS_LEN = fval_ELONGATION = fval_FLATNESS = 0.0;
}

void D3_SurfaceFeature::save_value(std::vector<std::vector<double>>& fvals)
{
	fvals[(int)Nyxus::Feature3D::AREA][0] = fval_AREA;
	fvals[(int)Nyxus::Feature3D::AREA_2_VOLUME][0] = fval_AREA_2_VOLUME;
	fvals[(int)Nyxus::Feature3D::COMPACTNESS1][0] = fval_COMPACTNESS1;
	fvals[(int)Nyxus::Feature3D::COMPACTNESS2][0] = fval_COMPACTNESS2;
	fvals[(int)Nyxus::Feature3D::MESH_VOLUME][0] = fval_MESH_VOLUME;
	fvals[(int)Nyxus::Feature3D::SPHERICAL_DISPROPORTION][0] = fval_SPHERICAL_DISPROPORTION;
	fvals[(int)Nyxus::Feature3D::SPHERICITY][0] = fval_SPHERICITY;
	fvals[(int)Nyxus::Feature3D::VOLUME_CONVEXHULL][0] = fval_VOLUME_CONVEXHULL;
	fvals[(int)Nyxus::Feature3D::VOXEL_VOLUME][0] = fval_VOXEL_VOLUME;

	fvals[(int)Nyxus::Feature3D::MAJOR_AXIS_LEN][0] = fval_MAJOR_AXIS_LEN;
	fvals[(int)Nyxus::Feature3D::MINOR_AXIS_LEN][0] = fval_MINOR_AXIS_LEN;
	fvals[(int)Nyxus::Feature3D::LEAST_AXIS_LEN][0] = fval_LEAST_AXIS_LEN;
	fvals[(int)Nyxus::Feature3D::ELONGATION][0] = fval_ELONGATION;
	fvals[(int)Nyxus::Feature3D::FLATNESS][0] = fval_FLATNESS;
}

void D3_SurfaceFeature::cleanup_instance()
{
	fval_AREA =
	fval_AREA_2_VOLUME =
	fval_COMPACTNESS1 =
	fval_COMPACTNESS2 =
	fval_MESH_VOLUME =
	fval_SPHERICAL_DISPROPORTION =
	fval_SPHERICITY =
	fval_VOLUME_CONVEXHULL =
	fval_VOXEL_VOLUME =
	fval_MAJOR_AXIS_LEN =
	fval_MINOR_AXIS_LEN =
	fval_LEAST_AXIS_LEN =
	fval_ELONGATION =
	fval_FLATNESS = 0;
}

void D3_SurfaceFeature::reduce (size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData, const Fsettings & s, const Dataset & _)
{
	for (auto i = start; i < end; i++)
	{
		int lab = (*ptrLabels)[i];
		LR& r = (*ptrLabelData)[lab];
		D3_SurfaceFeature f;
		f.calculate (r, s);
		f.save_value (r.fvals);
	}
}

/*static*/ void D3_SurfaceFeature::extract (LR& r, const Fsettings& s)
{
	D3_SurfaceFeature f;
	f.calculate (r, s);
	f.save_value (r.fvals);
}


