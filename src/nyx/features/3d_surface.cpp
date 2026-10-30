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

	// mesh volume and area: the volume enclosed by the ROI's marching-cubes surface and the area of
	// that surface, which is the surface IBSI section 3.1 defines volume (mesh) and surface area (mesh)
	// over. MIRP and pyradiomics integrate the same kind of surface; theirs can differ from this one
	// where a cube face is ambiguous. The convex hull is the hull of that surface: of its vertices,
	// which the same walk collects in doubled coordinates. The volumes are taken on the lattice, where
	// they are exact, and a grid of sx*sy*sz voxels is its linear image: the surface and its hull scale
	// with it, so each volume is the lattice one times a voxel's. The area has no such factor, so the
	// walk takes each facet's area on the grid itself.
	std::vector<std::array<double, 3>> hullPoints;
	const Nyxus::MeshMeasures mesh = Nyxus::roi_mesh_measures (r.raw_pixels_3D, sx, sy, sz, &hullPoints);
	fval_MESH_VOLUME = mesh.volume;
	fval_AREA = mesh.area;
	fval_VOLUME_CONVEXHULL = convex_hull_volume (hullPoints) / 8.0 * sx * sy * sz;

	set_area_ratios();

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

void D3_SurfaceFeature::set_area_ratios()
{
	// IBSI's volume-area ratios take the mesh area against the volume enclosed by the same mesh.
	// sphere_area is the area of the sphere of the same volume.
	const double area3 = fval_AREA * fval_AREA * fval_AREA,
		sphere_area = std::pow(36. * M_PI * fval_MESH_VOLUME * fval_MESH_VOLUME, 1. / 3.);
	fval_AREA_2_VOLUME = fval_AREA / fval_MESH_VOLUME;
	fval_COMPACTNESS1 = fval_MESH_VOLUME / std::sqrt(M_PI * area3);
	fval_COMPACTNESS2 = 36. * M_PI * fval_MESH_VOLUME * fval_MESH_VOLUME / area3;
	fval_SPHERICAL_DISPROPORTION = fval_AREA / sphere_area;
	fval_SPHERICITY = sphere_area / fval_AREA;
}

void D3_SurfaceFeature::set_whole_box (StatsInt w, StatsInt h, StatsInt d, double sx, double sy, double sz)
{
	// The ROI is the whole box, so its marching-cubes surface is a bevelled box, with a closed form
	// for both mesh measures. The cells along an interior edge of the box and the 8 corner cells cut
	// the staircase off: an edge along x is a strip of width sqrt(sy^2 + sz^2)/2 and length (w-1)*sx,
	// a corner a triangle of area sqrt(sx^2 sy^2 + sy^2 sz^2 + sx^2 sz^2)/8, and the flat faces are what
	// remains, (h-1)*(d-1) cells of sy*sz across x and so on. On a cubic grid this is
	// 2(wh+hd+wd) - 2(w+h+d-3)(2-sqrt 2) - (6-sqrt 3). Both hold down to a one-voxel box. The volume
	// reproduces the general path's roi_mesh_measures() exactly, and a voxel of sx*sy*sz scales it as
	// it scales the general path's; the area is the same sum regrouped, so it agrees with the general
	// path's facet-by-facet sum to rounding, not bit for bit.
	const double wm = double(w) - 1., hm = double(h) - 1., dm = double(d) - 1.;
	fval_AREA = 2. * (hm * dm * sy * sz + wm * dm * sx * sz + wm * hm * sx * sy)
		+ 2. * (wm * sx * std::sqrt(sy * sy + sz * sz) + hm * sy * std::sqrt(sx * sx + sz * sz) + dm * sz * std::sqrt(sx * sx + sy * sy))
		+ std::sqrt(sx * sx * sy * sy + sy * sy * sz * sz + sx * sx * sz * sz);

	const double edgecells = double(w) + double(h) + double(d) - 3.;
	fval_MESH_VOLUME = (double(w) * h * d - edgecells / 2. - 5. / 6.) * sx * sy * sz;

	// The bevelled box is convex, so its hull is itself
	fval_VOLUME_CONVEXHULL = fval_MESH_VOLUME;

	// 3VOXEL_VOLUME reports the box volume, the volume of the union of the voxels. The general path's
	// voxel volume carries the packing constant instead.
	fval_VOXEL_VOLUME = double(w) * h * d * sx * sy * sz;

	set_area_ratios();

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
	// time instead of holding the whole cube (raw_pixels_3D). Covariance is accumulated online over
	// the planes; the mesh walk re-reads them two at a time. Peak memory is two Z-planes plus the hull
	// candidates, two per lattice row -- bounded by area, not volume. Values mirror the in-core
	// calculate().
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

	// Online covariance sums over all voxels
	double sx = 0, sy = 0, sz = 0, sxx = 0, syy = 0, szz = 0, sxy = 0, sxz = 0, syz = 0;

	const size_t depth = r.raw_voxels_NT.depth();
	std::vector<Pixel3> slab;
	for (size_t z = 0; z < depth; z++)
	{
		r.raw_voxels_NT.read_slab (z, slab);
		if (slab.empty())
			continue;
		for (const auto& v : slab)
			r.aabb.require_contains (v.x, v.y, v.z);

		// -- online covariance sums
		for (const auto& v : slab)
		{
			double dx = (double) v.x, dy = (double) v.y, dz = (double) v.z;
			sx += dx; sy += dy; sz += dz;
			sxx += dx * dx; syy += dy * dy; szz += dz * dz;
			sxy += dx * dy; sxz += dx * dz; syz += dy * dz;
		}
	}

	// -- mesh volume, mesh area and convex hull: the in-core integrals and hull, fed one Z-plane at a
	//    time from the disk-backed cloud, the volumes on the lattice and scaled by a voxel's volume and
	//    the area on the grid, as in calculate()
	const Nyxus::LatticeBounds bounds = { r.aabb.get_xmin(), r.aabb.get_xmax(), r.aabb.get_ymin(), r.aabb.get_ymax(), r.aabb.get_zmin(), r.aabb.get_zmax() };
	std::vector<std::array<double, 3>> hullPoints;
	const Nyxus::MeshMeasures mesh = Nyxus::roi_mesh_measures (bounds,
		[&r](StatsInt z, std::vector<Pixel3>& voxels)
		{
			if (z >= 0)
				r.raw_voxels_NT.read_slab ((size_t) z, voxels);
		},
		spx, spy, spz, &hullPoints);
	fval_MESH_VOLUME = mesh.volume;
	fval_AREA = mesh.area;
	fval_VOLUME_CONVEXHULL = convex_hull_volume (hullPoints) / 8.0 * spx * spy * spz;

	set_area_ratios();

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


