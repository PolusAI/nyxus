#define _USE_MATH_DEFINES	// for M_PI, etc.
#include <regex>
#include "../featureset.h"
#include "../environment.h"	
#include "../3rdparty/quickhull.hpp"
#include "../3rdparty/dsyevj3.h"
#include "3d_surface.h"

namespace Nyxus
{
	void build_contour_imp(
		// out
		std::vector <size_t>& contour,	// indices in the cloud
		// in
		const std::vector <Pixel3>& cloud,	// achtung! coordinates need fixing wrt AABB!
		const std::vector <size_t>& plane,
		int z,
		int width,
		int height,
		int minx,
		int miny,
		int verbose_level)
	{
		//==== Pad the image

		int paddingColor = 0;
		std::vector<PixIntens> paddedImage((height + 2) * (width + 2), paddingColor);
		for (auto idx : plane) // 'zplane' is indices of the z-plane to build the contour for

		{
			auto& px = cloud[idx];
			auto x = px.x - minx + 1,
				y = px.y - miny + 1;
			paddedImage[x + y * (width + 2)] = idx/*px.inten*/ + 1;	// Decorate the intensity
		}

		VERBOSLVL4(verbose_level,
			std::cout << "\n\n\n" << "-- build_contour_imp() / z=" << z << "/ Padded image--\n";
		for (int y = 0; y < height + 2; y++)
		{
			for (int x = 0; x < width + 2; x++)
			{
				size_t idx = x + y * (width + 2);
				auto inte = paddedImage[idx];
				if (inte)
					std::cout << '*';
				else
					std::cout << '.';
			}
			std::cout << "\n";
		}
		std::cout << "\n\n\n";
		);

		const int BLANK = 0;
		bool inside = false;
		int pos = 0;

		//==== Prepare the contour ("border") image
		std::vector<PixIntens> borderImage((height + 2) * (width + 2), 0);

		// Initialize the entire image to blank
		for (int y = 0; y < (height + 2); y++)
			for (int x = 0; x < (width + 2); x++)
				borderImage[x + y * (width + 2)] = BLANK;

		//==== Scan the padded image and fill the border one
		for (int y = 0; y < (height + 2); y++)
			for (int x = 0; x < (width + 2); x++)
			{
				pos = x + y * (width + 2);

				// Scan for a non-blank pixel
				if (borderImage[pos] != 0 && !inside)		// Entering an already discovered border
				{
					inside = true;
				}
				else if (paddedImage[pos] != 0 && inside)	// Already discovered border point
				{
					continue;
				}
				else if (paddedImage[pos] == BLANK && inside)	// Leaving a border
				{
					inside = false;
				}
				else if (paddedImage[pos] != 0 && !inside)	// Undiscovered border point
				{
					borderImage[pos] = paddedImage[pos];	// Non-blank

					int checkLocationNr = 1;	// The neighbor number of the location we want to check for a new border point
					int checkPosition;			// The corresponding absolute array address of checkLocationNr
					int newCheckLocationNr; 	// Variable that holds the neighborhood position we want to check if we find a new border at checkLocationNr
					int startPos = pos;			// Set start position
					int counter = 0; 			// Counter is used for the jacobi stop criterion
					int counter2 = 0; 			// Counter2 is used to determine if the point we have discovered is one single point

					// Defines the neighborhood offset position from current position and the neighborhood
					// position we want to check next if we find a new border at checkLocationNr
					int neighborhood[8][2] = {
							{-1,7},
							{-3 - width,7},
							{-width - 2,1},
							{-1 - width,1},
							{1,3},
							{3 + width,3},
							{width + 2,5},
							{1 + width,5}
					};

					// Trace around the neighborhood
					while (true)
					{
						checkPosition = pos + neighborhood[checkLocationNr - 1][0];
						newCheckLocationNr = neighborhood[checkLocationNr - 1][1];

						if (paddedImage[checkPosition] != 0) // Next border point found?
						{
							if (checkPosition == startPos)
							{
								counter++;

								// Stopping criterion (jacob)
								if (newCheckLocationNr == 1 || counter >= 3)
								{
									// Close loop
									inside = true; // Since we are starting the search at were we first started we must set inside to true
									break;
								}
							}

							checkLocationNr = newCheckLocationNr; // Update which neighborhood position we should check next
							pos = checkPosition;
							counter2 = 0; 						// Reset the counter that keeps track of how many neighbors we have visited
							borderImage[checkPosition] = paddedImage[checkPosition]; // Non-blank
						}
						else
						{
							// Rotate clockwise in the neighborhood
							checkLocationNr = 1 + (checkLocationNr % 8);
							if (counter2 > 8)
							{
								// If counter2 is above 8, we have sought around the neighborhood and
								// therefore the border is a single non-blank pixel, and we can exit
								counter2 = 0;
								break;
							}
							else
							{
								counter2++;
							}
						}
					}
				}
			}

		VERBOSLVL4(verbose_level,
			std::cout << "\n\n\n" << "-- ContourFeature / buildRegularContour / Contour image --\n";
		// header
		std::cout << "\t";	// indent
		for (int i = 0; i < width; i++)
			if (i % 10 == 0)
				std::cout << '|';
			else
				std::cout << '_';
		std::cout << "\n";
		//---
		for (int y = 0; y < height + 2; y++)
		{
			std::cout << "y=" << y << "\t";
			for (int x = 0; x < width + 2; x++)
			{
				size_t idx = x + y * (width + 2);
				auto inte = borderImage[idx];
				if (inte)
					std::cout << ' ';
				else
					std::cout << '+';
			}
			std::cout << "\n";
		}
		std::cout << "\n\n\n";
		);

		//==== Remove padding and save the contour image as a vector of non-blank pixels

		contour.clear();

		for (int y = 0; y < height + 2; y++)
			for (int x = 0; x < width + 2; x++)
			{
				size_t idx = x + y * (width + 2);
				auto ix = borderImage[idx];
				PixIntens inte = 0;
				if (ix)
					inte = cloud[ix - 1].inten;
				if (inte)
				{
					//Pixel3 p(x + minx, y + miny, z, inte); // indexed scenario: '..., z, inte-1'		// Cast pixel position from relative to absolute and undecorate its intensity
					contour.push_back(ix - 1); // push_back(p);
				}
			}
	}

}	//- namespace Nyxus

bool D3_SurfaceFeature::required (const FeatureSet & fs)
{
	return fs.anyEnabled (D3_SurfaceFeature::featureset);
}

D3_SurfaceFeature::D3_SurfaceFeature() : FeatureMethod("D3_SurfaceFeature")
{
	provide_features (D3_SurfaceFeature::featureset);
}

void dump_skinny_contour_3D (
	const std::vector <size_t>& K,	// contour	indices of contour voxels in the cloud
	const std::vector <Pixel3>& C,	// cloud	achtung! coordinates need fixing wrt AABB!
	int width,
	int height,
	int minx,
	int miny)
{
	if (K.size() == 0)
	{
		std::cout << "ERROR: blank contour\n";
		return;
	}

	// check if contour is on the same z-plane
	auto z0 = C[K[0]].z;
	for (auto ik : K)
		if (C[ik].z != z0)
		{
			std::cout << "ERROR: contour spans multiple z-planes: z_0=" << z0 << ", z_k=" << C[ik].z << "\n";
			return;
		}

	// sparse --> matrix
	PixIntens val0 = 0;
	std::vector<PixIntens> M (height*width, val0);
	for (auto ik : K)
	{
		auto& px = C [ik];
		auto x = px.x - minx,
			y = px.y - miny;
		M [x + y * width] = px.inten;
	}

	// display it
	std::cout << "\n\n\n" << "-- Skinny contour --\n";
	// header
	std::cout << "\t";	// indent
	for (int i = 0; i < width; i++)
		if (i % 10 == 0)
			std::cout << '|';
		else
			std::cout << '_';
	std::cout << "\n";
	//---
	for (int y = 0; y < height; y++)
	{
		std::cout << "y=" << y << "\t";
		for (int x = 0; x < width; x++)
		{
			size_t idx = x + y * width;
			auto inte = M[idx];
			if (inte)
				std::cout << ' ';
			else
				std::cout << '+';
		}
		std::cout << "\n";
	}
	std::cout << "\n\n\n";
}

void D3_SurfaceFeature::build_surface (LR & r)
{
	// read the point cloud of contours
	constexpr std::size_t dim = 3;
	using Points = std::vector<std::array<float, dim>>;
	Points P;
	for (auto& plane : r.contours_3D)
	{
		for (auto ip : plane)
		{
			auto v = r.raw_pixels_3D[ip];
			P.push_back (std::array<float, 3>({ (float)v.x, (float)v.y, (float)v.z }));
		}
	}

	build_hull (P);
}

// Build the convex-hull facet complex from a contour point cloud. Shared by the in-core
// build_surface and the out-of-core osized_calculate (which collects the same contour points
// from the disk-backed voxel cloud), so both produce an identical hull.
void D3_SurfaceFeature::build_hull (const std::vector<std::array<float, 3>>& P)
{
	constexpr std::size_t dim = 3;
	using Points = std::vector<std::array<float, dim>>;

	const auto eps = 1e-10f;
	quick_hull<typename Points::const_iterator> qh{ dim, eps };
	qh.add_points(std::cbegin(P), std::cend(P));
	auto initial_simplex = qh.get_affine_basis();
	if (initial_simplex.size() < dim + 1) 
	{
#ifdef WITH_PYTHON_H
		throw std::runtime_error ("degenerate convex shell input");
#endif
		std::cerr << "degenerate convex shell input \n";
		return;
	}
	qh.create_initial_simplex(std::cbegin(initial_simplex), std::prev(std::cend(initial_simplex)));
	qh.create_convex_hull();

	// gather the complex
	hull_complex.clear();
	for (auto f : qh.facets_)
	{
		const auto & V = f.vertices_;
		auto ax = (*V[0])[0], ay = (*V[0])[1], az = (*V[0])[2];
		auto bx = (*V[1])[0], by = (*V[1])[1], bz = (*V[1])[2];
		auto cx = (*V[2])[0], cy = (*V[2])[1], cz = (*V[2])[2];
		float a[3] = { ax, ay, az }, b[3] = { bx, by, bz }, c[3] = {cx, cy, cz};
		Simplex3 s(a, b, c);
		hull_complex.push_back(s);
	}
}

void D3_SurfaceFeature::calculate (LR& r, const Fsettings& s)
{
	// is shape data non-informative ?
	if (r.raw_pixels_3D.size() == 0)
	{
		cleanup_instance();
		return;
	}

	if (STNGS_SINGLEROI(s))	// former Nyxus::theEnvironment.singleROI
	{
		auto w = r.aabb.get_width(),
			h = r.aabb.get_height(),
			d = r.aabb.get_z_depth();

		fval_AREA = 2 * (w*h + h*d + w*d);
		fval_VOLUME_CONVEXHULL = fval_VOXEL_VOLUME = fval_MESH_VOLUME = w * h * d;
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

		return;
	}

	// volume

	// (fast approximation based on cubic lattice packaging of balls)
	double ball_r3 = 1. / 8.,	// r^3, after the anisotropy correction, lattice is expected to be cubic
		sumPackedV = 0.0;
	for (const auto& vox : r.raw_pixels_3D)
		sumPackedV += 4. / 3. * M_PI * ball_r3;
	fval_VOXEL_VOLUME = sumPackedV / 0.5236;		// packaging density at kissing number = 4 (cubic lattice)

	// surface

	// -- order z-planes' indices
	std::vector<int> zindices;
	for (auto& plane : r.zplanes)
		zindices.push_back (plane.first);
	std::sort (zindices.begin(), zindices.end());

	// -- calculate contours
	r.contours_3D.clear();

	for (auto zi : zindices)
	{
		auto& planeVoxs = r.zplanes[zi];	// deterministically indexed plane

		// skinny contour
		std::vector<size_t> K;	// indices in the cloud
		Nyxus::build_contour_imp (
			K, 
			r.raw_pixels_3D, 
			planeVoxs,
			zi,
			r.aabb.get_width(), 
			r.aabb.get_height(), 
			r.aabb.get_xmin(), 
			r.aabb.get_ymin(),
			STNGS_VERBOSLVL(s));

		// store it
		r.contours_3D.push_back (K);
	}

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

	fval_AREA = 0.0;
	for (const auto& vox : r.raw_pixels_3D)
	{
		for (const auto& d : nbr)
		{
			if (voxels.find({ vox.x + d[0], vox.y + d[1], vox.z + d[2] }) == voxels.end())
				fval_AREA += 1.0;
		}
	}

	// -- build the hull complex
	build_surface (r);

	// convex hull volume

	// -- centroid

	double cx = 0, cy = 0, cz = 0;
	size_t hullCloudLen = 0;
	for (auto& plane : r.contours_3D)
	{
		for (auto ip : plane)
		{
			auto v = r.raw_pixels_3D[ip];
			cx += v.x;
			cy += v.y;
			cz += v.z;
			hullCloudLen++;
		}
	}
	cx /= double(hullCloudLen);
	cy /= double(hullCloudLen);
	cz /= double(hullCloudLen);

	// -- volume by all the conv hull faces

	//
	//						| x1  y1  z1  1 |
	//						| x2  y2  z2  1 |
	// V = 1 / 3!  *	| x3  y3  z3  1 |
	//						| x4  y4  z4  1 |
	//

	fval_VOLUME_CONVEXHULL = 0;
	for (const auto& s : hull_complex)
	{
		// layout: x:[0], y:[1], z:[2]
		double AB[3] = { s.b[0] - s.a[0], s.b[1] - s.a[1], s.b[2] - s.a[2] },
			AC[3] = { s.c[0] - s.a[0], s.c[1] - s.a[1], s.c[2] - s.a[2] };

		double d = Nyxus::det4(
			s.a[0], s.a[1], s.a[2], 1,
			s.b[0], s.b[1], s.b[2], 1,
			s.c[0], s.c[1], s.c[2], 1,
			cx, cy, cz, 1);

		fval_VOLUME_CONVEXHULL += d / 6;
	}

	// volume-area ratio features
	fval_MESH_VOLUME = fval_VOLUME_CONVEXHULL;
	fval_AREA_2_VOLUME = fval_AREA / fval_VOXEL_VOLUME;
	fval_COMPACTNESS1 = fval_VOXEL_VOLUME / std::sqrt(M_PI * fval_AREA * fval_AREA * fval_AREA);
	fval_COMPACTNESS2 = 36. * M_PI * fval_VOXEL_VOLUME * fval_VOXEL_VOLUME / (fval_AREA * fval_AREA * fval_AREA);
	fval_SPHERICAL_DISPROPORTION = fval_AREA / std::pow(36. * M_PI * fval_VOXEL_VOLUME * fval_VOXEL_VOLUME, 1. / 3.);
	fval_SPHERICITY = std::pow(36. * M_PI * fval_VOXEL_VOLUME * fval_VOXEL_VOLUME, 1. / 3.) / fval_AREA;

	// pca features
	double K[3][3];
	Pixel3::calc_cov_matrix (K, r.raw_pixels_3D);
	double L[3];
	if (Nyxus::calc_eigvals(L, K))
	{
		// FIX: calc_eigvals returns L sorted DESCENDING (L[0] largest). The axis lengths were indexed
		// wrong (MAJOR<-L[1], MINOR<-L[2], LEAST<-L[0]), producing LEAST>MAJOR and FLATNESS>1 (both
		// structurally impossible). Correct mapping: MAJOR<-L[0] (largest), MINOR<-L[1], LEAST<-L[2];
		// ELONGATION=MINOR/MAJOR=sqrt(L[1]/L[0]), FLATNESS=LEAST/MAJOR=sqrt(L[2]/L[0]). Matches MIRP/IBSI.
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

void D3_SurfaceFeature::osized_add_online_pixel(size_t x, size_t y, uint32_t intensity) {}

void D3_SurfaceFeature::osized_calculate (LR& r, const Fsettings& s, ImageLoader& ldr)
{
	// Out-of-core surface: stream the disk-backed voxel cloud (raw_voxels_NT) one Z-plane at a
	// time instead of holding the whole cube (raw_pixels_3D). Per plane we extract its contour and
	// tally exposed-face adjacencies; covariance is accumulated online. Peak memory is the
	// collected boundary points (~surface area) plus two Z-plane occupancy bitmaps -- bounded by
	// area, not volume. Values mirror the in-core calculate().
	size_t n = r.raw_voxels_NT.size();
	if (n == 0)
	{
		cleanup_instance();
		return;
	}

	if (STNGS_SINGLEROI(s))
	{
		auto w = r.aabb.get_width(),
			h = r.aabb.get_height(),
			d = r.aabb.get_z_depth();

		fval_AREA = 2 * (w*h + h*d + w*d);
		fval_VOLUME_CONVEXHULL = fval_VOXEL_VOLUME = fval_MESH_VOLUME = w * h * d;
		fval_AREA_2_VOLUME = fval_AREA / fval_VOXEL_VOLUME;
		fval_COMPACTNESS1 = fval_VOXEL_VOLUME / std::sqrt(M_PI * fval_AREA * fval_AREA * fval_AREA);
		fval_COMPACTNESS2 = 36. * M_PI * fval_VOXEL_VOLUME * fval_VOXEL_VOLUME / (fval_AREA * fval_AREA * fval_AREA);
		fval_SPHERICAL_DISPROPORTION = fval_AREA / std::pow(36. * M_PI * fval_VOXEL_VOLUME * fval_VOXEL_VOLUME, 1. / 3.);
		fval_SPHERICITY = std::pow(36. * M_PI * fval_VOXEL_VOLUME * fval_VOXEL_VOLUME, 1. / 3.) / fval_AREA;
		fval_MAJOR_AXIS_LEN = fval_MINOR_AXIS_LEN = fval_LEAST_AXIS_LEN = fval_ELONGATION = fval_FLATNESS = 0;
		return;
	}

	// -- VOXEL_VOLUME: same cubic-lattice ball packing, a function of voxel count only
	double sumPackedV = double(n) * (4. / 3. * M_PI * (1. / 8.));
	fval_VOXEL_VOLUME = sumPackedV / 0.5236;

	const int W = (int) r.aabb.get_width(),
		H = (int) r.aabb.get_height(),
		minx = (int) r.aabb.get_xmin(),
		miny = (int) r.aabb.get_ymin();
	const int verb = STNGS_VERBOSLVL(s);

	// Collected boundary points for the convex hull (bounded by surface area), plus the running
	// centroid of those points (matches the in-core centroid over contour voxels).
	std::vector<std::array<float, 3>> P;
	double cx = 0, cy = 0, cz = 0;
	size_t hullCloudLen = 0;

	// Surface area as exposed faces = 6*N - 2*(adjacent voxel pairs). x/y adjacencies are tallied
	// within a plane; z adjacencies between a plane and the previous one -- a 2-plane window.
	double adjacencies = 0.0;
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

		// -- contour of this plane (build over a plane-local cloud; the returned indices point
		//    into 'slab', and the resulting boundary POINTS/order match the in-core path)
		std::vector<size_t> planeIdx (slab.size());
		for (size_t i = 0; i < slab.size(); i++)
			planeIdx[i] = i;
		std::vector<size_t> K;
		Nyxus::build_contour_imp (K, slab, planeIdx, (int) z, W, H, minx, miny, verb);
		for (auto ik : K)
		{
			const auto& v = slab[ik];
			P.push_back (std::array<float, 3>({ (float) v.x, (float) v.y, (float) v.z }));
			cx += v.x; cy += v.y; cz += v.z;
			hullCloudLen++;
		}

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
				adjacencies += 1.0;
			if (ly + 1 < H && lx >= 0 && lx < W && curOcc[(size_t) (ly + 1) * W + lx])
				adjacencies += 1.0;
			// +z: pair with the previous plane at the same (x,y) (each unordered z pair once)
			if (prevZ == (long long) z - 1 && lx >= 0 && lx < W && ly >= 0 && ly < H
				&& prevOcc[(size_t) ly * W + lx])
				adjacencies += 1.0;
		}

		prevOcc.swap (curOcc);
		prevZ = (long long) z;
	}

	fval_AREA = 6.0 * double(n) - 2.0 * adjacencies;

	// -- convex hull from the collected boundary points
	build_hull (P);

	// -- convex hull volume via signed tetrahedra from the contour centroid
	if (hullCloudLen)
	{
		cx /= double(hullCloudLen);
		cy /= double(hullCloudLen);
		cz /= double(hullCloudLen);
	}
	fval_VOLUME_CONVEXHULL = 0;
	for (const auto& sx3 : hull_complex)
	{
		double d = Nyxus::det4(
			sx3.a[0], sx3.a[1], sx3.a[2], 1,
			sx3.b[0], sx3.b[1], sx3.b[2], 1,
			sx3.c[0], sx3.c[1], sx3.c[2], 1,
			cx, cy, cz, 1);
		fval_VOLUME_CONVEXHULL += d / 6;
	}

	// -- volume-area ratio features
	fval_MESH_VOLUME = fval_VOLUME_CONVEXHULL;
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

	double L[3];
	if (Nyxus::calc_eigvals(L, K))
	{
		fval_MAJOR_AXIS_LEN = 4.0 * sqrt(L[1]);
		fval_MINOR_AXIS_LEN = 4.0 * sqrt(L[2]);
		fval_LEAST_AXIS_LEN = 4.0 * sqrt(L[0]);
		fval_ELONGATION = sqrt(L[2] / L[1]);
		fval_FLATNESS = sqrt(L[0] / L[1]);
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


