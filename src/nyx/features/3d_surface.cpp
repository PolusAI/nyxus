#define _USE_MATH_DEFINES	// for M_PI, etc.
#include <regex>
#include "../featureset.h"
#include "../environment.h"	
#include "../parallel.h"
#include "../3rdparty/quickhull.hpp"
#include "../3rdparty/dsyevj3.h"
#include "3d_surface.h"

bool D3_SurfaceFeature::required (const FeatureSet & fs)
{
	return fs.anyEnabled (D3_SurfaceFeature::featureset);
}

D3_SurfaceFeature::D3_SurfaceFeature() : FeatureMethod("D3_SurfaceFeature")
{
	provide_features (D3_SurfaceFeature::featureset);
}

void build_contour_imp (
	// out
	std::vector <size_t>& contour,	// indices in the cloud
	// in
	const std::vector <Pixel3>& cloud,	// achtung! coordinates need fixing wrt AABB!
	const std::vector <size_t>& plane,
	int z,
	int width,
	int height,
	int minx,
	int miny)
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

	VERBOSLVL4(
		std::cout << "\n\n\n" << "-- ContourFeature / buildRegularContour / Padded image --\n";
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

	VERBOSLVL4(
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
				contour.push_back(ix-1); // push_back(p);
			}
		}
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

	const auto eps = 1e-10f;
	quick_hull<typename Points::const_iterator> qh{ dim, eps };
	qh.add_points(std::cbegin(P), std::cend(P));
	auto initial_simplex = qh.get_affine_basis();
	if (initial_simplex.size() < dim + 1) 
	{
		VERBOSLVL1 (std::cerr << "degenerate convex shell input \n");
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
		auto bx = (*V[1])[0], by = (*V[1])[1], bz = (*V[2])[2];
		auto cx = (*V[2])[0], cy = (*V[2])[1], cz = (*V[2])[2];
		float a[3] = { ax, ay, az }, b[3] = { bx, by, bz }, c[3] = {cx, cy, cz};
		Simplex3 s(a, b, c);
		hull_complex.push_back(s);
	}
}

void D3_SurfaceFeature::calculate (LR& r)
{
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
		build_contour_imp (
			K, 
			r.raw_pixels_3D, 
			planeVoxs,
			zi,
			r.aabb.get_width(), 
			r.aabb.get_height(), 
			r.aabb.get_xmin(), 
			r.aabb.get_ymin());

		// store it
		r.contours_3D.push_back (K);
	}

	// -- build the hull complex
	build_surface (r);

	// -- features
	fval_AREA = 0;
	for (const auto& s : hull_complex)
	{
		// layout: x:[0], y:[1], z:[2]
		double AB[3] = { s.b[0] - s.a[0], s.b[1] - s.a[1], s.b[2] - s.a[2] },
			AC[3] = { s.c[0] - s.a[0], s.c[1] - s.a[1], s.c[2] - s.a[2] };

		// AB x AC = 
		// 
		//	|	i		j		k		|
		//	|	ABx	ABy	ABz	| = 
		// |	ACx	ACy	ACz	|
		// 
		//	= i * (ABy*ACz - ABz*ACy) -j * (ABx*ACz - ABz*ACx) + k * (ABx*ACy - ABy*ACx)
		double i = AB[1] * AC[2] - AB[2] * AC[1], j = -(AB[0] * AC[2] - AB[2] * AC[0]), k = AB[0] * AC[1] - AB[1] * AC[0];
		double mag = i * i + j * j + k * k;

		fval_AREA += std::sqrt(mag) / 2.0;
	}

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
		fval_MAJOR_AXIS_LEN = 4.0 * sqrt(L[1]);
		fval_MINOR_AXIS_LEN = 4.0 * sqrt(L[2]);
		fval_LEAST_AXIS_LEN = 4.0 * sqrt(L[0]);
		fval_ELONGATION = sqrt(L[2] / L[1]);
		fval_FLATNESS = sqrt(L[0] / L[1]);
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

void D3_SurfaceFeature::osized_calculate(LR& r, ImageLoader& imloader) {}

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
	fval_AREA  = 
	fval_AREA_2_VOLUME = 
	fval_COMPACTNESS1 = 
	fval_COMPACTNESS2 = 
	fval_MESH_VOLUME = 
	fval_SPHERICAL_DISPROPORTION = 
	fval_SPHERICITY = 
	fval_VOLUME_CONVEXHULL = 
	fval_VOXEL_VOLUME = 
	fval_MAJOR_AXIS_LEN,
	fval_MINOR_AXIS_LEN,
	fval_LEAST_AXIS_LEN,
	fval_ELONGATION,
	fval_FLATNESS = 0;
}

void D3_SurfaceFeature::parallel_process (std::vector<int>& roi_labels, std::unordered_map <int, LR>& roiData, int n_threads)
{
	size_t jobSize = roi_labels.size(),
		workPerThread = jobSize / n_threads;

	Nyxus::runParallel (D3_SurfaceFeature::parallel_process_1_batch, n_threads, workPerThread, jobSize, &roi_labels, &roiData);
}

void D3_SurfaceFeature::parallel_process_1_batch (size_t firstitem, size_t lastitem, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData)
{
	// Calculate the feature for each batch ROI item 
	for (auto i = firstitem; i < lastitem; i++)
	{
		// Get ahold of ROI's label and cache
		int roiLabel = (*ptrLabels)[i];
		LR& r = (*ptrLabelData)[roiLabel];

		// Skip the ROI if its data is invalid to prevent nans and infs in the output
		if (r.has_bad_data())
			continue;

		// Calculate the feature and save it in ROI's csv-friendly buffer 'fvals'
		D3_SurfaceFeature f;
		f.calculate(r);
		f.save_value(r.fvals);
	}
}

void D3_SurfaceFeature::reduce(size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData)
{
	for (auto i = start; i < end; i++)
	{
		int lab = (*ptrLabels)[i];
		LR& r = (*ptrLabelData)[lab];
		D3_SurfaceFeature f;
		f.calculate(r);
		f.save_value(r.fvals);
	}
}

/*static*/ void D3_SurfaceFeature::extract (LR& r)
{
	D3_SurfaceFeature f;
	f.calculate(r);
	f.save_value(r.fvals);
}


