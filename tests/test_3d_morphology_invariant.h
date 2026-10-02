#pragma once

#include <cmath>
#include <string>
#include <vector>
#include "../src/nyx/features/3d_surface.h"   // D3_SurfaceFeature
#include "../src/nyx/features/pixel.h"        // Pixel3
#include "../src/nyx/roi_cache.h"             // LR
#include "test_3d_morphology_common.h"        // gtest, agrees_gt

// D3_SurfaceFeature's two paths must report the same surface (SPEC 2 invariant tier -- a required
// relation between two implementations, not an oracle claim):
//
//   in-RAM        calculate(), over the voxel list
//   out-of-core   osized_calculate(), streaming the disk-backed voxel cloud one Z-plane at a time
//
// An ROI goes out-of-core because of its size alone, so which path runs must not be observable in
// any value. 3MESH_VOLUME is asserted with exact equality: both paths run the one marching-cubes
// walk over the same planes and sum the same triangles in the same order, so any difference at all
// means they are not the same integral. 3AREA is a count of exposed faces on both sides, so it is
// exact too. 3VOLUME_CONVEXHULL is the exact hull volume of the same contour voxels on both sides,
// but the paths gather those voxels in different orders, and the facet sum can differ in its last
// bits; it is held at rel=1e-12.
//
// The fixture is an L-shaped slab with a bore through its corner and a notch cut into one arm:
// non-convex, so mesh and hull volume differ and the comparison cannot pass by one path reporting
// the other's hull.

namespace
{
	std::vector<Pixel3> make_3d_morphology_bored_ell()
	{
		const int C = 3;	// offset, kept off the lattice origin
		std::vector<Pixel3> cloud;
		for (int z = 0; z < 6; z++)	// ascending z: the order the out-of-core cloud is written in
			for (int y = 0; y < 20; y++)
				for (int x = 0; x < 20; x++)
				{
					const bool inEll = x < 6 || y < 6,
						inBore = x >= 2 && x <= 3 && y >= 2 && y <= 3,
						inNotch = x >= 14 && y < 2 && z >= 3;
					if (inEll && ! inBore && ! inNotch)
						cloud.push_back (Pixel3 (x + C, y + C, z + C, 1000));
				}
		return cloud;
	}
	struct Surface3DValues
	{
		double area, mesh_volume, hull_volume;
	};

	Surface3DValues read_3d_surface_values (const LR& r)
	{
		return { r.fvals[(int)Nyxus::Feature3D::AREA][0],
			r.fvals[(int)Nyxus::Feature3D::MESH_VOLUME][0],
			r.fvals[(int)Nyxus::Feature3D::VOLUME_CONVEXHULL][0] };
	}

	// Featurizes one cloud both ways and returns the two sets of values
	void calculate_3d_surface_both_paths (const std::vector<Pixel3>& cloud, bool single_roi,
		Surface3DValues& in_ram, Surface3DValues& ooc)
	{
		Fsettings s;
		s.resize ((int)NyxSetting::__COUNT__);
		s[(int)NyxSetting::SINGLEROI].bval = single_roi;
		s[(int)NyxSetting::VERBOSLVL].ival = 0;

		LR r (1);
		r.aabb.init_x (cloud[0].x);
		r.aabb.init_y (cloud[0].y);
		r.aabb.init_z (cloud[0].z);
		for (size_t i = 0; i < cloud.size(); i++)
		{
			r.aabb.update_x (cloud[i].x);
			r.aabb.update_y (cloud[i].y);
			r.aabb.update_z (cloud[i].z);
		}

		{
			r.raw_pixels_3D = cloud;
			for (size_t i = 0; i < cloud.size(); i++)
				r.zplanes[(int)cloud[i].z].push_back (i);

			D3_SurfaceFeature f;
			r.initialize_fvals();
			ASSERT_NO_THROW (f.calculate (r, s));
			f.save_value (r.fvals);
			in_ram = read_3d_surface_values (r);

			r.raw_pixels_3D.clear();
			r.zplanes.clear();
			r.contours_3D.clear();
		}

		{
			r.raw_voxels_NT.init (r.label, single_roi ? "surface3d_paths_agree_singleroi" : "surface3d_paths_agree");
			StatsInt z_open = -1;
			for (const auto& v : cloud)
			{
				if (v.z != z_open)
				{
					r.raw_voxels_NT.begin_slab ((size_t) v.z);
					z_open = v.z;
				}
				r.raw_voxels_NT.add_voxel (v);
			}

			D3_SurfaceFeature f;
			ImageLoader dummy;	// osized_calculate reads the cloud, not the loader
			r.initialize_fvals();
			ASSERT_NO_THROW (f.osized_calculate (r, s, dummy));
			f.save_value (r.fvals);
			ooc = read_3d_surface_values (r);

			r.raw_voxels_NT.clear();
		}
	}

	void assert_3d_surface_paths_agree (bool single_roi, Surface3DValues& in_ram)
	{
		const auto cloud = make_3d_morphology_bored_ell();

		Surface3DValues ooc {};
		calculate_3d_surface_both_paths (cloud, single_roi, in_ram, ooc);
		if (::testing::Test::HasFatalFailure())
			return;

		ASSERT_EQ (in_ram.mesh_volume, ooc.mesh_volume) << "3MESH_VOLUME differs across the path boundary";
		ASSERT_EQ (in_ram.area, ooc.area) << "3AREA differs across the path boundary";
		ASSERT_TRUE (agrees_gt (ooc.hull_volume, in_ram.hull_volume, 1e12))
			<< "3VOLUME_CONVEXHULL differs across the path boundary: in-RAM " << in_ram.hull_volume
			<< ", out-of-core " << ooc.hull_volume;
	}
}

// The general path: the bored L is meshed and hulled voxel by voxel. The fixture must keep its
// mesh and hull volumes apart, or an out-of-core path that reported the hull as the mesh volume
// would pass the comparison.
void test_3d_morphology_surface_paths_agree_invariant()
{
	Surface3DValues in_ram {};
	assert_3d_surface_paths_agree (false, in_ram);
	if (::testing::Test::HasFatalFailure())
		return;
	ASSERT_GT (in_ram.hull_volume, 1.05 * in_ram.mesh_volume) << "the fixture no longer separates mesh and hull volume";
}

// The whole-volume (SINGLEROI) branch: both paths take the ROI to be its bounding box and report the
// same closed forms.
void test_3d_morphology_surface_paths_agree_singleroi_invariant()
{
	Surface3DValues in_ram {};
	assert_3d_surface_paths_agree (true, in_ram);
}