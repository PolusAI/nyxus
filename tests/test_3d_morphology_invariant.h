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
// exact too. 3VOLUME_CONVEXHULL hulls the vertices that same walk produces, and its volume is a sum
// of exact integer terms, so it is held to exact equality as well.
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
		ASSERT_EQ (in_ram.hull_volume, ooc.hull_volume) << "3VOLUME_CONVEXHULL differs across the path boundary";
	}

	// In-RAM values of one cloud
	Surface3DValues calculate_3d_surface_in_ram (const std::vector<Pixel3>& cloud)
	{
		Fsettings s;
		s.resize ((int)NyxSetting::__COUNT__);
		s[(int)NyxSetting::SINGLEROI].bval = false;
		s[(int)NyxSetting::VERBOSLVL].ival = 0;

		LR r (1);
		r.raw_pixels_3D = cloud;
		r.aabb.init_x (cloud[0].x);
		r.aabb.init_y (cloud[0].y);
		r.aabb.init_z (cloud[0].z);
		for (size_t i = 0; i < cloud.size(); i++)
		{
			r.aabb.update_x (cloud[i].x);
			r.aabb.update_y (cloud[i].y);
			r.aabb.update_z (cloud[i].z);
			r.zplanes[(int)cloud[i].z].push_back (i);
		}

		D3_SurfaceFeature f;
		r.initialize_fvals();
		f.calculate (r, s);
		f.save_value (r.fvals);
		return read_3d_surface_values (r);
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

// A ragged ROI keeps its values when it moves. Each lattice voxel of a 10 x 9 x 6 box is kept or
// dropped by a fixed pseudo-random sequence, so the surface has ambiguous faces and a hull with
// facets at many orientations; the ROI is then placed at the origin's neighbourhood and thousands of
// voxels away. Both volumes are summed relative to a point on the body, from integer (doubled)
// coordinates, so every term is exact and the three values must agree bit for bit. A sum taken on
// absolute coordinates against a non-integer centre loses that: its rounding grows with the
// distance from the origin.
void test_3d_morphology_translated_roi_invariant()
{
	std::vector<Pixel3> blob;
	unsigned int seed = 12345u;
	for (int z = 0; z < 6; z++)
		for (int y = 0; y < 9; y++)
			for (int x = 0; x < 10; x++)
			{
				seed = seed * 1103515245u + 12345u;
				if ((seed >> 16) % 10 < 6)
					blob.push_back (Pixel3 (x, y, z, 1000));
			}
	ASSERT_GT (blob.size(), (size_t) 100);

	const int offsets[][3] = { {2, 3, 1}, {2000, 2000, 1}, {4093, 3001, 2500} };
	Surface3DValues base {};
	for (int k = 0; k < 3; k++)
	{
		std::vector<Pixel3> moved;
		for (const auto& v : blob)
			moved.push_back (Pixel3 (v.x + offsets[k][0], v.y + offsets[k][1], v.z + offsets[k][2], v.inten));

		const auto got = calculate_3d_surface_in_ram (moved);
		if (k == 0)
		{
			base = got;
			ASSERT_GT (base.hull_volume, base.mesh_volume) << "the blob must be non-convex";
			continue;
		}
		ASSERT_EQ (got.mesh_volume, base.mesh_volume) << "3MESH_VOLUME moved with the ROI, offset " << offsets[k][0];
		ASSERT_EQ (got.hull_volume, base.hull_volume) << "3VOLUME_CONVEXHULL moved with the ROI, offset " << offsets[k][0];
		ASSERT_EQ (got.area, base.area) << "3AREA moved with the ROI, offset " << offsets[k][0];
	}
}

// A lattice-discretised ball. 3MESH_VOLUME converges on the smooth sphere as the radius grows --
// -3.6% at r=5, -0.15% by r=15 -- so 1% is a real bound there. The voxel count meets that bound too,
// so the mesh volume is also held below it: on a convex body the surface through the face centres
// cuts every boundary corner off the union of voxel cubes, and a volume equal to the count would be
// counting voxels rather than integrating the surface.
void test_3d_morphology_sphere_mesh_volume_invariant()
{
	for (int r : { 15, 20 })
	{
		std::vector<Pixel3> cloud;
		for (int x = -r; x <= r; x++)
			for (int y = -r; y <= r; y++)
				for (int z = -r; z <= r; z++)
					if (x * x + y * y + z * z <= r * r)
						cloud.push_back (Pixel3(x + 64, y + 64, z + 64, 1000));

		const double mesh_volume = calculate_3d_surface_in_ram (cloud).mesh_volume,
			want_v = 4. / 3. * M_PI * double(r) * r * r;

		ASSERT_LE (std::abs(mesh_volume - want_v) / want_v, 0.01)
			<< "r=" << r << " 3MESH_VOLUME actual=" << mesh_volume << " 4/3 pi r^3=" << want_v;
		ASSERT_LT (mesh_volume, double(cloud.size()))
			<< "r=" << r << " 3MESH_VOLUME is not below the voxel count";
	}
}