#pragma once

#include "../src/nyx/roi_cache.h"
#include "../src/nyx/globals.h"
#include "../src/nyx/environment.h"
#include "test_main_nyxus.h"
#include "test_fmaps.h"  // reuse FmapTestResult

namespace Nyxus
{
	/// Helper: populates a test 3D parent ROI (output param to avoid copy).
	static void make_test_parent_3D (
		LR& parent, int W, int H, int D,
		int xoff = 0, int yoff = 0, int zoff = 0,
		PixIntens (*intensity_fn)(int, int, int) = nullptr)
	{
		for (int z = 0; z < D; z++)
			for (int y = 0; y < H; y++)
				for (int x = 0; x < W; x++)
				{
					int gx = xoff + x, gy = yoff + y, gz = zoff + z;
					PixIntens val = intensity_fn ? intensity_fn(x, y, z) : 100;
					if (val == 0)
						continue;
					if (parent.aux_area == 0)
						init_label_record_3D(parent, gx, gy, gz, parent.label, val);
					else
						update_label_record_3D(parent, gx, gy, gz, parent.label, val);
					parent.raw_pixels_3D.push_back(Pixel3(gx, gy, gz, val));
				}
		parent.make_nonanisotropic_aabb();
		parent.aux_image_cube.allocate(W, H, D);
		parent.aux_image_cube.calculate_from_pixelcloud(parent.raw_pixels_3D, parent.aabb);
	}

	/// Helper: run generateChildRois_3D and return the count.
	static FmapTestResult run_generate_3D (const LR& parent, int kernel_size, int64_t startLabel = 1)
	{
		FmapTestResult r;
		r.nChildren = generateChildRois_3D(parent, kernel_size, r.childLabels, r.childRoiData, r.childToParentMap, startLabel);
		return r;
	}

	// Intensity functions for 3D tests
	static PixIntens intensity_varying_3D (int x, int y, int z) { return 100 * (z + 1) + 10 * (y + 1) + (x + 1); }
	static PixIntens intensity_offset_3D (int x, int y, int z) { return (PixIntens)(x + y + z + 1); }
	static PixIntens intensity_checkerboard_3D (int x, int y, int z) { return ((x + y + z) % 2 == 0) ? 100 : 0; }

	/// Test correct child count and output container sizes for 3D
	static void test_fmaps3d_child_roi_count()
	{
		LR parent(1);
		make_test_parent_3D(parent, 8, 8, 8);
		auto res = run_generate_3D(parent, 3);

		int expected = (8 - 2) * (8 - 2) * (8 - 2); // 216
		if (res.nChildren != expected)
			throw std::runtime_error("Expected " + std::to_string(expected) + " children, got " + std::to_string(res.nChildren));
		if ((int)res.childLabels.size() != res.nChildren)
			throw std::runtime_error("childLabels size mismatch");
		if ((int)res.childRoiData.size() != res.nChildren)
			throw std::runtime_error("childRoiData size mismatch");
		if ((int)res.childToParentMap.size() != res.nChildren)
			throw std::runtime_error("childToParentMap size mismatch");
	}

	/// Test that each child AABB is kernel_size x kernel_size x kernel_size
	static void test_fmaps3d_child_roi_dimensions()
	{
		LR parent(1);
		make_test_parent_3D(parent, 6, 6, 6);
		auto res = run_generate_3D(parent, 3);

		for (auto& [label, child] : res.childRoiData)
		{
			int cw = child.aabb.get_width(), ch = child.aabb.get_height(), cd = child.aabb.get_z_depth();
			if (cw != 3 || ch != 3 || cd != 3)
				throw std::runtime_error("Child " + std::to_string(label) +
					" dimensions " + std::to_string(cw) + "x" + std::to_string(ch) + "x" + std::to_string(cd) + ", expected 3x3x3");
		}
	}

	/// Test that all children map back to the correct parent label
	static void test_fmaps3d_parent_mapping()
	{
		LR parent(42);
		make_test_parent_3D(parent, 5, 5, 5);
		auto res = run_generate_3D(parent, 3);

		for (auto& [label, info] : res.childToParentMap)
		{
			if (info.parent_label != 42)
				throw std::runtime_error("Child " + std::to_string(label) +
					" has parent_label " + std::to_string(info.parent_label) + ", expected 42");
		}
	}

	/// Test that child voxel intensities match the parent's image cube
	static void test_fmaps3d_child_voxel_values()
	{
		LR parent(1);
		make_test_parent_3D(parent, 5, 5, 5, 0, 0, 0, intensity_varying_3D);
		auto res = run_generate_3D(parent, 3);

		// Find the child centered at (2,2,2)
		for (auto& [label, info] : res.childToParentMap)
		{
			if (info.center_x == 2 && info.center_y == 2 && info.center_z == 2)
			{
				const LR& child = res.childRoiData.at(label);
				if (child.aux_area != 27)
					throw std::runtime_error("Center child should have 27 voxels, got " + std::to_string(child.aux_area));

				for (const auto& px : child.raw_pixels_3D)
				{
					PixIntens expected = 100 * (px.z + 1) + 10 * (px.y + 1) + (px.x + 1);
					if (px.inten != expected)
						throw std::runtime_error("Voxel (" + std::to_string(px.x) + "," + std::to_string(px.y) + "," + std::to_string(px.z) +
							") intensity " + std::to_string(px.inten) + ", expected " + std::to_string(expected));
				}
				return;
			}
		}
		throw std::runtime_error("Could not find child ROI at center (2,2,2)");
	}

	/// Test coordinate handling with an offset parent (not at origin)
	static void test_fmaps3d_nonorigin_parent()
	{
		const int xoff = 50, yoff = 100, zoff = 25;
		LR parent(1);
		make_test_parent_3D(parent, 6, 6, 6, xoff, yoff, zoff, intensity_offset_3D);
		auto res = run_generate_3D(parent, 3);

		int expected = (6 - 2) * (6 - 2) * (6 - 2); // 64
		if (res.nChildren != expected)
			throw std::runtime_error("Nonorigin parent: expected " + std::to_string(expected) +
				" children, got " + std::to_string(res.nChildren));

		for (auto& [label, child] : res.childRoiData)
		{
			if (child.aabb.get_xmin() < xoff || child.aabb.get_ymin() < yoff || child.aabb.get_zmin() < zoff)
				throw std::runtime_error("Child AABB not offset correctly");
		}

		for (auto& [label, info] : res.childToParentMap)
		{
			if (info.center_x < xoff || info.center_y < yoff || info.center_z < zoff)
				throw std::runtime_error("Child center not in global coordinates");
		}
	}

	/// Test sparse mask: 3D checkerboard pattern exercises the zero-center skip path
	static void test_fmaps3d_sparse_mask()
	{
		LR parent(1);
		make_test_parent_3D(parent, 7, 7, 7, 0, 0, 0, intensity_checkerboard_3D);
		auto res = run_generate_3D(parent, 3);

		// Count expected: kernel centers at (1..5, 1..5, 1..5), only non-zero centers produce children
		int expectedCount = 0;
		for (int cz = 1; cz <= 5; cz++)
			for (int cy = 1; cy <= 5; cy++)
				for (int cx = 1; cx <= 5; cx++)
					if ((cx + cy + cz) % 2 == 0)
						expectedCount++;

		if (res.nChildren != expectedCount)
			throw std::runtime_error("Sparse mask: expected " + std::to_string(expectedCount) +
				" children, got " + std::to_string(res.nChildren));

		for (auto& [label, child] : res.childRoiData)
		{
			if (child.aux_area > 27)
				throw std::runtime_error("Sparse mask: child has too many voxels");
			if (child.aux_area == 0)
				throw std::runtime_error("Sparse mask: child has zero voxels");
		}
	}

	/// Test that parent smaller than kernel produces zero children
	static void test_fmaps3d_parent_too_small()
	{
		LR parent(1);
		make_test_parent_3D(parent, 3, 3, 3);
		auto res = run_generate_3D(parent, 5);

		if (res.nChildren != 0)
			throw std::runtime_error("Parent too small: expected 0 children, got " + std::to_string(res.nChildren));
	}

	/// Test that startLabel offsets prevent label collisions across batches
	static void test_fmaps3d_start_label_offset()
	{
		LR parent(1);
		make_test_parent_3D(parent, 5, 5, 5);
		auto r1 = run_generate_3D(parent, 3, 1);
		auto r2 = run_generate_3D(parent, 3, 1 + r1.nChildren);

		for (auto lab : r2.childLabels)
			if (r1.childLabels.count(lab) > 0)
				throw std::runtime_error("Label collision: label " + std::to_string(lab) + " in both batches");

		int minLabel2 = *std::min_element(r2.childLabels.begin(), r2.childLabels.end());
		if (minLabel2 != 1 + r1.nChildren)
			throw std::runtime_error("Second batch min label should be " + std::to_string(1 + r1.nChildren) +
				", got " + std::to_string(minLabel2));
	}
}
