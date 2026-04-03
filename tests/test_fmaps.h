#pragma once

#include "../src/nyx/roi_cache.h"
#include "../src/nyx/globals.h"
#include "../src/nyx/environment.h"
#include "test_main_nyxus.h"

namespace Nyxus
{
	/// Helper: populates a test parent ROI (output param to avoid copy).
	static void make_test_parent (
		LR& parent, int W, int H,
		int xoff = 0, int yoff = 0,
		PixIntens (*intensity_fn)(int, int) = nullptr)
	{
		for (int y = 0; y < H; y++)
			for (int x = 0; x < W; x++)
			{
				int gx = xoff + x, gy = yoff + y;
				PixIntens val = intensity_fn ? intensity_fn(x, y) : 100;
				if (val == 0)
					continue;
				if (parent.aux_area == 0)
					init_label_record_3(parent, gx, gy, val);
				else
					update_label_record_3(parent, gx, gy, val);
				parent.raw_pixels.push_back(Pixel2(gx, gy, val));
			}
		parent.make_nonanisotropic_aabb();
		parent.aux_image_matrix.allocate(W, H);
		parent.aux_image_matrix.calculate_from_pixelcloud(parent.raw_pixels, parent.aabb);
	}

	/// Helper: run generateChildRois and return the count.
	struct FmapTestResult
	{
		int nChildren;
		std::unordered_set<int> childLabels;
		std::unordered_map<int, LR> childRoiData;
		std::unordered_map<int, FmapChildInfo> childToParentMap;
	};

	static FmapTestResult run_generate (const LR& parent, int kernel_size, int64_t startLabel = 1)
	{
		FmapTestResult r;
		r.nChildren = generateChildRois(parent, kernel_size, r.childLabels, r.childRoiData, r.childToParentMap, startLabel);
		return r;
	}

	// Intensity functions for tests that need non-uniform values
	static PixIntens intensity_varying (int x, int y) { return 10 * (y + 1) + (x + 1); }
	static PixIntens intensity_offset (int x, int y) { return (PixIntens)(x + y + 1); }
	static PixIntens intensity_checkerboard (int x, int y) { return ((x + y) % 2 == 0) ? 100 : 0; }

	/// Test correct child count and output container sizes
	static void test_fmaps_child_roi_count()
	{
		LR parent(1);
		make_test_parent(parent, 8, 8);
		auto res = run_generate(parent, 3);

		int expected = (8 - 2) * (8 - 2); // 36
		if (res.nChildren != expected)
			throw std::runtime_error("Expected " + std::to_string(expected) + " children, got " + std::to_string(res.nChildren));
		if ((int)res.childLabels.size() != res.nChildren)
			throw std::runtime_error("childLabels size mismatch");
		if ((int)res.childRoiData.size() != res.nChildren)
			throw std::runtime_error("childRoiData size mismatch");
		if ((int)res.childToParentMap.size() != res.nChildren)
			throw std::runtime_error("childToParentMap size mismatch");
	}

	/// Test that each child AABB is kernel_size x kernel_size
	static void test_fmaps_child_roi_dimensions()
	{
		LR parent(1);
		make_test_parent(parent, 6, 6);
		auto res = run_generate(parent, 3);

		for (auto& [label, child] : res.childRoiData)
		{
			int cw = child.aabb.get_width(), ch = child.aabb.get_height();
			if (cw != 3 || ch != 3)
				throw std::runtime_error("Child " + std::to_string(label) +
					" dimensions " + std::to_string(cw) + "x" + std::to_string(ch) + ", expected 3x3");
		}
	}

	/// Test that all children map back to the correct parent label
	static void test_fmaps_parent_mapping()
	{
		LR parent(42);
		make_test_parent(parent, 5, 5);
		auto res = run_generate(parent, 3);

		for (auto& [label, info] : res.childToParentMap)
		{
			if (info.parent_label != 42)
				throw std::runtime_error("Child " + std::to_string(label) +
					" has parent_label " + std::to_string(info.parent_label) + ", expected 42");
		}
	}

	/// Test that child pixel intensities match the parent's image matrix
	static void test_fmaps_child_pixel_values()
	{
		LR parent(1);
		make_test_parent(parent, 5, 5, 0, 0, intensity_varying);
		auto res = run_generate(parent, 3);

		// Find the child centered at (2,2)
		for (auto& [label, info] : res.childToParentMap)
		{
			if (info.center_x == 2 && info.center_y == 2)
			{
				const LR& child = res.childRoiData.at(label);
				if (child.aux_area != 9)
					throw std::runtime_error("Center child should have 9 pixels, got " + std::to_string(child.aux_area));

				for (const auto& px : child.raw_pixels)
				{
					PixIntens expected = 10 * (px.y + 1) + (px.x + 1);
					if (px.inten != expected)
						throw std::runtime_error("Pixel (" + std::to_string(px.x) + "," + std::to_string(px.y) +
							") intensity " + std::to_string(px.inten) + ", expected " + std::to_string(expected));
				}
				return;
			}
		}
		throw std::runtime_error("Could not find child ROI at center (2,2)");
	}

	/// Test coordinate handling with an offset parent (not at origin)
	static void test_fmaps_nonorigin_parent()
	{
		const int xoff = 50, yoff = 100;
		LR parent(1);
		make_test_parent(parent, 6, 6, xoff, yoff, intensity_offset);
		auto res = run_generate(parent, 3);

		int expected = (6 - 2) * (6 - 2);
		if (res.nChildren != expected)
			throw std::runtime_error("Nonorigin parent: expected " + std::to_string(expected) +
				" children, got " + std::to_string(res.nChildren));

		for (auto& [label, child] : res.childRoiData)
		{
			if (child.aabb.get_xmin() < xoff || child.aabb.get_ymin() < yoff)
				throw std::runtime_error("Child AABB not offset correctly");
		}

		for (auto& [label, info] : res.childToParentMap)
		{
			if (info.center_x < xoff || info.center_y < yoff)
				throw std::runtime_error("Child center not in global coordinates");
		}
	}

	/// Test sparse mask: checkerboard pattern exercises the zero-center skip path
	static void test_fmaps_sparse_mask()
	{
		LR parent(1);
		make_test_parent(parent, 7, 7, 0, 0, intensity_checkerboard);
		auto res = run_generate(parent, 3);

		// Count expected: kernel centers at (1..5, 1..5), only non-zero centers produce children
		int expectedCount = 0;
		for (int cy = 1; cy <= 5; cy++)
			for (int cx = 1; cx <= 5; cx++)
				if ((cx + cy) % 2 == 0)
					expectedCount++;

		if (res.nChildren != expectedCount)
			throw std::runtime_error("Sparse mask: expected " + std::to_string(expectedCount) +
				" children, got " + std::to_string(res.nChildren));

		for (auto& [label, child] : res.childRoiData)
		{
			if (child.aux_area > 9)
				throw std::runtime_error("Sparse mask: child has too many pixels");
			if (child.aux_area == 0)
				throw std::runtime_error("Sparse mask: child has zero pixels");
		}
	}

	/// Test that parent smaller than kernel produces zero children
	static void test_fmaps_parent_too_small()
	{
		LR parent(1);
		make_test_parent(parent, 3, 3);
		auto res = run_generate(parent, 5);

		if (res.nChildren != 0)
			throw std::runtime_error("Parent too small: expected 0 children, got " + std::to_string(res.nChildren));
	}

	/// Test that startLabel offsets prevent label collisions across batches
	static void test_fmaps_start_label_offset()
	{
		LR parent(1);
		make_test_parent(parent, 5, 5);
		auto r1 = run_generate(parent, 3, 1);
		auto r2 = run_generate(parent, 3, 1 + r1.nChildren);

		for (auto lab : r2.childLabels)
			if (r1.childLabels.count(lab) > 0)
				throw std::runtime_error("Label collision: label " + std::to_string(lab) + " in both batches");

		int minLabel2 = *std::min_element(r2.childLabels.begin(), r2.childLabels.end());
		if (minLabel2 != 1 + r1.nChildren)
			throw std::runtime_error("Second batch min label should be " + std::to_string(1 + r1.nChildren) +
				", got " + std::to_string(minLabel2));
	}
}
