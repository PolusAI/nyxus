#pragma once

// The ROI buffers built from a pixel cloud and its bounding box.
//
// Each writer places a pixel at its offset from the box's low corner. A pixel outside the box has
// no place in the buffer: past the right edge it would land at the start of the next row, and past
// the last row in the memory after the buffer. The in-RAM writers (ImageMatrix, SimpleCube) refuse
// such a pixel by name, on every batch of every path, since all of them build their buffers there.
// The out-of-core writers check the disk-backed cloud the same way: WriteImageMatrix_nontriv's
// allocate_from_cloud and allocate_from_cloud_coarser_grayscale, Power2PaddedImageMatrix_NT, and
// the padded images of the out-of-core contour, Euler number and 3D surface.
//
// The fixture box spans x 10..13, y 20..22 (and z 5..6 in 3D), so a pixel at x 14 maps to the
// first cell of the second row: inside the allocation, which a write would corrupt silently.

#include <gtest/gtest.h>
#include <stdexcept>
#include "../src/nyx/features/3d_surface.h"
#include "../src/nyx/features/contour.h"
#include "../src/nyx/features/euler_number.h"
#include "../src/nyx/features/image_cube.h"
#include "../src/nyx/features/image_matrix_nontriv.h"

static AABB buffer_from_cloud_box_2d()
{
	AABB box;
	box.init_x (10);
	box.update_x (13);
	box.init_y (20);
	box.update_y (22);
	return box;
}

static AABB buffer_from_cloud_box_3d()
{
	AABB box = buffer_from_cloud_box_2d();
	box.init_z (5);
	box.update_z (6);
	return box;
}

// A cloud inside its box is written at its offsets
void test_buffer_from_cloud_places_pixels_mechanics()
{
	AABB box = buffer_from_cloud_box_2d();
	std::vector<Pixel2> cloud = { {10, 20, (PixIntens) 7}, {13, 20, (PixIntens) 8}, {10, 21, (PixIntens) 9}, {13, 22, (PixIntens) 11} };

	ImageMatrix m;
	m.allocate (box.get_width(), box.get_height());
	ASSERT_NO_THROW (m.calculate_from_pixelcloud (cloud, box));
	const pixData& P = m.ReadablePixels();
	EXPECT_EQ (P.yx(0, 0), 7u);
	EXPECT_EQ (P.yx(0, 3), 8u);
	EXPECT_EQ (P.yx(1, 0), 9u);
	EXPECT_EQ (P.yx(2, 3), 11u);
	EXPECT_EQ (P.yx(1, 1), 0u);

	ImageMatrix c (cloud, box);
	EXPECT_EQ (c.ReadablePixels().yx(2, 3), 11u);

	AABB box3 = buffer_from_cloud_box_3d();
	std::vector<Pixel3> cloud3 = { {10, 20, 5, (PixIntens) 7}, {13, 22, 6, (PixIntens) 11} };
	SimpleCube<PixIntens> cube;
	ASSERT_NO_THROW (cube.calculate_from_pixelcloud (cloud3, box3));
	EXPECT_EQ (cube.xyz(0, 0, 0), 7u);
	EXPECT_EQ (cube.xyz(3, 2, 1), 11u);
}

// A pixel past the box's right edge is refused, not written into the next row
void test_buffer_from_cloud_refuses_pixel_outside_box_mechanics()
{
	AABB box = buffer_from_cloud_box_2d();
	std::vector<Pixel2> cloud = { {10, 20, (PixIntens) 7}, {14, 20, (PixIntens) 8} };

	ImageMatrix m;
	m.allocate (box.get_width(), box.get_height());
	EXPECT_THROW (m.calculate_from_pixelcloud (cloud, box), std::out_of_range);

	EXPECT_THROW (ImageMatrix c (cloud, box), std::out_of_range);

	AABB box3 = buffer_from_cloud_box_3d();
	std::vector<Pixel3> cloud3 = { {10, 20, 5, (PixIntens) 7}, {14, 20, 5, (PixIntens) 8} };
	SimpleCube<PixIntens> cube;
	EXPECT_THROW (cube.calculate_from_pixelcloud (cloud3, box3), std::out_of_range);
}

// An empty voxel cloud has no box to report
void test_buffer_from_cloud_refuses_empty_voxel_cloud_mechanics()
{
	AABB box;
	EXPECT_THROW (box.update_from_voxelcloud (std::vector<Pixel3>()), std::invalid_argument);

	std::vector<Pixel3> cloud = { {3, 4, 5, (PixIntens) 1}, {6, 2, 9, (PixIntens) 1} };
	ASSERT_NO_THROW (box.update_from_voxelcloud (cloud));
	EXPECT_EQ (box.get_xmin(), 3);
	EXPECT_EQ (box.get_xmax(), 6);
	EXPECT_EQ (box.get_ymin(), 2);
	EXPECT_EQ (box.get_ymax(), 4);
	EXPECT_EQ (box.get_zmin(), 5);
	EXPECT_EQ (box.get_zmax(), 9);
}

// Every face of the box refuses the pixel one step past it, and every corner just inside is accepted.
// The check is called on its own, with no buffer, so a pixel on any side is safe to try.
void test_buffer_from_cloud_box_faces_mechanics()
{
	AABB box = buffer_from_cloud_box_2d();
	for (StatsInt x : {10, 13})
		for (StatsInt y : {20, 22})
			EXPECT_NO_THROW (box.require_contains (x, y)) << "corner (" << x << "," << y << ")";
	EXPECT_THROW (box.require_contains (9, 21), std::out_of_range) << "x below xmin";
	EXPECT_THROW (box.require_contains (14, 21), std::out_of_range) << "x above xmax";
	EXPECT_THROW (box.require_contains (11, 19), std::out_of_range) << "y below ymin";
	EXPECT_THROW (box.require_contains (11, 23), std::out_of_range) << "y above ymax";

	AABB box3 = buffer_from_cloud_box_3d();
	for (StatsInt x : {10, 13})
		for (StatsInt y : {20, 22})
			for (StatsInt z : {5, 6})
				EXPECT_NO_THROW (box3.require_contains (x, y, z)) << "corner (" << x << "," << y << "," << z << ")";
	EXPECT_THROW (box3.require_contains (9, 21, 5), std::out_of_range) << "x below xmin";
	EXPECT_THROW (box3.require_contains (14, 21, 5), std::out_of_range) << "x above xmax";
	EXPECT_THROW (box3.require_contains (11, 19, 5), std::out_of_range) << "y below ymin";
	EXPECT_THROW (box3.require_contains (11, 23, 5), std::out_of_range) << "y above ymax";
	EXPECT_THROW (box3.require_contains (11, 21, 4), std::out_of_range) << "z below zmin";
	EXPECT_THROW (box3.require_contains (11, 21, 7), std::out_of_range) << "z above zmax";
}

// The out-of-core writers keep their buffer in a temporary file, so a pixel outside the box lands
// at a wrong offset in that file rather than outside any allocation.
static void buffer_from_cloud_fill_box_2d (OutOfRamPixelCloud& cloud, const std::string& name, bool with_outside_pixel)
{
	cloud.init (1, name);
	for (StatsInt y = 20; y <= 22; y++)
		for (StatsInt x = 10; x <= 13; x++)
			cloud.add_pixel (Pixel2 (x, y, (PixIntens) (1 + (x - 10) + 4 * (y - 20))));
	if (with_outside_pixel)
		cloud.add_pixel (Pixel2 ((StatsInt) 14, (StatsInt) 20, (PixIntens) 99));
}

// A disk-backed cloud inside its box is written at its offsets
void test_buffer_from_cloud_ooc_places_pixels_mechanics()
{
	AABB box = buffer_from_cloud_box_2d();
	OutOfRamPixelCloud cloud;
	cloud.init (1, "buffer_from_cloud_ooc_places");
	cloud.add_pixel (Pixel2 ((StatsInt) 10, (StatsInt) 20, (PixIntens) 7));
	cloud.add_pixel (Pixel2 ((StatsInt) 13, (StatsInt) 20, (PixIntens) 8));
	cloud.add_pixel (Pixel2 ((StatsInt) 10, (StatsInt) 21, (PixIntens) 9));
	cloud.add_pixel (Pixel2 ((StatsInt) 13, (StatsInt) 22, (PixIntens) 11));

	{
		WriteImageMatrix_nontriv m ("buffer_from_cloud_ooc_places_m", 1);
		ASSERT_NO_THROW (m.allocate_from_cloud (cloud, box, false));
		EXPECT_EQ (m.yx (0, 0), 7.0);
		EXPECT_EQ (m.yx (0, 3), 8.0);
		EXPECT_EQ (m.yx (1, 0), 9.0);
		EXPECT_EQ (m.yx (2, 3), 11.0);
		EXPECT_EQ (m.yx (1, 1), 0.0);
	}

	{
		Power2PaddedImageMatrix_NT p ("buffer_from_cloud_ooc_places_p", 1, cloud, box, 0, 1.0);
		EXPECT_EQ (p.get_width(), 4u);
		EXPECT_EQ (p.yx (0, 3), 8.0);
		EXPECT_EQ (p.yx (2, 3), 11.0);
		EXPECT_EQ (p.yx (3, 3), 0.0);
	}

	cloud.clear();
}

// A disk-backed cloud with a pixel past the box's right edge is refused by every out-of-core
// writer, and the same cloud without it is accepted
void test_buffer_from_cloud_ooc_refuses_pixel_outside_box_mechanics()
{
	AABB box = buffer_from_cloud_box_2d();

	for (bool outside : {false, true})
	{
		SCOPED_TRACE (outside ? "with a pixel at x 14" : "inside the box");
		OutOfRamPixelCloud cloud;
		buffer_from_cloud_fill_box_2d (cloud, "buffer_from_cloud_ooc_refuses", outside);

		auto expect_refused = [outside](auto write, const char* writer)
		{
			if (outside)
				EXPECT_THROW (write(), std::out_of_range) << writer;
			else
				EXPECT_NO_THROW (write()) << writer;
		};

		expect_refused ([&]()
			{
				WriteImageMatrix_nontriv m ("buffer_from_cloud_ooc_refuses_m", 1);
				m.allocate_from_cloud (cloud, box, false);
			}, "allocate_from_cloud");
		expect_refused ([&]()
			{
				WriteImageMatrix_nontriv m ("buffer_from_cloud_ooc_refuses_g", 1);
				m.allocate_from_cloud_coarser_grayscale (cloud, box, 1, 99, 8);
			}, "allocate_from_cloud_coarser_grayscale");
		expect_refused ([&]()
			{
				Power2PaddedImageMatrix_NT p ("buffer_from_cloud_ooc_refuses_p", 1, cloud, box, 1, 1.0);
			}, "Power2PaddedImageMatrix_NT");

		cloud.clear();
	}
}

// The out-of-core contour, Euler number and 3D surface build their padded images from the ROI's
// disk-backed cloud and box: a pixel outside the box is refused, and the same ROI without it is not
void test_buffer_from_cloud_ooc_features_refuse_pixel_outside_box_mechanics()
{
	Fsettings s;
	s.resize ((int) NyxSetting::__COUNT__);
	s[(int) NyxSetting::SINGLEROI].bval = false;
	s[(int) NyxSetting::VERBOSLVL].ival = 0;
	ImageLoader dummy;	// these read their pixels from the cloud, not the loader

	for (bool outside : {false, true})
	{
		SCOPED_TRACE (outside ? "with a pixel past the box" : "inside the box");

		{
			LR r (1);
			r.aabb = buffer_from_cloud_box_2d();
			buffer_from_cloud_fill_box_2d (r.raw_pixels_NT, "buffer_from_cloud_ooc_contour", outside);
			ContourFeature f;
			if (outside)
				EXPECT_THROW (f.osized_calculate (r, s, dummy), std::out_of_range) << "contour";
			else
				EXPECT_NO_THROW (f.osized_calculate (r, s, dummy)) << "contour";
			r.raw_pixels_NT.clear();
		}

		{
			LR r (1);
			r.aabb = buffer_from_cloud_box_2d();
			buffer_from_cloud_fill_box_2d (r.raw_pixels_NT, "buffer_from_cloud_ooc_euler", outside);
			EulerNumberFeature f;
			if (outside)
				EXPECT_THROW (f.osized_calculate (r, s, dummy), std::out_of_range) << "Euler number";
			else
				EXPECT_NO_THROW (f.osized_calculate (r, s, dummy)) << "Euler number";
			r.raw_pixels_NT.clear();
		}

		{
			LR r (1);
			r.aabb = buffer_from_cloud_box_3d();
			r.raw_voxels_NT.init (r.label, "buffer_from_cloud_ooc_surface");
			for (StatsInt z = 5; z <= 6; z++)
			{
				r.raw_voxels_NT.begin_slab ((size_t) z);
				for (StatsInt y = 20; y <= 22; y++)
					for (StatsInt x = 10; x <= 13; x++)
						r.raw_voxels_NT.add_voxel (Pixel3 (x, y, z, (PixIntens) 1));
				if (outside && z == 5)
					r.raw_voxels_NT.add_voxel (Pixel3 ((StatsInt) 14, (StatsInt) 20, z, (PixIntens) 1));
			}
			D3_SurfaceFeature f;
			if (outside)
				EXPECT_THROW (f.osized_calculate (r, s, dummy), std::out_of_range) << "3D surface";
			else
				EXPECT_NO_THROW (f.osized_calculate (r, s, dummy)) << "3D surface";
			r.raw_voxels_NT.clear();
		}
	}
}
