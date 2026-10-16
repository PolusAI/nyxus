#pragma once

// Mechanics of the grey binning every 2D and 3D texture family shares (texture_feature.h). Claims no
// oracle (SPEC 2).

#include <gtest/gtest.h>
#include <stdexcept>
#include "../src/nyx/features/3d_ooc_volume.h"	// bin_roi_3d, and through it texture_feature.h and LR

// PyRadiomics-style binning divides by the ROI's range, and a ROI of one intensity has a range of zero
// width. That range is one bin: every ROI voxel lands in bin 1, and the background, intensity 0, stays
// at 0. 3D GLCM and 3D NGTDM bin such a ROI, having no constant-ROI intercept before their binning, and
// a negative coarse grey depth sends them here.
void test_texture_binning_zero_width_range_mechanics()
{
	EXPECT_EQ (TextureFeature::to_grayscale_radiomix (7, 7, 7, 8), 1u);
	EXPECT_EQ (TextureFeature::to_grayscale_radiomix (7, 7, 7, 1), 1u);
	EXPECT_EQ (TextureFeature::to_grayscale_radiomix (0, 7, 7, 8), 0u);

	// through the cube binning the 3D families call, at a negative grey depth
	SimpleCube<PixIntens> I, D;
	I.allocate (3, 2, 2);
	I.fill (7);
	I.zyx (0, 0, 0) = 0;	// one background cell
	D.allocate (3, 2, 2);
	TextureFeature tf;
	tf.bin_intensities_3d (D, I, 7, 7, -8);
	EXPECT_EQ (D.zyx (0, 0, 0), 0u);
	for (size_t i = 1; i < D.size(); i++)
		EXPECT_EQ (D[i], 1u) << "cell " << i;
}

// The 3D families' in-RAM binning writes each ROI voxel's level at its offset from the box corner and
// leaves every other cell at 0. The box corner is not the origin here, so an offset taken from the
// wrong corner lands a voxel in the wrong cell. The cube takes its shape from the box the voxels are
// checked against, whatever r.aux_image_cube holds: here that is a larger cube, so a binning sized
// from it has 36 cells and a row stride of 4.
void test_texture_binning_roi_cube_places_voxels_mechanics()
{
	LR r;
	r.aux_image_cube.allocate (4, 3, 3);
	r.aabb.init_x (10);	r.aabb.update_x (12);
	r.aabb.init_y (20);	r.aabb.update_y (21);
	r.aabb.init_z (5);	r.aabb.update_z (6);
	r.raw_pixels_3D = { Pixel3 (10, 20, 5, 1), Pixel3 (12, 21, 6, 64), Pixel3 (11, 20, 6, 33) };
	r.aux_min = 1;
	r.aux_max = 64;

	SimpleCube<PixIntens> D;
	Nyxus::bin_roi_3d (D, r, 4);	// MATLAB, 4 levels over [0, 64]: floor(4 x / 64 + 1), clipped to 4
	ASSERT_EQ (D.size(), 12u);
	ASSERT_EQ (D.width(), 3);
	ASSERT_EQ (D.height(), 2);
	ASSERT_EQ (D.depth(), 2);
	EXPECT_EQ (D.zyx (0, 0, 0), 1u);
	EXPECT_EQ (D.zyx (1, 1, 2), 4u);
	EXPECT_EQ (D.zyx (1, 0, 1), 3u);
	size_t background = 0;
	for (auto v : D)
		background += (v == 0);
	EXPECT_EQ (background, 9u);
}

// The refusal path: a voxel outside the box has no cell in the cube, and is refused rather than
// written past it
void test_texture_binning_roi_cube_refuses_voxel_outside_box_mechanics()
{
	LR r;
	r.aux_image_cube.allocate (3, 2, 2);
	r.aabb.init_x (10);	r.aabb.update_x (12);
	r.aabb.init_y (20);	r.aabb.update_y (21);
	r.aabb.init_z (5);	r.aabb.update_z (6);
	r.raw_pixels_3D = { Pixel3 (10, 20, 5, 1), Pixel3 (13, 20, 5, 9) };	// x = 13 is one past the box
	r.aux_min = 1;
	r.aux_max = 9;

	SimpleCube<PixIntens> D;
	EXPECT_THROW (Nyxus::bin_roi_3d (D, r, 4), std::out_of_range);
}

// Negative case: a range of nonzero width still bins by IBSI's fixed bin number,
// floor(n (x - min) / (max - min)) + 1, the top level clipped into bin n.
void test_texture_binning_nonzero_range_unchanged_mechanics()
{
	EXPECT_EQ (TextureFeature::to_grayscale_radiomix (1, 1, 8, 8), 1u);
	EXPECT_EQ (TextureFeature::to_grayscale_radiomix (4, 1, 8, 8), 4u);	// floor(8 * 3 / 7) + 1
	EXPECT_EQ (TextureFeature::to_grayscale_radiomix (8, 1, 8, 8), 8u);
}
