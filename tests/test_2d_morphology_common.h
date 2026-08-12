#pragma once

#include <gtest/gtest.h>

#include <string>

#include "../src/nyx/roi_cache.h"
#include "../src/nyx/features/basic_morphology.h"
#include "../src/nyx/features/contour.h"
#include "../src/nyx/features/convex_hull.h"
#include "../src/nyx/features/ellipse_fitting.h"
#include "../src/nyx/features/extrema.h"
#include "../src/nyx/features/roi_radius.h"
#include "../src/nyx/features/euler_number.h"
#include "../src/nyx/features/fractal_dim.h"
#include "../src/nyx/features/circle.h"
#include "../src/nyx/features/geodetic_len_thickness.h"
#include "../src/nyx/features/erosion.h"
#include "test_data.h"
#include "test_main_nyxus.h"

#include <filesystem>
#include <memory>
#include "../src/nyx/grayscale_tiff.h"

static Fsettings make_shape2d_settings()
{
	Fsettings s;
	s.resize(static_cast<int>(NyxSetting::__COUNT__));
	s[static_cast<int>(NyxSetting::SOFTNAN)].rval = 0.0;
	s[static_cast<int>(NyxSetting::TINY)].rval = 0.0;
	s[static_cast<int>(NyxSetting::SINGLEROI)].bval = false;
	s[static_cast<int>(NyxSetting::GREYDEPTH)].ival = 128;
	s[static_cast<int>(NyxSetting::PIXELSIZEUM)].rval = 2.0;
	s[static_cast<int>(NyxSetting::XYRES)].rval = 1.0;
	s[static_cast<int>(NyxSetting::PIXELDISTANCE)].ival = 1;
	s[static_cast<int>(NyxSetting::USEGPU)].bval = false;
	s[static_cast<int>(NyxSetting::VERBOSLVL)].ival = 0;
	s[static_cast<int>(NyxSetting::IBSI)].bval = false;
	return s;
}

static void calculate_shape2d_feature_values(std::vector<std::vector<double>>& fvals)
{
	Fsettings s = make_shape2d_settings();

	LR roidata(101);
	load_masked_test_roi_data(
		roidata,
		shape2d_morphology_intensity,
		shape2d_morphology_mask,
		sizeof(shape2d_morphology_mask) / sizeof(NyxusPixel));
	roidata.initialize_fvals();

	BasicMorphologyFeatures basic;
	basic.calculate(roidata, s);
	basic.save_value(roidata.fvals);

	ContourFeature contour;
	contour.calculate(roidata, s);
	contour.save_value(roidata.fvals);

	ConvexHullFeature hull;
	hull.calculate(roidata, s);
	hull.save_value(roidata.fvals);

	EllipseFittingFeature ellipse;
	ellipse.calculate(roidata, s);
	ellipse.save_value(roidata.fvals);

	ExtremaFeature extrema;
	extrema.calculate(roidata, s);
	extrema.save_value(roidata.fvals);

	RoiRadiusFeature radius;
	radius.calculate(roidata, s);
	radius.save_value(roidata.fvals);

	EulerNumberFeature euler;
	euler.calculate(roidata, s);
	euler.save_value(roidata.fvals);

	FractalDimensionFeature fractal;
	fractal.calculate(roidata, s);
	fractal.save_value(roidata.fvals);

	EnclosingInscribingCircumscribingCircleFeature circle;
	circle.calculate(roidata, s);
	circle.save_value(roidata.fvals);

	GeodeticLengthThicknessFeature geodetic;
	geodetic.calculate(roidata, s);
	geodetic.save_value(roidata.fvals);

	ErosionPixelsFeature erosion;
	erosion.calculate(roidata, s);
	erosion.save_value(roidata.fvals);

	fvals = roidata.fvals;
}



// Loads the large ROI mask tests/data/fractal_blob512_seg.ome.tif (path resolved relative to this
// source file) into a single-ROI LR and computes the fractal features.
static void calculate_fractal_blob512_feature_values(std::vector<std::vector<double>>& fvals)
{
	Fsettings s = make_shape2d_settings();

	std::filesystem::path here(__FILE__);
	std::string path = (here.parent_path() / "data" / "fractal_blob512_seg.ome.tif").string();
	NyxusGrayscaleTiffStripLoader<uint16_t> loader(1, path);
	size_t W = loader.fullWidth(0), H = loader.fullHeight(0);
	size_t tw = loader.tileWidth(0), th = loader.tileHeight(0), td = loader.tileDepth(0);
	auto tile = std::make_shared<std::vector<uint16_t>>(tw * th * td);
	loader.loadTileFromFile(tile, 0, 0, 0, 0);	// 512x512 <= 1024 strip tile: one read

	std::vector<NyxusPixel> px;
	for (size_t y = 0; y < H; y++)
		for (size_t x = 0; x < W; x++)
		{
			uint16_t v = (*tile)[y * tw + x];
			if (v != 0)
				px.push_back(NyxusPixel{ x, y, (unsigned int)v });
		}

	// Finalize the ROI through the shared masked-loader helper (raw_pixels -> AABB -> image matrix)
	// instead of re-implementing those steps inline, so this fixture tracks the same finalize path
	// the other 2D shape tests use. (intensity == mask here: the shape features ignore intensity.)
	LR roidata(1);
	load_masked_test_roi_data(roidata, px.data(), px.data(), px.size());
	roidata.initialize_fvals();

	BasicMorphologyFeatures basic;
	basic.calculate(roidata, s);
	basic.save_value(roidata.fvals);

	// FRACT_DIM_PERIMETER walks the contour, so populate it before the fractal feature
	ContourFeature contour;
	contour.calculate(roidata, s);
	contour.save_value(roidata.fvals);

	FractalDimensionFeature fractal;
	fractal.calculate(roidata, s);
	fractal.save_value(roidata.fvals);

	fvals = roidata.fvals;
}

// Fixtures only, no reference data and no assertions (SPEC 6.3.1). The six tables that used to live
// here now sit with the assertions that read them, and the two shared assert helpers are split per
// oracle. They had to be: shape2d_ref_val() searched the matlab, skimage, imea and regression tables
// in one pass, so a _matlab assertion could resolve its "oracle" value out of a snapshot, and
// assert_unvetted_no_direct_oracle_shape2d_feature() judged _skimage and _cellprofiler functions
// against morphology_2d_regression_ref_vals -- the SPEC 6.2.1 defect, reached through file layout
// rather than through naming.
