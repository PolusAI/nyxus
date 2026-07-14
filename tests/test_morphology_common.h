#pragma once

#include <gtest/gtest.h>

#include <array>
#include <string>
#include <unordered_map>

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

// "Unvetted" means this table is not claiming an accepted independent
// built-in/package oracle for these rows. Some entries, such as area,
// centroids, and bounding-box limits, are easy to inspect on this small
// fixture; that kind of hand recomputation is useful as a sanity check but is
// not V&V by our tracker definition, because it restates the fixture geometry
// and Nyxus coordinate/spacing conventions rather than using a third-party
// oracle. Rows with accepted external built-in/API comparators live in the
// oracle_3p table below.
static std::unordered_map<std::string, double> unvetted_nyxus_regression_shape2d_feature_golden_values{
	{"AREA_PIXELS_COUNT", 26.0},
	{"AREA_UM2", 104.0},
	{"CENTROID_X", 2.61538461538462},
	{"CENTROID_Y", 2.84615384615385},
	{"WEIGHTED_CENTROID_X", 2.84160305343511},
	{"WEIGHTED_CENTROID_Y", 3.43893129770992},
	{"MASS_DISPLACEMENT", 0.634476074243407},
	{"COMPACTNESS", 0.027517678630878},
	{"BBOX_XMIN", 0.0},
	{"BBOX_YMIN", 0.0},
	{"BBOX_WIDTH", 6.0},
	{"BBOX_HEIGHT", 7.0},
	{"DIAMETER_EQUAL_AREA", 5.75362739175159},
	{"EXTENT", 0.619047619047619},
	{"ASPECT_RATIO", 0.857142857142857},
	{"MAJOR_AXIS_LENGTH", 6.96881616898619},
	{"MINOR_AXIS_LENGTH", 5.48870991295738},
	{"ELONGATION", 0.787610087547462},
	{"ECCENTRICITY", 0.616173960820708},
	{"ORIENTATION", 70.4173944984207},
	{"ROUNDNESS", 0.681656295209303},
	{"PERIMETER", 26.9349412836191},
	{"EDGE_MEAN_INTENSITY", 41.8333333333333},
	{"EDGE_STDDEV_INTENSITY", 16.7691944455582},
	{"EDGE_MAX_INTENSITY", 68.0},
	{"EDGE_MIN_INTENSITY", 12.0},
	{"EDGE_INTEGRATED_INTENSITY", 753.0},
	{"CIRCULARITY", 0.671081973229055},
	{"EULER_NUMBER", 0.0},
	{"DIAMETER_MIN_ENCLOSING_CIRCLE", 6.32475519180298},
	{"ROI_RADIUS_MEAN", 1.07692307692308},
	{"ROI_RADIUS_MAX", 4.0},
	{"ROI_RADIUS_MEDIAN", 1.0},
};

// Fractal dimensions are validated on tests/data/fractal_blob512_seg.ome.tif, a large (512x512)
// irregular ROI. Offline ImageJ/FracLac oracles:
//   FRACT_DIM_BOXCOUNT  = box count of the filled ROI (single origin-aligned grid, the same method
//                         as Nyxus' large-ROI path) -> 1.8706, asserted at 1%.
//   FRACT_DIM_PERIMETER = box count of the ROI edge (cross-method vs Nyxus' Richardson divider,
//                         estimating the same boundary dimension) -> 1.0493, asserted at 3%
//                         (Nyxus divider = 1.0572, a 0.8% gap).
static std::unordered_map<std::string, double> oracle_fractal_blob512_golden_values{
	{"FRACT_DIM_BOXCOUNT", 1.8706},   // vetted by ImageJ/FracLac: an independent implementation of the same box-count method
	{"FRACT_DIM_PERIMETER", 1.0493},  // vetted by ImageJ/FracLac edge box-count: cross-method vs Nyxus' divider
};

static std::unordered_map<std::string, double> oracle_3p_shape2d_feature_golden_values{
	// CONVEX_HULL_AREA / SOLIDITY are cross-checked against scikit-image on this exact ROI. Nyxus
	// computes a Pick's-theorem pixel-count hull area (convex_hull_nontriv.cpp) = 27, solidity
	// 26/27 = 0.9629630. Because Nyxus hulls through pixel CENTRES, this reproduces skimage's
	// convex_hull_image(offset_coordinates=False) == 27 EXACTLY, so we vet against THAT convention
	// (27 / 0.9629630) with a tight 1% tolerance (frac_tolerance=100). The hull area is a
	// provably-exact integer lattice count, so 1% is float/platform slack -- not a convention fudge --
	// and it still catches a >=1 px regression. (skimage's regionprops DEFAULT uses
	// offset_coordinates=True, which first expands every pixel to its +/-0.5 corners and rasterises
	// the hull to 28 / 0.9285714; that +1 px is a corner-expansion convention, not an error, and is
	// why we pin the offset_coordinates=False value rather than the default.) SOLIDITY is thus a real
	// skimage-vetted <= 1 check (unlike the old impossible 1.3), matched exactly rather than within a
	// loose band.
	{"CONVEX_HULL_AREA", 27.0},
	{"SOLIDITY", 0.9629629629629629},
	{"DIAMETER_EQUAL_PERIMETER", 8.57365809435587},
	{"DIAMETER_CIRCUMSCRIBING_CIRCLE", 12.3317073399088},
	{"DIAMETER_INSCRIBING_CIRCLE", 0.828486893405308},
	{"GEODETIC_LENGTH", 10.0},
	{"THICKNESS", 3.0},
	{"EROSIONS_2_VANISH", 1.0},
	{"EXTREMA_P1_X", 2.0},
	{"EXTREMA_P1_Y", 0.0},
	{"EXTREMA_P2_X", 3.0},
	{"EXTREMA_P2_Y", 0.0},
	{"EXTREMA_P3_X", 5.0},
	{"EXTREMA_P3_Y", 2.0},
	{"EXTREMA_P4_X", 5.0},
	{"EXTREMA_P4_Y", 3.0},
	{"EXTREMA_P5_X", 4.0},
	{"EXTREMA_P5_Y", 6.0},
	{"EXTREMA_P6_X", 3.0},
	{"EXTREMA_P6_Y", 6.0},
	{"EXTREMA_P7_X", 0.0},
	{"EXTREMA_P7_Y", 3.0},
	{"EXTREMA_P8_X", 0.0},
	{"EXTREMA_P8_Y", 2.0},
};

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

static void assert_unvetted_no_direct_oracle_shape2d_feature(
	const std::vector<std::vector<double>>& fvals,
	Nyxus::Feature2D feature,
	const std::string& feature_name,
	double frac_tolerance = 1000.0)
{
	SCOPED_TRACE(std::string("UNVETTED_NO_DIRECT_ORACLE__") + feature_name);
	ASSERT_TRUE(unvetted_nyxus_regression_shape2d_feature_golden_values.count(feature_name) > 0);
	ASSERT_TRUE(agrees_gt(fvals[static_cast<int>(feature)][0], unvetted_nyxus_regression_shape2d_feature_golden_values[feature_name], frac_tolerance));
}

static void assert_verifiable_with_3p_builtin_oracle_shape2d_feature(
	const std::vector<std::vector<double>>& fvals,
	Nyxus::Feature2D feature,
	const std::string& feature_name,
	double frac_tolerance = 1000.0)
{
	SCOPED_TRACE(std::string("VERIFIABLE_WITH_3P_BUILTIN_ORACLE__") + feature_name);
	ASSERT_TRUE(oracle_3p_shape2d_feature_golden_values.count(feature_name) > 0);
	ASSERT_TRUE(agrees_gt(fvals[static_cast<int>(feature)][0], oracle_3p_shape2d_feature_golden_values[feature_name], frac_tolerance));
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
