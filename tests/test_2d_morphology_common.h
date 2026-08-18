#pragma once

#include <gtest/gtest.h>

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

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
#include "../src/nyx/features/caliper.h"
#include "../src/nyx/features/chords.h"
#include "../src/nyx/features/hexagonality_polygonality.h"
#include "../src/nyx/features/neighbors.h"
#include "../src/nyx/features/radial_distribution.h"
#include "../src/nyx/features/zernike.h"
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

// Every 2D shape feature the tests read off the 8x8 shape2d fixture: basic morphology, contour,
// hull, ellipse fit, extrema, radii, Euler number, fractal dimension, circles, geodetic length,
// calipers, chords, erosions, radial distribution and Zernike moments. One ROI, one pass -- the
// feature classes are independent, so a caller that reads only a few of them still gets the same
// numbers it would from a narrower fixture.
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

	CaliperFeretFeature feret;
	feret.calculate(roidata, s);
	feret.save_value(roidata.fvals);

	CaliperMartinFeature martin;
	martin.calculate(roidata, s);
	martin.save_value(roidata.fvals);

	CaliperNassensteinFeature nassenstein;
	nassenstein.calculate(roidata, s);
	nassenstein.save_value(roidata.fvals);

	ChordsFeature chords;
	chords.calculate(roidata, s);
	chords.save_value(roidata.fvals);

	ErosionPixelsFeature erosion;
	erosion.calculate(roidata, s);
	erosion.save_value(roidata.fvals);

	RadialDistributionFeature radial;
	radial.calculate(roidata, s);
	radial.save_value(roidata.fvals);

	ZernikeFeature zernike;
	zernike.calculate(roidata, s);
	zernike.save_value(roidata.fvals);

	fvals = roidata.fvals;
}

// The multi-ROI neighborhood scene, one LR per label: hexagonality and polygonality need the
// neighbor reduction, so every ROI gets its shape features and then NeighborsFeature runs over
// the whole set.
static void calculate_polygonality_scene_feature_values(std::unordered_map<int, LR>& roiData)
{
	Fsettings s = make_shape2d_settings();
	s[static_cast<int>(NyxSetting::PIXELSIZEUM)].rval = 1.0;
	std::unordered_set<int> uniqueLabels;

	for (const auto& px : neighborhood2d_scene_labels)
	{
		int label = static_cast<int>(px.intensity);
		uniqueLabels.insert(label);

		auto [it, inserted] = roiData.try_emplace(label, label);
		LR& roi = it->second;

		if (inserted)
			init_label_record_3(roi, static_cast<int>(px.x), static_cast<int>(px.y), 1);
		else
			update_label_record_3(roi, static_cast<int>(px.x), static_cast<int>(px.y), 1);

		roi.raw_pixels.push_back(Pixel2(static_cast<size_t>(px.x), static_cast<size_t>(px.y), static_cast<PixIntens>(1)));
	}

	BasicMorphologyFeatures basic;
	ContourFeature contour;
	ConvexHullFeature hull;
	CaliperFeretFeature feret;
	for (auto& item : roiData)
	{
		LR& roi = item.second;
		roi.make_nonanisotropic_aabb();
		roi.aux_image_matrix = ImageMatrix(roi.raw_pixels);
		roi.initialize_fvals();

		basic.calculate(roi, s);
		basic.save_value(roi.fvals);

		contour.calculate(roi, s);
		contour.save_value(roi.fvals);

		hull.calculate(roi, s);
		hull.save_value(roi.fvals);

		feret.calculate(roi, s);
		feret.save_value(roi.fvals);
	}

	NeighborsFeature::manual_reduce(roiData, s, uniqueLabels);

	HexagonalityPolygonalityFeature hexpoly;
	for (auto& item : roiData)
	{
		hexpoly.calculate(item.second, s);
		hexpoly.save_value(item.second.fvals);
	}
}

// Build a filled ellipse (a=20, b=10) ROI and compute its caliper features. Mirrors the
// rasterization in morph_oracle/caliper_proto.py so the imea reference values in
// test_2d_morphology_imea.h line up.
static void calculate_ellipse_caliper_values(std::vector<std::vector<double>>& fvals)
{
	Fsettings s = make_shape2d_settings();

	LR roi(1);
	const double a = 20.0, b = 10.0, cx = 26.0, cy = 16.0;	// pad=6, matches the prototype fixture
	bool first = true;
	for (int y = 0; y <= 32; y++)
		for (int x = 0; x <= 52; x++)
		{
			double dx = (x - cx) / a, dy = (y - cy) / b;
			if (dx * dx + dy * dy <= 1.0)
			{
				if (first)
				{
					init_label_record_3(roi, x, y, 1);
					first = false;
				}
				else
					update_label_record_3(roi, x, y, 1);
				roi.raw_pixels.push_back(Pixel2(static_cast<size_t>(x), static_cast<size_t>(y), static_cast<PixIntens>(1)));
			}
		}
	roi.make_nonanisotropic_aabb();
	roi.aux_image_matrix = ImageMatrix(roi.raw_pixels);
	roi.initialize_fvals();

	BasicMorphologyFeatures basic;	// provides CENTROID_X/Y for the circle features
	basic.calculate(roi, s);
	basic.save_value(roi.fvals);

	ContourFeature contour;
	contour.calculate(roi, s);
	contour.save_value(roi.fvals);

	ConvexHullFeature hull;
	hull.calculate(roi, s);
	hull.save_value(roi.fvals);

	CaliperFeretFeature feret;
	feret.calculate(roi, s);
	feret.save_value(roi.fvals);

	CaliperMartinFeature martin;
	martin.calculate(roi, s);
	martin.save_value(roi.fvals);

	CaliperNassensteinFeature nassenstein;
	nassenstein.calculate(roi, s);
	nassenstein.save_value(roi.fvals);

	// ALLCHORDS_MIN is vetted against imea on this clean fixture, not on the 8x8 raster
	ChordsFeature chords;
	chords.calculate(roi, s);
	chords.save_value(roi.fvals);

	EnclosingInscribingCircumscribingCircleFeature circle;
	circle.calculate(roi, s);
	circle.save_value(roi.fvals);

	fvals = roi.fvals;
}

// Build a filled circle (r=15) ROI and compute basic morphology + contour + the 3 circle diameters.
static void calculate_circle_shape_values(std::vector<std::vector<double>>& fvals)
{
	Fsettings s = make_shape2d_settings();

	LR roi(2);
	const double r = 15.0, cx = 21.0, cy = 21.0;	// pad=6
	bool first = true;
	for (int y = 0; y <= 42; y++)
		for (int x = 0; x <= 42; x++)
		{
			double dx = (x - cx) / r, dy = (y - cy) / r;
			if (dx * dx + dy * dy <= 1.0)
			{
				if (first) { init_label_record_3(roi, x, y, 1); first = false; }
				else update_label_record_3(roi, x, y, 1);
				roi.raw_pixels.push_back(Pixel2(static_cast<size_t>(x), static_cast<size_t>(y), static_cast<PixIntens>(1)));
			}
		}
	roi.make_nonanisotropic_aabb();
	roi.aux_image_matrix = ImageMatrix(roi.raw_pixels);
	roi.initialize_fvals();

	BasicMorphologyFeatures basic;
	basic.calculate(roi, s);
	basic.save_value(roi.fvals);

	ContourFeature contour;
	contour.calculate(roi, s);
	contour.save_value(roi.fvals);

	EnclosingInscribingCircumscribingCircleFeature circle;
	circle.calculate(roi, s);
	circle.save_value(roi.fvals);

	fvals = roi.fvals;
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

// Fixtures only, no reference data and no assertions (SPEC 6.3.1). The tables that used to live here
// now sit with the assertions that read them, and the two shared assert helpers are split per oracle.
// They had to be: one lookup searched the matlab, skimage, imea and regression tables in a single
// pass, so a _matlab assertion could resolve its "oracle" value out of a snapshot, and one helper
// judged _skimage and _cellprofiler functions against morphology_2d_regression_ref_vals -- the SPEC
// 6.2.1 defect, reached through file layout rather than through naming.
//
// This is the only shared 2D shape fixture header. The zernike and radial regression files include it
// too: their features are measured on the shape2d ROI built here, and SPEC 6.3.1 shares fixtures
// across families freely -- it is reference data that may not be shared.
