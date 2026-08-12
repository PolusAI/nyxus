#pragma once

#include <gtest/gtest.h>

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "../src/nyx/roi_cache.h"
#include "../src/nyx/features/basic_morphology.h"
#include "../src/nyx/features/caliper.h"
#include "../src/nyx/features/chords.h"
#include "../src/nyx/features/circle.h"
#include "../src/nyx/features/contour.h"
#include "../src/nyx/features/convex_hull.h"
#include "../src/nyx/features/erosion.h"
#include "../src/nyx/features/hexagonality_polygonality.h"
#include "../src/nyx/features/neighbors.h"
#include "../src/nyx/features/radial_distribution.h"
#include "../src/nyx/features/zernike.h"
#include "test_data.h"
#include "test_main_nyxus.h"

// The 19 caliper statistics below are vetted against imea (registry: status=vetted,
// oracle=imea). They shared a table with 14 status=regression keys until SPEC 6.3.1 required
// a table to name one oracle; the snapshot half now lives in morphology_2d_regression_caliper_chords_ref_vals.

// Pinned Nyxus output: erosion complement, Feret angles and chord statistics. No third-party
// oracle backs these, so the name says regression rather than borrowing the imea claim of the
// caliper table above.





// ZERNIKE2D golden vector. Named for what it is: a pinned Nyxus snapshot, not a third-party
// oracle. The registry (MIGRATION 6.1) does not accept mahotas, the only tool that computes
// Zernike moments, so ZERNIKE2D is regression-only and this table and its assert are named to
// match. Sole key: ZERNIKE2D.

// ---------------------------------------------------------------------------------------------------
// Martin / Nassenstein caliper vetting vs imea (external oracle).
//
// The 8x8 shape2d fixture above is too small/aliased to serve as a tight caliper oracle, so the
// corrected Martin (area-bisecting chord) and Nassenstein (bottom-tangent vertical chord) diameters
// are vetted on a clean, larger convex fixture: a filled ellipse a=20, b=10 (same rasterization as
// morph_oracle/caliper_proto.py). imea (imea.measure_2d.statistical_length, dalpha=10) is the
// reference. Nyxus rotates the convex hull and measures analytically while imea rotates the filled
// raster, so the two agree only up to a ~1-2px hull-vs-raster convention gap (same gap already
// accepted for Feret) — hence a 10% relative tolerance on the robust stats. The point that this pins
// is that the diameters are now the *correct* quantities (min > 0), not the old min+max-chord bug
// that produced physically-impossible 0-length Nassenstein diameters.

static Fsettings make_remaining2d_settings()
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

static void calculate_remaining2d_shape_feature_values(std::vector<std::vector<double>>& fvals)
{
	Fsettings s = make_remaining2d_settings();

	LR roidata(1201);
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

static void calculate_remaining2d_polygonality_feature_values(std::unordered_map<int, LR>& roiData)
{
	Fsettings s = make_remaining2d_settings();
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
// rasterization in morph_oracle/caliper_proto.py so the imea reference values above line up.
static void calculate_ellipse_caliper_values(std::vector<std::vector<double>>& fvals)
{
	Fsettings s = make_remaining2d_settings();

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

	EnclosingInscribingCircumscribingCircleFeature circle;
	circle.calculate(roi, s);
	circle.save_value(roi.fvals);

	fvals = roi.fvals;
}

// Build a filled circle (r=15) ROI and compute basic morphology + contour + the 3 circle diameters.
static void calculate_circle_shape_values(std::vector<std::vector<double>>& fvals)
{
	Fsettings s = make_remaining2d_settings();

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

// Assert a caliper stat agrees with imea within a relative tolerance (hull-vs-raster convention gap).








// The seven test functions that used to live here have been distributed to their taxonomy homes
// (registry target_test): erosion/caliper/chords/chord-angle/polygonality -> test_2d_morphology_regression.h,
// radial distribution -> test_2d_intensity_histogram_regression.h, zernike2d -> test_2d_zernike_regression.h.
// This header now carries only the shared fixture/oracle-data those files include.

// Fixtures only, no reference data and no assertions (SPEC 6.3.1). The six tables that lived here are
// now beside the assertions that read them; remaining2d_caliper_ref_val() went with them, since it
// searched an imea table and a regression table in one pass and let a _regression function resolve
// imea-vetted values without saying so.
