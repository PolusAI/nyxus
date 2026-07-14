#pragma once

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "../src/nyx/roi_cache.h"
#include "../src/nyx/features/2d_geomoments.h"
#include "../src/nyx/features/contour.h"
#include "test_main_nyxus.h"

struct GeomomentGoldenValue
{
	Nyxus::Feature2D feature;
	const char* name;
	double golden_value;
};

static Fsettings make_2d_geomoment_settings()
{
	Fsettings s;
	s.resize(static_cast<int>(NyxSetting::__COUNT__));
	s[static_cast<int>(NyxSetting::SOFTNAN)].rval = 0.0;
	s[static_cast<int>(NyxSetting::TINY)].rval = 0.0;
	s[static_cast<int>(NyxSetting::SINGLEROI)].bval = false;
	s[static_cast<int>(NyxSetting::GREYDEPTH)].ival = 256;
	s[static_cast<int>(NyxSetting::PIXELSIZEUM)].rval = 1.0;
	s[static_cast<int>(NyxSetting::XYRES)].rval = 1.0;
	s[static_cast<int>(NyxSetting::PIXELDISTANCE)].ival = 1;
	s[static_cast<int>(NyxSetting::USEGPU)].bval = false;
	s[static_cast<int>(NyxSetting::VERBOSLVL)].ival = 0;
	s[static_cast<int>(NyxSetting::IBSI)].bval = false;
	return s;
}

static double geomoment_fixture_intensity(int x, int y)
{
	return 10.0 + 3.0 * x + 5.0 * y + static_cast<double>((x * y) % 7);
}

static void load_geomoment_fixture(LR& roidata)
{
	constexpr int width = 48;
	constexpr int height = 40;

	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			const auto intensity = static_cast<PixIntens>(geomoment_fixture_intensity(x, y));
			if (roidata.aux_area == 0)
				init_label_record_3(roidata, x, y, intensity);
			else
				update_label_record_3(roidata, x, y, intensity);
		}
	}

	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			const auto intensity = static_cast<PixIntens>(geomoment_fixture_intensity(x, y));
			roidata.raw_pixels.push_back(Pixel2(x, y, intensity));
		}
	}

	roidata.make_nonanisotropic_aabb();
	roidata.aux_image_matrix.allocate(
		roidata.aabb.get_width(),
		roidata.aabb.get_height());
	roidata.aux_image_matrix.calculate_from_pixelcloud(roidata.raw_pixels, roidata.aabb);
}

static void calculate_2d_geomoment_feature_values(std::vector<std::vector<double>>& fvals)
{
	Fsettings s = make_2d_geomoment_settings();

	LR roidata(1401);
	load_geomoment_fixture(roidata);
	roidata.initialize_fvals();

	ContourFeature contour;
	contour.calculate(roidata, s);
	contour.save_value(roidata.fvals);

	Smoms2D_feature shape_moments;
	shape_moments.calculate(roidata, s);
	shape_moments.save_value(roidata.fvals);

	Imoms2D_feature intensity_moments;
	intensity_moments.calculate(roidata, s);
	intensity_moments.save_value(roidata.fvals);

	fvals = roidata.fvals;
}

static void assert_2d_geomoment_features(
	const std::vector<std::vector<double>>& fvals,
	const std::vector<GeomomentGoldenValue>& golden_values,
	const std::string& review_prefix)
{
	for (const auto& item : golden_values)
	{
		const double actual = fvals[static_cast<int>(item.feature)][0];
		const double scale = std::max({1.0, std::abs(item.golden_value), std::abs(actual)});
		const double tolerance = 1e-6 * scale;
		const std::string review_name = review_prefix + item.name;
		SCOPED_TRACE(review_name);
		ASSERT_TRUE(std::isfinite(actual)) << review_name << " returned a non-finite value";
		ASSERT_NEAR(actual, item.golden_value, tolerance) << review_name;
	}
}
