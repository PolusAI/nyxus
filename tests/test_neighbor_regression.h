#pragma once

#include <gtest/gtest.h>

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "../src/nyx/roi_cache.h"
#include "../src/nyx/features/basic_morphology.h"
#include "../src/nyx/features/contour.h"
#include "../src/nyx/features/neighbors.h"
#include "test_data.h"
#include "test_main_nyxus.h"

static std::unordered_map<int, std::unordered_map<std::string, double>> unvetted_nyxus_regression_neighbor2d_distance_feature_golden_values_by_label{
	{1, {
		{"NUM_NEIGHBORS", 4.0},
		{"PERCENT_TOUCHING", 87.5},
		{"CLOSEST_NEIGHBOR1_DIST", 2.5},
		{"CLOSEST_NEIGHBOR2_DIST", 2.54950975679639},
	}},
	{2, {
		{"NUM_NEIGHBORS", 1.0},
		{"PERCENT_TOUCHING", 66.6666666666667},   // FIX #13 (neighbors.cpp): deduped adjacency touching pixels / contour length (was 33.33)
		{"CLOSEST_NEIGHBOR1_DIST", 2.54950975679639},
		{"CLOSEST_NEIGHBOR2_DIST", 0.0},
	}},
	{3, {
		{"NUM_NEIGHBORS", 1.0},
		{"PERCENT_TOUCHING", 66.6666666666667},
		{"CLOSEST_NEIGHBOR1_DIST", 2.54950975679639},
		{"CLOSEST_NEIGHBOR2_DIST", 0.0},
	}},
	{4, {
		{"NUM_NEIGHBORS", 1.0},
		{"PERCENT_TOUCHING", 50.0},
		{"CLOSEST_NEIGHBOR1_DIST", 2.5},
		{"CLOSEST_NEIGHBOR2_DIST", 0.0},
	}},
	{5, {
		{"NUM_NEIGHBORS", 1.0},
		{"PERCENT_TOUCHING", 33.3333333333333},
		{"CLOSEST_NEIGHBOR1_DIST", 2.54950975679639},
		{"CLOSEST_NEIGHBOR2_DIST", 0.0},
	}}
};

static std::unordered_map<int, std::unordered_map<std::string, double>> unvetted_nyxus_regression_neighbor2d_angle_feature_golden_values_by_label{
	{1, {
		{"CLOSEST_NEIGHBOR1_ANG", 0.0},
		{"CLOSEST_NEIGHBOR2_ANG", 191.30993247402},
		{"ANG_BW_NEIGHBORS_MEAN", 132.172516881495},
		{"ANG_BW_NEIGHBORS_STDDEV", 115.230018010206},
		{"ANG_BW_NEIGHBORS_MODE", 0.0},
	}},
	{2, {
		{"CLOSEST_NEIGHBOR1_ANG", 11.3099324740202},
		{"CLOSEST_NEIGHBOR2_ANG", 0.0},
		{"ANG_BW_NEIGHBORS_MEAN", 11.3099324740202},
		{"ANG_BW_NEIGHBORS_STDDEV", 0.0},
		{"ANG_BW_NEIGHBORS_MODE", 11.0},
	}},
	{3, {
		{"CLOSEST_NEIGHBOR1_ANG", 78.6900675259798},
		{"CLOSEST_NEIGHBOR2_ANG", 0.0},
		{"ANG_BW_NEIGHBORS_MEAN", 78.6900675259798},
		{"ANG_BW_NEIGHBORS_STDDEV", 0.0},
		{"ANG_BW_NEIGHBORS_MODE", 79.0},
	}},
	{4, {
		{"CLOSEST_NEIGHBOR1_ANG", 180.0},
		{"CLOSEST_NEIGHBOR2_ANG", 0.0},
		{"ANG_BW_NEIGHBORS_MEAN", 180.0},
		{"ANG_BW_NEIGHBORS_STDDEV", 0.0},
		{"ANG_BW_NEIGHBORS_MODE", 180.0},
	}},
	{5, {
		{"CLOSEST_NEIGHBOR1_ANG", 258.69006752598},
		{"CLOSEST_NEIGHBOR2_ANG", 0.0},
		{"ANG_BW_NEIGHBORS_MEAN", 258.69006752598},
		{"ANG_BW_NEIGHBORS_STDDEV", 0.0},
		{"ANG_BW_NEIGHBORS_MODE", 259.0},
	}}
};

static Fsettings make_neighbors2d_settings()
{
	Fsettings s;
	s.resize(static_cast<int>(NyxSetting::__COUNT__));
	s[static_cast<int>(NyxSetting::SOFTNAN)].rval = 0.0;
	s[static_cast<int>(NyxSetting::TINY)].rval = 0.0;
	s[static_cast<int>(NyxSetting::SINGLEROI)].bval = false;
	s[static_cast<int>(NyxSetting::GREYDEPTH)].ival = 128;
	s[static_cast<int>(NyxSetting::PIXELSIZEUM)].rval = 1.0;
	s[static_cast<int>(NyxSetting::XYRES)].rval = 1.0;
	s[static_cast<int>(NyxSetting::PIXELDISTANCE)].ival = 1;
	s[static_cast<int>(NyxSetting::USEGPU)].bval = false;
	s[static_cast<int>(NyxSetting::VERBOSLVL)].ival = 0;
	s[static_cast<int>(NyxSetting::IBSI)].bval = false;
	return s;
}

static void calculate_neighborhood2d_feature_values(std::unordered_map<int, LR>& roiData)
{
	Fsettings s = make_neighbors2d_settings();
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
	}

	NeighborsFeature::manual_reduce(roiData, s, uniqueLabels);
}

static void assert_neighbor2d_feature(
	const std::unordered_map<int, LR>& roiData,
	int label,
	Nyxus::Feature2D feature,
	const std::string& feature_name,
	double frac_tolerance = 1000.0)
{
	ASSERT_TRUE(unvetted_nyxus_regression_neighbor2d_distance_feature_golden_values_by_label.count(label) > 0);
	ASSERT_TRUE(unvetted_nyxus_regression_neighbor2d_distance_feature_golden_values_by_label[label].count(feature_name) > 0);
	ASSERT_TRUE(agrees_gt(roiData.at(label).fvals[static_cast<int>(feature)][0], unvetted_nyxus_regression_neighbor2d_distance_feature_golden_values_by_label[label][feature_name], frac_tolerance));
}

static void assert_unvetted_no_direct_oracle_neighbor2d_feature(
	const std::unordered_map<int, LR>& roiData,
	int label,
	Nyxus::Feature2D feature,
	const std::string& feature_name,
	double frac_tolerance = 1000.0)
{
	SCOPED_TRACE(std::string("UNVETTED_NO_DIRECT_ORACLE__") + feature_name);
	ASSERT_TRUE(unvetted_nyxus_regression_neighbor2d_angle_feature_golden_values_by_label.count(label) > 0);
	ASSERT_TRUE(unvetted_nyxus_regression_neighbor2d_angle_feature_golden_values_by_label[label].count(feature_name) > 0);
	ASSERT_TRUE(agrees_gt(roiData.at(label).fvals[static_cast<int>(feature)][0], unvetted_nyxus_regression_neighbor2d_angle_feature_golden_values_by_label[label][feature_name], frac_tolerance));
}

void test_neighborhood2d_counts_and_touching()
{
	std::unordered_map<int, LR> roiData;
	calculate_neighborhood2d_feature_values(roiData);

	for (int label : {1, 2, 3, 4, 5})
	{
		assert_neighbor2d_feature(roiData, label, Nyxus::Feature2D::NUM_NEIGHBORS, "NUM_NEIGHBORS");
		assert_neighbor2d_feature(roiData, label, Nyxus::Feature2D::PERCENT_TOUCHING, "PERCENT_TOUCHING");
	}
}

void test_neighborhood2d_closest_neighbors()
{
	std::unordered_map<int, LR> roiData;
	calculate_neighborhood2d_feature_values(roiData);

	for (int label : {1, 2, 3, 4, 5})
	{
		assert_neighbor2d_feature(roiData, label, Nyxus::Feature2D::CLOSEST_NEIGHBOR1_DIST, "CLOSEST_NEIGHBOR1_DIST");
		assert_neighbor2d_feature(roiData, label, Nyxus::Feature2D::CLOSEST_NEIGHBOR2_DIST, "CLOSEST_NEIGHBOR2_DIST");
	}
}

void test_neighborhood2d_unvetted_no_direct_oracle_closest_neighbor_angles()
{
	std::unordered_map<int, LR> roiData;
	calculate_neighborhood2d_feature_values(roiData);

	for (int label : {1, 2, 3, 4, 5})
	{
		assert_unvetted_no_direct_oracle_neighbor2d_feature(roiData, label, Nyxus::Feature2D::CLOSEST_NEIGHBOR1_ANG, "CLOSEST_NEIGHBOR1_ANG");
		assert_unvetted_no_direct_oracle_neighbor2d_feature(roiData, label, Nyxus::Feature2D::CLOSEST_NEIGHBOR2_ANG, "CLOSEST_NEIGHBOR2_ANG");
	}
}

void test_neighborhood2d_unvetted_no_direct_oracle_neighbor_angle_stats()
{
	std::unordered_map<int, LR> roiData;
	calculate_neighborhood2d_feature_values(roiData);

	for (int label : {1, 2, 3, 4, 5})
	{
		assert_unvetted_no_direct_oracle_neighbor2d_feature(roiData, label, Nyxus::Feature2D::ANG_BW_NEIGHBORS_MEAN, "ANG_BW_NEIGHBORS_MEAN");
		assert_unvetted_no_direct_oracle_neighbor2d_feature(roiData, label, Nyxus::Feature2D::ANG_BW_NEIGHBORS_STDDEV, "ANG_BW_NEIGHBORS_STDDEV");
		assert_unvetted_no_direct_oracle_neighbor2d_feature(roiData, label, Nyxus::Feature2D::ANG_BW_NEIGHBORS_MODE, "ANG_BW_NEIGHBORS_MODE");
	}
}
