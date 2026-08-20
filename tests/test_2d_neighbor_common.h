#pragma once

// Shared fixture for the 2D neighbour-graph tests: the settings recipe and the scene builder that
// turns neighborhood2d_scene_labels into a populated roiData map with basic-morphology, contour and
// neighbour features computed.
//
// Fixtures only, no reference data (SPEC 6.3.1). Every file in the family reaches the scene builder
// through this header, so no oracle file has to include another file's snapshot table to borrow
// scaffolding, and no reference table is ever in scope of an assertion that does not own it.

// No <gtest/gtest.h>: this header asserts nothing, it only builds ROIs. The files that include it
// bring gtest in themselves for their own assertions.
//
// Every include below is named for the symbol this file uses directly, rather than left to arrive
// transitively through test_main_nyxus.h: a transitive include compiles, but it makes this header
// depend on what an unrelated file happens to pull in.
#include <unordered_map>
#include <unordered_set>

#include "../src/nyx/feature_settings.h"        // Fsettings, NyxSetting
#include "../src/nyx/roi_cache.h"               // LR, init_label_record_3, update_label_record_3
#include "../src/nyx/features/basic_morphology.h"  // BasicMorphologyFeatures
#include "../src/nyx/features/contour.h"        // ContourFeature
#include "../src/nyx/features/image_matrix.h"   // ImageMatrix
#include "../src/nyx/features/neighbors.h"      // NeighborsFeature
#include "../src/nyx/features/pixel.h"          // Pixel2, PixIntens
#include "test_data.h"                          // neighborhood2d_scene_labels
#include "test_main_nyxus.h"                    // the Nyxus namespace the loaders live in

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

static void calculate_neighbor_feature_values(std::unordered_map<int, LR>& roiData)
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
