#pragma once

#include <gtest/gtest.h>
#include <string>                     // std::string, std::to_string for the SCOPED_TRACE label

#include "test_2d_neighbor_common.h"  // fixture only: calculate_neighbor_feature_values

// Required properties of PERCENT_TOUCHING on the neighborhood2d_scene_labels fixture. SPEC 2
// invariant tier: these are bounds and closed forms the feature must obey by construction, not
// comparisons against a reference implementation, so they establish no vetting and carry no oracle
// token (SPEC 4.4: "does output obey a required property/bound/relation").
//
// They live here rather than in test_2d_neighbor_analytic.h because an _analytic suffix is an oracle
// claim: every coverage scanner in this tree attributes an oracle from the test function's name, so
// an invariant asserted under that name would credit PERCENT_TOUCHING with an oracle the registry
// correctly says it does not have.
//
// No reference table: an invariant compares against a property, not against pinned data, so there is
// nothing here for SPEC 6.3.1 to govern.

// The ratio is distinct touching contour pixels over contour length, so it lies in [0, 100]. The
// "distinct" is what makes the upper bound hold: a pixel adjacent to two neighbours is one touching
// pixel, not two.
void test_2d_neighbor_percent_touching_bounded_invariant()
{
	std::unordered_map<int, LR> roiData;
	calculate_neighbor_feature_values(roiData);

	// the feature is named on the assertion lines themselves, not hoisted into an index variable:
	// every coverage scanner in this tree reads feature coverage off assertion lines, so a hoisted
	// index makes the assertion invisible to them and the feature look untested
	for (int label : {1, 2, 3, 4, 5})
	{
		SCOPED_TRACE(std::string("INVARIANT__PERCENT_TOUCHING__L") + std::to_string(label));
		ASSERT_GE(roiData.at(label).fvals[static_cast<int>(Nyxus::Feature2D::PERCENT_TOUCHING)][0], 0.0);
		ASSERT_LE(roiData.at(label).fvals[static_cast<int>(Nyxus::Feature2D::PERCENT_TOUCHING)][0], 100.0);
	}
}

// Closed form: a ROI enclosed by neighbours on every side has every contour pixel 8-adjacent to some
// neighbour, so the ratio is exactly 1. Label 1 is the 3x3 centre block of this scene and is such a
// ROI. Exact equality is the whole point -- the value is 100 by construction, not by computation, so
// any tolerance at all would weaken it.
void test_2d_neighbor_percent_touching_enclosed_invariant()
{
	std::unordered_map<int, LR> roiData;
	calculate_neighbor_feature_values(roiData);

	ASSERT_DOUBLE_EQ(roiData.at(1).fvals[static_cast<int>(Nyxus::Feature2D::PERCENT_TOUCHING)][0], 100.0);
}
