#pragma once

// Mechanics of the 3D NGLDM family: what a setting does to the grey levels the matrix is built on,
// rather than the values against a reference. Claims no oracle (SPEC 2).

// Nothing of its own: gtest, <string> and the phantom run arrive with the common header.
#include "test_3d_ngldm_common.h"

// A coarse grey depth of 0 bins nothing, so NGLDM reads the raw levels -- the same grey levels IBSI
// mode reads -- and every feature equals the IBSI run bit for bit, on the segmented phantom's raw
// levels. A 0 that collapsed them to one level would move every grey-level feature.
void test_3d_ngldm_zero_greydepth_equals_ibsi_mechanics()
{
	for (const auto& [name, fc] : Nyxus::UserFacing_3D_featureNames)
	{
		if (name.rfind ("3NGLDM_", 0) != 0)
			continue;
		SCOPED_TRACE (name);
		double none = 0.0, ibsi = 0.0;
		ASSERT_NO_FATAL_FAILURE (calculate_3d_ngldm_feature_value (name, fc, none, false, 0));
		ASSERT_NO_FATAL_FAILURE (calculate_3d_ngldm_feature_value (name, fc, ibsi, true, 64));
		ASSERT_EQ (none, ibsi);
	}
}
