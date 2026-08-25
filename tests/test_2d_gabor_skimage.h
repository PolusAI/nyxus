#pragma once

// Oracle tests for the 2D GABOR feature: goldens vetted against scikit-image
// (tests/vetting/oracles/gen_gabor_skimage.py). SPEC 2 -- this file claims correctness.
//
// GABOR is asserted at two config points because Nyxus carries two different default
// (frequency, angle) sets -- the one compiled into GaborFeature::f0_theta_pairs and the one the
// option parser builds from the default frequency and angle lists. See
// tests/vetting/config_recipes.md (gabor.cpp_static_defaults, gabor.python_raw_defaults),
// tests/vetting/matrix/gabor.md and
// tests/vetting/audit/gabor_2d_skimage_vetting_report.md.
//
// Both assertions run the in-RAM CPU path. The GPU path carries no oracle claim and is guarded in
// tests/test_2d_gabor_mechanics.h.

void test_2d_gabor_cpp_static_defaults_skimage ();
void test_2d_gabor_python_raw_defaults_skimage ();
