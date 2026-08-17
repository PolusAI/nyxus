#pragma once

// Oracle tests for the 2D GABOR feature: goldens vetted against scikit-image
// (tests/vetting/oracles/gen_gabor_skimage.py). SPEC 2 -- this file claims correctness.
//
// GABOR is asserted at two config points because Nyxus carries two different default
// (frequency, angle) sets -- the one compiled into GaborFeature::f0_theta_pairs and the one the
// option parser builds from the documented defaults. See tests/vetting/config_recipes.md
// (gabor.cpp_static_defaults, gabor.documented_defaults) and
// tests/vetting/audit/gabor_2d_skimage_vetting_report.md.

void assert_2d_gabor_skimage (bool gpu = false);
void assert_2d_gabor_documented_defaults_skimage ();
