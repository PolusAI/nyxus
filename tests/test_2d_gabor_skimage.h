#pragma once
//#include "gtest/gest.h"
#include "../src/nyx/globals.h"

// Oracle test for the 2D GABOR feature: goldens vetted against scikit-image
// (tests/vetting/oracles/gen_gabor_skimage.py). SPEC 2 -- this file claims correctness.

void assert_2d_gabor_skimage(bool gpu=false);
