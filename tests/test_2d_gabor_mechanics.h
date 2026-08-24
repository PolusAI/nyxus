#pragma once

#include <gtest/gtest.h>

#include "../src/nyx/features/gabor.h"
#include "test_main_nyxus.h"   // load_test_roi_data, and the dsb_data / roi_cache.h / featureset.h it includes

// Plumbing guard for the GPU Gabor path (SPEC 2 mechanics tier -- no oracle claim).
//
// GaborFeature::calculate_gpu convolves through an FFT rather than directly, and on the DSB2018
// fixture its filter responses diverge from the CPU path, so the skimage goldens in
// tests/test_2d_gabor_skimage.cc do not describe it. What can be asserted is that the CUDA
// kernels build, launch, and hand back one score per configured filter -- which is what this
// checks, on the same four ROIs, at the compiled-in GaborFeature::f0_theta_pairs default.
//
// The value comparison the GPU path does get is GPU-vs-CPU equality in tests/python/test_nyxus.py
// (test_gabor_gpu, skip_ci); that too claims no oracle.
inline void test_2d_gabor_gpu_runs_mechanics()
{
#ifdef USE_GPU
    const size_t n_filters = GaborFeature::f0_theta_pairs.size();

    for (size_t i = 0; i < dsb_data.size(); ++i)
    {
        LR roidata;
        roidata.initialize_fvals();
        load_test_roi_data(roidata, (int)i);
        roidata.make_nonanisotropic_aabb();

        GaborFeature f;
        ASSERT_NO_THROW(f.calculate_gpu(roidata));   // single-filter GPU path

        f.save_value(roidata.fvals);

        ASSERT_EQ(n_filters, roidata.fvals[(int)Nyxus::Feature2D::GABOR].size())
            << "ROI " << i << ": the GPU path returned "
            << roidata.fvals[(int)Nyxus::Feature2D::GABOR].size() << " scores for "
            << n_filters << " configured filters";
    }
#else
    GTEST_SKIP() << "built without USE_GPU";
#endif
}
