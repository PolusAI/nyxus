#define _USE_MATH_DEFINES   // for M_PI_4, M_PI_2
#include <cmath>            // M_PI_4, M_PI_2, exposed by the define above
#include <iostream>
#include <utility>
#include <gtest/gtest.h>
#include "../src/nyx/featureset.h"
#include "../src/nyx/feature_settings.h"
#include "../src/nyx/roi_cache.h"
#include "../src/nyx/features/gabor.h"
#include "test_2d_gabor_skimage.h"
#include "test_dsb2018_data.h"
#include "test_main_nyxus.h"
#include "test_ref_vals.h"   // ref_vals_map, and the <string> / <vector> it already includes

using namespace Nyxus;

// SPEC 7 same-definition-oracle tier: rel 1e-3, expressed as agrees_gt's fractional
// divisor. The measured Nyxus-vs-oracle agreement is exact (max |diff| 0.000e+00 over all
// 16 values at either config); the tolerance is not loose enough to hide a real disagreement
// because the feature is a ratio of pixel counts -- the smallest possible wrong answer differs
// by one counted pixel, i.e. by 1/baseline_count, which is 4.5e-3 .. 1.1e-2 on these four ROIs
// (gen_gabor_skimage.py part C), several times the tolerance.
static constexpr double GABOR_ORACLE_FRAC_TOL = 1000.;   // ground_truth/1000 == rel 1e-3

// GABOR goldens for the 4 DSB2018 ROIs (test_dsb2018_data.h), in ROI order, one inner vector of
// 4 filter scores per ROI.
//
// Provenance (SPEC 6.4): tool = scikit-image 0.26.0 (skimage.filters.gabor_kernel) + scipy 1.17.1
// (convolve2d) + numpy 2.4.6; generator = tests/vetting/oracles/gen_gabor_skimage.py; kernel
// mapping frequency=f0/2pi, sigma_x=sig2lam*2pi/f0, sigma_y=sigma_x/gamma, offset=0, cropped to
// the Nyxus 16x16 grid and L1-normalized. Shared settings for both tables: kersize n=16,
// gamma=0.1, sig2lam=0.8, baseline f0LP=0.1 at theta=pi/2, GRAYthr=0.025.
//
// The 16x16 crop and the zero-padded 'full' convolution are part of what these values mean: at
// gamma=0.1 the analytic kernel is far wider than the window (7.9% of its L1 mass survives the crop
// at the lowest frequency), so goldens regenerated through a tool's own filtering instead would
// differ by up to 1.0 -- audit/gabor_2d_skimage_vetting_report.md 4.1.

// Recipe gabor.cpp_static_defaults -- the (frequency, angle) set compiled into
// GaborFeature::f0_theta_pairs, i.e. what a run that sets no Gabor options computes:
// f0 = {0, pi/4, pi/2, 3pi/4}, theta = {4, 16, 32, 64} radians.
const static ref_vals_list<std::vector<double>> gabor_2d_skimage_cpp_defaults_ref_vals = {
    {   1.0112359550561798,
        0.9213483146067416,
        0.9662921348314607,
        0.6179775280898876 },

    {   1.0044843049327354,
        0.93273542600896864,
        0.11210762331838565,
        0.17488789237668162 },

    {   1.0053763440860215,
        0.978494623655914,
        0.37096774193548387,
        0.0 },

    {   1.0051546391752577,
        0.95876288659793818,
        0.4845360824742268,
        0.046391752577319589 },
};

// Recipe gabor.documented_defaults -- the (frequency, angle) set the option parser builds from the
// documented defaults gabor_freqs = [4, 16, 32, 64], gabor_thetas = [0, 45, 90, 135] degrees, i.e.
// what the Python API computes: f0 = {4, 16, 32, 64}, theta = {0, pi/4, pi/2, 3pi/4} radians.
const static ref_vals_list<std::vector<double>> gabor_2d_skimage_documented_defaults_ref_vals = {
    {   0.97752808988764045,
        1.0,
        1.0112359550561798,
        0.91011235955056182 },

    {   0.58744394618834082,
        0.99551569506726456,
        0.92825112107623315,
        0.83408071748878925 },

    {   0.84946236559139787,
        0.97311827956989245,
        0.95161290322580649,
        0.83870967741935487 },

    {   0.66494845360824739,
        0.94329896907216493,
        0.96391752577319589,
        0.88144329896907214 },
};

// The documented-defaults pair set, in the (frequency, angle-in-radians) order every consumer of
// f0_theta_pairs reads -- the order GaborOptions::parse_input builds when the user passes
// --gaborfreqs/--gabortheta, and the one the Python API always passes.
static const std::vector<std::pair<double, double>> documented_default_pairs = {
    {4.0,  0.0},
    {16.0, M_PI_4},
    {32.0, M_PI_2},
    {64.0, M_PI_4*3.0}
};

static void assert_2d_gabor_scores_skimage (const ref_vals_list<std::vector<double>>& ref_vals, bool gpu)
{
    for (size_t i = 0; i < dsb_data.size(); ++i)
    {
        LR roidata;
        roidata.initialize_fvals();

        // Feed data to the ROI
        load_test_roi_data(roidata, (int)i);

        // Anisotropy (none)
        roidata.make_nonanisotropic_aabb();

        // Calculate features
        GaborFeature f;
        Fsettings s;
        if(gpu)
        {
            #ifdef USE_GPU
                ASSERT_NO_THROW(f.calculate_gpu(roidata));   // single-filter GPU path
            #else
                std::cerr << "GPU build is not enabled. Defaulting to CPU version." << std::endl;
                ASSERT_NO_THROW(f.calculate(roidata, s));
            #endif
        }
        else
        {
            ASSERT_NO_THROW(f.calculate(roidata, s));
        }

        f.save_value (roidata.fvals);

        ASSERT_TRUE(ref_vals[i].size() == roidata.fvals[(int)Nyxus::Feature2D::GABOR].size())
            << "ROI " << i << ": " << roidata.fvals[(int)Nyxus::Feature2D::GABOR].size()
            << " filter scores against " << ref_vals[i].size() << " goldens";

        // The skimage-vetted goldens are compared against the in-RAM (CPU) path. The GPU path
        // (calculate_gpu, FFT-based convolution) diverges from the direct-convolution CPU path on
        // these ROIs, so the GPU branch above only verifies that the kernels compile and run.
        if (!gpu)
            for (size_t j = 0; j < ref_vals[i].size(); ++j)
            {
                // the Nyxus value first: agrees_gt takes its band from the second argument, so the
                // golden is what the tolerance is a fraction of
                const double actual = roidata.fvals[(int)Nyxus::Feature2D::GABOR][j];
                ASSERT_TRUE(agrees_gt(actual, ref_vals[i][j], GABOR_ORACLE_FRAC_TOL))
                    << "ROI " << i << " filter " << j << ": actual=" << actual
                    << " skimage=" << ref_vals[i][j];
            }
    }
}

void assert_2d_gabor_skimage (bool gpu)
{
    SCOPED_TRACE("GABOR_SKIMAGE_CPP_DEFAULTS");

    assert_2d_gabor_scores_skimage (gabor_2d_skimage_cpp_defaults_ref_vals, gpu);
}

void assert_2d_gabor_documented_defaults_skimage ()
{
    SCOPED_TRACE("GABOR_SKIMAGE_DOCUMENTED_DEFAULTS");

    // f0_theta_pairs is process-wide state, so the documented-defaults config is installed for the
    // duration of this test only and the compiled-in default is put back afterwards.
    auto saved_pairs = GaborFeature::f0_theta_pairs;
    GaborFeature::f0_theta_pairs = documented_default_pairs;

    assert_2d_gabor_scores_skimage (gabor_2d_skimage_documented_defaults_ref_vals, false);

    GaborFeature::f0_theta_pairs = saved_pairs;
}
