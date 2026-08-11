#include <vector>

// GABOR ground truth for the 4 DSB2018 ROIs (test_dsb2018_data.h), at
//      'gabor_f0': 0.1
//      'gabor_freqs' : [4.0, 16.0, 32.0, 64.0]
//      'gabor_thetas' : [0.0, 45.0, 90.0, 135.0]
//      'gabor_gamma' : 0.1
//      'gabor_kersize' : 16
//      'gabor_sig2lam' : 0.8
//      'gabor_thold' : 0.025
//
// These values are VETTED against scikit-image 0.26.0 by an independent reimplementation of
// the Gabor pipeline (tests/vetting/oracles/gen_gabor_skimage.py). The oracle's filters come
// from skimage.filters.gabor_kernel itself -- frequency=f0/2pi, sigma_x=sig2lam*2pi/f0,
// sigma_y=sigma_x/gamma, offset=0 -- cropped to the Nyxus n x n grid, L1-normalized,
// full-convolved, cropped and scored the same way. The one exception is the f0=0 baseline
// filter, which gabor_kernel cannot express (frequency=0); at f0=0 the envelope and carrier
// are both identically 1, so that filter is the flat window derived in closed form.
// The oracle reproduces every value below exactly: max |diff| = 0.000e+00 over all 16.
//
// Asserted at the SPEC 7 same-definition-oracle tier, rel 1e-3 (see test_2d_gabor_skimage.cc).
//
// Updated after the gabor.cpp response-truncation fix: the filter-response magnitudes are
// now kept real-valued (double) instead of being truncated to PixIntens (unsigned int),
// which had floored sub-integer responses to 0 (e.g. the old rows had spurious 0.0 at the
// higher frequencies where the true response is small but nonzero).
const static std::vector<std::vector<double>> gabor_truth = {
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
