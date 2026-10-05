#pragma once

#include <algorithm>    // std::copy, std::fill in laplacian_variance()
#include <cmath>        // std::pow
#include <vector>
#include "../helpers/helpers.h"
#include "../roi_cache.h"
#include "image_matrix.h"
#include "../feature_method.h"
#include "../feature_settings.h"
#include "../environment.h"

/// @brief Extract face feature based on gabor filtering
class FocusScoreFeature: public FeatureMethod
{
public:

    const constexpr static std::initializer_list<Nyxus::FeatureIMQ> featureset = { Nyxus::FeatureIMQ::FOCUS_SCORE,  Nyxus::FeatureIMQ::LOCAL_FOCUS_SCORE};

    FocusScoreFeature();

    static bool required(const FeatureSet& fs);
   
    //=== Trivial ROIs ===
    void calculate (LR& r, const Fsettings& s);

    static void extract (LR& roi, const Fsettings& s);
    static void parallel_process_1_batch (size_t firstitem, size_t lastitem, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData, const Fsettings & s, const Dataset & ds);

    //=== Non-trivial ROIs ===
    void osized_add_online_pixel(size_t x, size_t y, uint32_t intensity) {}
    void osized_calculate (LR& r, const Fsettings& s, ImageLoader& ldr);

    // Result saver
    void save_value(std::vector<std::vector<double>>& feature_vals);

    static void reduce (size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData, const Fsettings & fst);

    // Adds the 3x3 Laplacian of the m_image x n_image image to out, with zero padding at the border.
    // ksize == 1 selects {{0,1,0},{1,-4,1},{0,1,0}}, any other value {{2,0,2},{0,-8,0},{2,0,2}}.
    static void laplacian(const std::vector<PixIntens>& image, std::vector<double>& out, int m_image, int n_image, int ksize=1);

private:

    // Result cache. calculate() and osized_calculate() assign both on every call.
    double focus_score_ = 0;
    double local_focus_score_ = 0;

    // Both scores are computed by the two templates below, which read the ROI image through
    // px(row, col) and nothing else. calculate() hands them the in-RAM image matrix and
    // osized_calculate() the disk-backed one, so the two paths share every arithmetic step.

    // Population variance of the 3x3 Laplacian of the h x w rectangle whose top-left pixel is
    // (y0, x0), zero padded at the rectangle's border
    template <class PixelAt>
    static double laplacian_variance (PixelAt& px, int y0, int x0, int h, int w, int ksize);

    // The mean of laplacian_variance() over a scale x scale grid of non-overlapping
    // (h/scale) x (w/scale) tiles. When a side is not a multiple of scale, its last h % scale rows
    // or w % scale columns belong to no tile. A side shorter than scale fits no tile, so the score
    // is undefined and 'undefined' is returned.
    template <class PixelAt>
    static double get_local_focus_score (PixelAt& px, int h, int w, double undefined, int ksize=1, int scale=2);
};

template <class PixelAt>
double FocusScoreFeature::laplacian_variance (PixelAt& px, int y0, int x0, int h, int w, int ksize)
{
    // The rectangle is filtered three rows at a time, so an out-of-core image is never held whole.
    // strip holds rectangle rows r-1, r and r+1, with 0 where a row falls outside the rectangle, so
    // the middle row of its Laplacian is the Laplacian of rectangle row r.
    const size_t W = w;
    std::vector<PixIntens> strip (3 * W);
    std::vector<double> lap (3 * W);

    auto load_row = [&] (size_t slot, int row)
    {
        for (int c = 0; c < w; c++)
            strip[slot * W + c] = (row >= 0 && row < h) ? (PixIntens) px (y0 + row, x0 + c) : 0;
    };

    // Two passes, the mean and then the squared deviations from it, each summed in row-major order
    const double n = double(h) * double(w);
    double mean = 0, sum = 0;
    for (int pass = 0; pass < 2; pass++)
    {
        sum = 0;
        load_row (0, -1);
        load_row (1, 0);
        load_row (2, 1);
        for (int r = 0; r < h; r++)
        {
            if (r > 0)
            {
                std::copy (strip.begin() + W, strip.end(), strip.begin());
                load_row (2, r + 1);
            }
            std::fill (lap.begin(), lap.end(), 0.);
            laplacian (strip, lap, 3, w, ksize);
            for (size_t c = 0; c < W; c++)
                sum += pass == 0 ? lap[W + c] : std::pow (lap[W + c] - mean, 2);
        }
        if (pass == 0)
            mean = sum / n;
    }

    // The population variance of the signed Laplacian -- the Pech-Pacheco focus measure, matching
    // cv2.Laplacian(img, CV_64F).var(). The absolute value is deliberately not taken: Var(|X|) is a
    // different, smaller statistic whenever mean(x) != 0, and zero padding at the border keeps
    // mean(x) away from 0.
    return sum / n;
}

template <class PixelAt>
double FocusScoreFeature::get_local_focus_score (PixelAt& px, int h, int w, double undefined, int ksize, int scale)
{
    int M = h / scale,
        N = w / scale;

    if (M == 0 || N == 0)
        return undefined;

    double total = 0;
    for (int ty = 0; ty < scale; ty++)
        for (int tx = 0; tx < scale; tx++)
            total += laplacian_variance (px, ty * M, tx * N, M, N, ksize);

    return total / (scale * scale);     // mean over the scale^2 tiles
}

