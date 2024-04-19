#include <vector>
#include <iostream>
#include <cmath>
#include <numeric>
#include <unordered_map>
#include "brisque_gaussian.h"
#include "helpers/helpers.h"


class BrisqueFeature {

private:
    std::unordered_map<MscnType, std::vector<double>> coefficients_;

    std::vector<unsigned int> image_;
    int rows_;
    int cols_;

    std::vector<double> get_normalized_gaussian_kernel(int kernel_size, float sigma) {
        int center = std::floor(kernel_size / 2);
        std::vector<double> kernel(kernel_size * kernel_size);

        for (int y = 0; y < kernel_size; ++y) {
            for (int x = 0; x < kernel_size; ++x) {
                double x_dist = x - center;
                double y_dist = y - center;
                kernel[y * kernel_size + x] = 1 / (2 * M_PI * sigma * sigma) * exp(-(x_dist * x_dist + y_dist * y_dist) / (2 * sigma * sigma));
            }
        }

        return Nyxus::normalize(kernel);
    }


    std::vector<double> calculate_features(MscnType type) {
        
        std::vector<double> x = coefficients_[type];
        auto agg = AsymmetricGeneralizedGaussian(x).fit();
        
        if (type == MscnType::mscn) {

            double var = (std::pow(agg.sigma_left(), 2) + std::pow(agg.sigma_right(), 2)) / 2.0;

            return {agg.alpha(), var};
        }
    
        return {agg.alpha(), agg.mean(), std::pow(agg.sigma_left(), 2), std::pow(agg.sigma_right(), 2)};

    }

public:

    BrisqueFeature(const std::vector<unsigned int>& image, int rows, int cols);

    std::vector<double> calculate();
};
