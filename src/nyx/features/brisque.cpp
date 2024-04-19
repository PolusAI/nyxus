#include "brisque.h"

BrisqueFeature::BrisqueFeature(const std::vector<unsigned int>& image, int rows, int cols) {
        
    int kernel_size = 7;
    float sigma = 7. / 6.;
    auto kernel = get_normalized_gaussian_kernel(kernel_size, sigma);

    image_ = image;
    rows_ = rows;
    cols_ = cols;

    Mscn mscn = Mscn(image_, rows_, cols_, kernel, kernel_size, kernel_size);

    coefficients_ = mscn.get_coefficients();
}

std::vector<double> BrisqueFeature::calculate() {

    std::vector<double> features;

    std::vector<double> temp;
    
    for (int i = 0;  i < num_mscn_types; ++i) {
        temp = calculate_features((MscnType)i);
        features.insert(features.end(), temp.begin(), temp.end());
    }

    return features;
}