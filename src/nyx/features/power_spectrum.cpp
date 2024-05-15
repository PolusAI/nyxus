#include <iostream>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <map>
#include <set>

#include "power_spectrum.h"
#include "../helpers/helpers.h"
#include "../helpers/fft.h"
#include "../helpers/lstsq.h"
#include "../parallel.h"

using namespace Nyxus;

PowerSpectrumFeature::PowerSpectrumFeature() : FeatureMethod("PowerSpectrumFeature") {
    provide_features({FeatureIMQ::POWER_SPECTRUM_SLOPE});
}

void PowerSpectrumFeature::calculate(LR& r) {

    // Get ahold of the ROI image matrix
    const ImageMatrix& Im0 = r.aux_image_matrix;

    slope_ = power_spectrum_slope(Im0); 
}

bool PowerSpectrumFeature::required(const FeatureSet& fs) 
{ 
    return fs.isEnabled (FeatureIMQ::POWER_SPECTRUM_SLOPE); 
}

void PowerSpectrumFeature::reduce (size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData)
{
    for (auto i = start; i < end; i++)
    {
        int lab = (*ptrLabels)[i];
        LR& r = (*ptrLabelData)[lab];

        PowerSpectrumFeature fsf;

        fsf.calculate (r);

        fsf.save_value (r.fvals);
    }
}

void PowerSpectrumFeature::parallel_process(std::vector<int>& roi_labels, std::unordered_map <int, LR>& roiData, int n_threads)
{
	size_t jobSize = roi_labels.size(),
		workPerThread = jobSize / n_threads;

	runParallel(PowerSpectrumFeature::parallel_process_1_batch, n_threads, workPerThread, jobSize, &roi_labels, &roiData);
}

void PowerSpectrumFeature::parallel_process_1_batch(size_t firstitem, size_t lastitem, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData)
{

	// Calculate the feature for each batch ROI item 
	for (auto i = firstitem; i < lastitem; i++)
	{

		// Get ahold of ROI's label and cache
		int roiLabel = (*ptrLabels)[i];
		LR& r = (*ptrLabelData)[roiLabel];

		// Skip the ROI if its data is invalid to prevent nans and infs in the output
		if (r.has_bad_data())
			continue;

		// Calculate the feature and save it in ROI's csv-friendly b uffer 'fvals'
		PowerSpectrumFeature f;
		f.calculate(r);

		f.save_value(r.fvals);
	}
}

void PowerSpectrumFeature::save_value(std::vector<std::vector<double>>& feature_vals) {
    
    feature_vals[(int)FeatureIMQ::POWER_SPECTRUM_SLOPE][0] = slope_;

}

double PowerSpectrumFeature::power_spectrum_slope(const ImageMatrix& Im) {

    readOnlyPixels image = Im.ReadablePixels();

    int rows = Im.height;
    int cols = Im.width;

    std::vector<double> magnitude, power;

    double max_width = std::min(rows, cols) / 8.;

    std::vector<int> radii(std::floor(max_width));
    std::iota(radii.begin(), radii.end(), 2);

    rps(image, rows, cols, magnitude, power);
    int num_radii = power.size();

    std::vector<double> result;

    if (std::accumulate(magnitude.begin(), magnitude.end(), 0.) > 0 && 
        !image.empty()) {

        for (int i = 0; i < magnitude.size(); ++i) {

            if (magnitude[i] < 0 || !std::isfinite(std::log(power[i]))) {
                if (i < radii.size()) radii.erase(radii.begin() + i);
                if (i < power.size()) power.erase(power.begin() + i);
            }
        }

        if (radii.size() > 1) {
            std::vector<std::vector<double>> A (radii.size(), std::vector<double>(2, 1));

            for (int i = 0; i < radii.size(); ++i) {
                A[i][0] = std::log(radii[i]);
            }

            // apply log to power vector
            std::transform(power.begin(), power.end(), power.begin(), [](double x) { return std::log(x); });

            return lstsq(A, power)[0]; // get slope from least squares
        }
    }

    return 0;
}

void PowerSpectrumFeature::invariant(const std::vector<unsigned int>& image, std::vector<double>& out) {

    
    double mean_value = std::accumulate(image.begin(), image.end(), 0.0)/image.size();

    auto temp = image;

    for (auto& element: temp) {
        element = abs(element - mean_value);
    }

    // calculate median
    auto sorted_temp = temp;
    size_t n = sorted_temp.size() / 2;
    std::nth_element(sorted_temp.begin(), sorted_temp.begin()+n, sorted_temp.end()); // sort to middle element to get median
    double median = sorted_temp[n];

    for (int i = 0; i < image.size(); ++i) {
        out[i] = (double)image[i] / median;
    }    
}

void PowerSpectrumFeature::rps(const std::vector<unsigned int>& image, int rows, int cols, std::vector<double>& mag_sum, std::vector<double>& power_sum) {

    double max_width = std::min(rows, cols) / 8.;

    if (std::floor(max_width) < 3) {
        mag_sum = {0.};
        power_sum = {0.};
        return;
    }

    int ptp = (*std::max_element(image.begin(), image.end())) - (*std::min_element(image.begin(), image.end()));

    // calculate invariant or convert the pixels to double
    std::vector<double> image_invariant(image.size(), 0.);
    if (ptp > 0) {
        invariant(image, image_invariant);
    } else {
        image_invariant = std::vector<double>(image.begin(), image.end());
    }

    // get mean
    double invariant_mean = std::accumulate(image_invariant.begin(), image_invariant.end(), 0.0) / image_invariant.size();

    // subtract mean from each pixel
    std::transform(image_invariant.begin(), image_invariant.end(), image_invariant.begin(),
               [&](auto& pix) { return pix - invariant_mean; });

    // convert image to complex for fft
    std::vector<std::complex<double>> complex_image;
    for (const auto& num: image_invariant) {
        complex_image.push_back(std::complex<double>(num, 0));
    }

    auto rows_po2 = next_power_of_2(rows);
    auto cols_po2 = next_power_of_2(cols);
    
    auto po2 = std::max(rows_po2, cols_po2); // find max power of 2 to make square 

    power_of_2_padding(complex_image, rows, cols);

    std::vector<std::complex<double>> after_fft = fft2d(complex_image, dj::fft_dir::DIR_FWD);

    std::vector<int> radii;
    for (auto& num: complex_image) {
        radii.emplace_back(std::floor(std::sqrt(std::abs(num))) + 1);
    }

    mag_sum.resize(complex_image.size(), 0);
    power_sum.resize(complex_image.size(), 0);

    double fft_double_value;
    for (int i = 0; i < complex_image.size(); ++i) {
        auto label_index = radii[i]-2;
        if (label_index >=0 && label_index < complex_image.size()) {
            fft_double_value = std::abs(complex_image[i]);
            mag_sum[label_index] += fft_double_value;
            power_sum[label_index] += fft_double_value * fft_double_value;
        }
    }
}

void PowerSpectrumFeature::power_of_2_padding(std::vector<std::complex<double>>& complex_image, int rows, int cols) {
    unsigned int max_dim = std::max(rows, cols);
    unsigned int new_size = next_power_of_2(max_dim);
    unsigned int new_total_size = new_size * new_size;
    
    // Resize the vector to the new total size
    complex_image.resize(new_total_size, 0);

    // Start from the last element and move backwards to avoid overwriting
    for (int i = rows - 1; i >= 0; --i) {
        for (int j = cols - 1; j >= 0; --j) {
            complex_image[i * new_size + j] = complex_image[i * cols + j];
        }
    }

}