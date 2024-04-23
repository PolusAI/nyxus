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
    provide_features({Feature2D::POWER_SPECTRUM_SLOPE});
}

void PowerSpectrumFeature::calculate(LR& r) {

    // Get ahold of the ROI image matrix
    const ImageMatrix& Im0 = r.aux_image_matrix;

    slope_ = power_spectrum_slope(Im0); 
}

bool PowerSpectrumFeature::required(const FeatureSet& fs) 
{ 
    return fs.isEnabled (Feature2D::POWER_SPECTRUM_SLOPE); 
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
    
    feature_vals[(int)Feature2D::POWER_SPECTRUM_SLOPE][0] = slope_;

}

double PowerSpectrumFeature::power_spectrum_slope(const ImageMatrix& Im) {

    readOnlyPixels image = Im.ReadablePixels();

    int rows = Im.height;
    int cols = Im.width;

    std::vector<int> radii;
    std::vector<double> magnitude, power;

    std::tie(radii, magnitude, power) = rps(image, rows, cols);

    std::vector<double> result;

    if (std::accumulate(magnitude.begin(), magnitude.end(), 0.) > 0 && 
        std::set<unsigned int>(image.begin(), image.end()).size() > 0) {

        std::vector<double> log_radii;
        for (int i = 0; i < magnitude.size(); ++i) {

            if (magnitude[i] < 0 || !std::isfinite(std::log(power[i]))) {
                radii.erase(radii.begin() + i);
                power.erase(power.begin() + i);
            } else {
                log_radii.push_back(std::log(radii[i]));
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

std::vector<double> PowerSpectrumFeature::invariant(std::vector<unsigned int> image) {

    
    double mean_value = mean(image);

    auto temp = image;

    for (auto& element: temp) {
        element = abs(element - mean_value);
    }

    // calculate median
    auto sorted_temp = temp;
    size_t n = sorted_temp.size() / 2;
    std::nth_element(sorted_temp.begin(), sorted_temp.begin()+n, sorted_temp.end()); // sort to middle element to get median
    double median = sorted_temp[n];

    std::vector<double> out (image.size(), 0.);

    for (int i = 0; i < image.size(); ++i) {
        out[i] = (double)image[i] / median;
    }    

    return out;
}

std::tuple<std::vector<int>, std::vector<double>, std::vector<double>>  PowerSpectrumFeature::rps(std::vector<unsigned int> image, int rows, int cols) {

    double max_width = std::min(rows, cols) / 8.;

    if (std::floor(max_width) < 3) {
        return std::make_tuple(std::vector<int> {2}, std::vector<double> {0.}, std::vector<double> {0.});
    }

    auto rows_arrange = arrange(0, rows, 1); 
    auto cols_arrange = arrange(0, cols, 1); 

    std::vector<std::vector<int>> added_vecs (rows, std::vector<int>(cols));

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            added_vecs[i][j] = std::pow(rows_arrange[i], 2) + std::pow(cols_arrange[j], 2);
        }
    }

    added_vecs = minimum(added_vecs, flipud(added_vecs));
    added_vecs = minimum(added_vecs, fliplr(added_vecs));

    int ptp = (*std::max_element(image.begin(), image.end())) - (*std::min_element(image.begin(), image.end()));
    
    std::vector<double> image_invariant = (ptp > 0) ? invariant(image) : std::vector<double>(image.begin(), image.end());

    double invariant_mean = mean(image_invariant);

    for(auto& element: image_invariant) {
        element -= invariant_mean;
    }


    std::vector<std::complex<double>> vec;

    for (const auto& num: image_invariant) {

        vec.push_back(std::complex<double>(num, 0));
    }

    std::vector<std::complex<double>> after_fft = fft2d(vec, dj::fft_dir::DIR_FWD);

    std::vector<double> mag;
    for (auto& num: after_fft) {
        mag.push_back(std::abs(num));
    }

    std::vector<int> radii;
    std::vector<double> power;
    for(auto& num: mag) {
        radii.emplace_back(std::floor(std::sqrt(num)) + 1);
        power.emplace_back(num * num);
    }

    std::vector<int> linear_labels = arrange(2, std::floor(max_width));

    auto mag_sum = nd_sum(mag, radii, linear_labels);
    auto power_sum = nd_sum(power, radii, linear_labels);

    return std::make_tuple(linear_labels, mag_sum, power_sum);
}

