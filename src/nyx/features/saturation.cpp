#include "saturation.h"

#include <iostream>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <map>
#include <set>

#include "../parallel.h"

using namespace Nyxus;

SaturationFeature::SaturationFeature() : FeatureMethod("SaturationFeature") {
    provide_features(SaturationFeature::featureset);
}

void SaturationFeature::calculate(LR& r) {

    // Get ahold of the ROI image matrix
    const ImageMatrix& Im0 = r.aux_image_matrix;

    std::tie(min_saturation_, max_saturation_) = get_percent_max_pixels(Im0);

}

void SaturationFeature::parallel_process(std::vector<int>& roi_labels, std::unordered_map <int, LR>& roiData, int n_threads)
{
	size_t jobSize = roi_labels.size(),
		workPerThread = jobSize / n_threads;

	runParallel(SaturationFeature::parallel_process_1_batch, n_threads, workPerThread, jobSize, &roi_labels, &roiData);
}

void SaturationFeature::parallel_process_1_batch(size_t firstitem, size_t lastitem, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData)
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

		// Calculate the feature and save it in ROI's csv-friendly buffer 'fvals'
		SaturationFeature f;
		f.calculate(r);
		f.save_value(r.fvals);
	}
}

bool SaturationFeature::required(const FeatureSet& fs) 
{ 
    return fs.anyEnabled (SaturationFeature::featureset); 
}

void SaturationFeature::reduce (size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData)
{
    for (auto i = start; i < end; i++)
    {
        int lab = (*ptrLabels)[i];
        LR& r = (*ptrLabelData)[lab];

        SaturationFeature fsf;

        fsf.calculate (r);

        fsf.save_value (r.fvals);
    }
}

void SaturationFeature::osized_calculate(LR& r, ImageLoader& imloader) {

    // Skip calculation in case of noninformative data
    if (r.aux_max == r.aux_min) return;

    WriteImageMatrix_nontriv Im0 ("SaturationFeature-osized_calculate-Im0", r.label);
    Im0.allocate_from_cloud (r.raw_pixels_NT, r.aabb, false);

    std::tie(min_saturation_, max_saturation_) = get_percent_max_pixels_NT(Im0);

    save_value(r.fvals);
}

void SaturationFeature::save_value(std::vector<std::vector<double>>& feature_vals) {
    
    feature_vals[(int)FeatureIMQ::MAX_SATURATION][0] = max_saturation_;
    feature_vals[(int)FeatureIMQ::MIN_SATURATION][0] = min_saturation_;
}

std::tuple<double, double> SaturationFeature::get_percent_max_pixels(const ImageMatrix& Im) {

    readOnlyPixels image = Im.ReadablePixels();

    auto [min_pixel_ptr, max_pixel_ptr] = std::minmax_element(image.begin(), image.end());

    auto min_pixel = *min_pixel_ptr;
    auto max_pixel = *max_pixel_ptr;

    double max_pixel_count = 0;
    double min_pixel_count = 0;

    for (const auto& pix: image) {
        if (pix == max_pixel) {
            ++max_pixel_count;
        } else if (pix == min_pixel) {
            ++min_pixel_count;
        }
    }

    return std::make_tuple(min_pixel_count / image.size(), max_pixel_count / image.size());
}

std::tuple<double, double> SaturationFeature::get_percent_max_pixels_NT(WriteImageMatrix_nontriv& Im) {

    auto min_pixel = Im.get_at(0);
    auto max_pixel = Im.get_at(0);

    auto width = Im.get_width(),
         height = Im.get_height(); 

    for (int row=0; row < height; row++) {
        for (int col = 0; col < width; col++)
        {
            size_t idx = row * width + col;
            auto pix = Im.get_at(idx);

            if (pix > max_pixel) max_pixel = pix;
            if (pix < min_pixel) min_pixel = pix;
        }    
    }

    double max_pixel_count = 0;
    double min_pixel_count = 0;

    for (int row=0; row < height; row++) {
        for (int col = 0; col < width; col++)
        {
            size_t idx = row * width + col;
            auto pix = Im.get_at(idx);

            if (pix == max_pixel) ++max_pixel_count;
            if (pix == min_pixel) ++min_pixel_count;
        }    
    }

    return std::make_tuple(min_pixel_count / (width*height), max_pixel_count / (width*height));
    
}
