#include "brisque.h"
#include "../helpers/helpers.h"
#include "../parallel.h"

using namespace Nyxus;

BrisqueFeature::BrisqueFeature() : FeatureMethod("BrisqueFeature") {
    provide_features({Feature2D::BRISQUE});
}

void BrisqueFeature::calculate(LR& r) {

    const ImageMatrix& Im0 = r.aux_image_matrix;

    brisque(Im0);
}

void BrisqueFeature::brisque(const ImageMatrix& Im) {

    readOnlyPixels image = Im.ReadablePixels();
    int rows = Im.height;
    int cols = Im.width;

    int kernel_size = 7;
    float sigma = 7. / 6.;
    auto kernel = get_normalized_gaussian_kernel(kernel_size, sigma);

    Mscn mscn = Mscn(image, rows, cols, kernel, kernel_size, kernel_size);

    coefficients_ = mscn.get_coefficients();

    std::vector<double> temp;
    

    for (const auto& mscn_type: mscn_types) {
        temp = calculate_features(mscn_type);
        features_.insert(features_.end(), temp.begin(), temp.end());
    }

}

void BrisqueFeature::parallel_process(std::vector<int>& roi_labels, std::unordered_map <int, LR>& roiData, int n_threads)
{
	size_t jobSize = roi_labels.size(),
		workPerThread = jobSize / n_threads;

	runParallel(BrisqueFeature::parallel_process_1_batch, n_threads, workPerThread, jobSize, &roi_labels, &roiData);
}

void BrisqueFeature::parallel_process_1_batch(size_t firstitem, size_t lastitem, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData)
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
		BrisqueFeature f;
		f.calculate(r);
		f.save_value(r.fvals);
	}
}

bool BrisqueFeature::required(const FeatureSet& fs) 
{ 
    return fs.isEnabled (Feature2D::BRISQUE); 
}

void BrisqueFeature::reduce (size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData)
{
    for (auto i = start; i < end; i++)
    {
        int lab = (*ptrLabels)[i];
        LR& r = (*ptrLabelData)[lab];

        BrisqueFeature br;

        br.calculate (r);

        br.save_value (r.fvals);
    }
}

void BrisqueFeature::save_value(std::vector<std::vector<double>>& feature_vals) {
    feature_vals[(int)Feature2D::BRISQUE].resize(features_.size());

    for (int i = 0; i < features_.size(); ++i) {
        feature_vals[(int)Feature2D::BRISQUE][0] = features_[i];
    }
}

std::vector<double> BrisqueFeature::get_normalized_gaussian_kernel(int kernel_size, float sigma) {
    int center = std::floor(kernel_size / 2);
    std::vector<double> kernel(kernel_size * kernel_size);

    for (int y = 0; y < kernel_size; ++y) {
        for (int x = 0; x < kernel_size; ++x) {
            double x_dist = x - center;
            double y_dist = y - center;
            kernel[y * kernel_size + x] = 1 / (2 * M_PI * sigma * sigma) * exp(-(x_dist * x_dist + y_dist * y_dist) / (2 * sigma * sigma));
        }
    }

    return normalize(kernel);
}


std::vector<double> BrisqueFeature::calculate_features(MscnType type) {
    
    std::vector<double> x = coefficients_[type];
    auto agg = AsymmetricGeneralizedGaussian(x).fit();
    
    if (type == MscnType::mscn) {

        double var = (std::pow(agg.sigma_left(), 2) + std::pow(agg.sigma_right(), 2)) / 2.0;

        return {agg.alpha(), var};
    }

    return {agg.alpha(), agg.mean(), std::pow(agg.sigma_left(), 2), std::pow(agg.sigma_right(), 2)};

}