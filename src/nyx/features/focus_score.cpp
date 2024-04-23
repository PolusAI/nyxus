#include "focus_score.h"
#include "../parallel.h"

using namespace Nyxus;

FocusScoreFeature::FocusScoreFeature() : FeatureMethod("FocusScoreFeature") {
    provide_features({Feature2D::FOCUS_SCORE});
}

void FocusScoreFeature::calculate(LR& r) {

    // Get ahold of the ROI image matrix
    const ImageMatrix& Im0 = r.aux_image_matrix;

    auto focus_score = this->variance(this->laplacian(Im0));

    focus_score_ = focus_score; 

}

void FocusScoreFeature::parallel_process(std::vector<int>& roi_labels, std::unordered_map <int, LR>& roiData, int n_threads)
{
	size_t jobSize = roi_labels.size(),
		workPerThread = jobSize / n_threads;

	runParallel(FocusScoreFeature::parallel_process_1_batch, n_threads, workPerThread, jobSize, &roi_labels, &roiData);
}

void FocusScoreFeature::parallel_process_1_batch(size_t firstitem, size_t lastitem, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData)
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
		FocusScoreFeature f;
		f.calculate(r);
		f.save_value(r.fvals);
	}
}

bool FocusScoreFeature::required(const FeatureSet& fs) 
{ 
    return fs.isEnabled (Feature2D::FOCUS_SCORE); 
}

void FocusScoreFeature::reduce (size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData)
{
    for (auto i = start; i < end; i++)
    {
        int lab = (*ptrLabels)[i];
        LR& r = (*ptrLabelData)[lab];

        FocusScoreFeature fsf;

        fsf.calculate (r);

        fsf.save_value (r.fvals);
    }
}

void FocusScoreFeature::save_value(std::vector<std::vector<double>>& feature_vals) {
    
    feature_vals[(int)Feature2D::FOCUS_SCORE][0] = focus_score_;

}

std::vector<double> FocusScoreFeature::laplacian(const ImageMatrix& Im, 
                                                int ksize) {
    
    int n_image = Im.height;
    int m_image = Im.width;

    int m_kernel = 3;
    int n_kernel = 3;

    std::vector<int> kernel;

    if (ksize == 1) {
        kernel = { 0, 1, 0, 
                   1, -4, 1, 
                   0, 1, 0 };
    } else {
        kernel = { 2, 0, 2, 
                   0, -8, 0, 
                   2, 0, 2 };
    }

    readOnlyPixels image = Im.ReadablePixels();

    std::vector<double> out(m_image * n_image, 0);

    int xKSize = n_kernel; // number of columns
    int yKSize = m_kernel; // number of rows

    int kernelCenterX = xKSize / 2.;
    int kernelCenterY = yKSize / 2.;

    int ikFlip, jkFlip;
    int ii, jj;

    for(int i = 0; i < m_image; ++i){
        for(int j = 0; j < n_image; ++j){
            for(int ik = 0; ik < yKSize; ++ik){
                ikFlip = yKSize - 1 - ik;
                for(int jk = 0; jk < xKSize; ++jk){
                    jkFlip = xKSize - 1 - jk;

                    ii = i + (kernelCenterY - ikFlip);
                    jj = j + (kernelCenterX - jkFlip);

                    if(ii >= 0 && ii < m_image && jj >= 0 && jj < n_image &&
                       ikFlip >= 0 && jkFlip >=0 && ikFlip < m_kernel && jkFlip < n_kernel){
                        out[i* n_image + j] += image[ii * n_image + jj] * kernel[ikFlip * n_kernel + jkFlip];
                    }
                }
            }
        }
    }
    
    return out;
}

double FocusScoreFeature::mean(std::vector<double> image) {

    double sum = 0.;

    for (const auto& pix: image) {
        sum += std::abs(pix);
    }

    return sum / image.size();
}


double FocusScoreFeature::variance(std::vector<double> image) {
    double image_mean = this->mean(image);
    double sum_squared_diff = 0.0;

    for (const auto& pix: image) {
        sum_squared_diff += std::pow(std::abs(pix) - image_mean, 2);
    }

    return sum_squared_diff / (image.size());
}
