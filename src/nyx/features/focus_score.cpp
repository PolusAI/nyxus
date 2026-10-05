#include "focus_score.h"

using namespace Nyxus;

namespace
{
    // The two Laplacian stencils laplacian() chooses between, per call, from its ksize argument.
    // Both are constant, so no call can change the kernel a later call sees.
    constexpr int laplacian_kernel_ksize1[9] = { 0, 1, 0,
                                                 1, -4, 1,
                                                 0, 1, 0 };
    constexpr int laplacian_kernel_ksizeN[9] = { 2, 0, 2,
                                                 0, -8, 0,
                                                 2, 0, 2 };
}

FocusScoreFeature::FocusScoreFeature() : FeatureMethod("FocusScoreFeature") {
    provide_features(FocusScoreFeature::featureset);
}

void FocusScoreFeature::calculate (LR& r, const Fsettings& s) 
{
    // Get ahold of the ROI image matrix
    const ImageMatrix& Im0 = r.aux_image_matrix;
    const pixData& pixels = Im0.ReadablePixels();
    const size_t width = Im0.width;
    auto px = [&pixels, width] (int row, int col) { return pixels[row * width + col]; };

    focus_score_ = laplacian_variance (px, 0, 0, Im0.height, Im0.width, 1);

    local_focus_score_ = get_local_focus_score (px, Im0.height, Im0.width, STNGS_NAN(s));
}

void FocusScoreFeature::extract (LR& r, const Fsettings& s)
{
	FocusScoreFeature f;
	f.calculate (r, s);
	f.save_value (r.fvals);
}

void FocusScoreFeature::parallel_process_1_batch (size_t firstitem, size_t lastitem, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData, const Fsettings & s, const Dataset & _)
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
		extract (r, s);
	}
}

bool FocusScoreFeature::required(const FeatureSet& fs) 
{ 
    return fs.anyEnabled (FocusScoreFeature::featureset); 
}

void FocusScoreFeature::reduce (size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData, const Fsettings & s)
{
    for (auto i = start; i < end; i++)
    {
        int lab = (*ptrLabels)[i];
        LR& r = (*ptrLabelData)[lab];

        FocusScoreFeature f;
        f.calculate (r, s);
        f.save_value (r.fvals);
    }
}

void FocusScoreFeature::osized_calculate (LR& r, const Fsettings& s, ImageLoader& ldr)
{
    // The same image calculate() reads, disk-backed: the ROI's bounding box, 0 off the mask. A
    // constant ROI is scored as calculate() scores it, not skipped.
    WriteImageMatrix_nontriv Im0 ("FocusScoreFeature-osized_calculate-Im0", r.label);
    Im0.allocate_from_cloud (r.raw_pixels_NT, r.aabb, false);
    auto px = [&Im0] (int row, int col) { return (PixIntens) Im0.yx (row, col); };

    int h = (int) Im0.get_height(),
        w = (int) Im0.get_width();

    focus_score_ = laplacian_variance (px, 0, 0, h, w, 1);

    local_focus_score_ = get_local_focus_score (px, h, w, STNGS_NAN(s));

    save_value(r.fvals);
}

void FocusScoreFeature::save_value(std::vector<std::vector<double>>& feature_vals) {
    
    feature_vals[(int)FeatureIMQ::FOCUS_SCORE][0] = focus_score_;
    feature_vals[(int)FeatureIMQ::LOCAL_FOCUS_SCORE][0] = local_focus_score_;

}

void FocusScoreFeature::laplacian(const std::vector<PixIntens>& image, std::vector<double>& out, int m_image, int n_image, int ksize) {

    int m_kernel = 3;
    int n_kernel = 3;

    const int* kernel = ksize == 1 ? laplacian_kernel_ksize1 : laplacian_kernel_ksizeN;

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
                        // Cast the pixel to double before multiplying: image is PixIntens (unsigned),
                        // so the negative kernel weights (e.g. -4) were converted to a huge unsigned
                        // value, wrapping the Laplacian (focus score reached ~1e18 from the overflow).
                        out[i* n_image + j] += static_cast<double>(image[ii * n_image + jj]) * kernel[ikFlip * n_kernel + jkFlip];
                    }
                }
            }
        }
    }
}

