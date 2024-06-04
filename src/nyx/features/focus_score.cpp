#include "focus_score.h"
#include "../parallel.h"

using namespace Nyxus;

int FocusScoreFeature::kernel[9] = { 0, 1, 0, 
                                     1, -4, 1, 
                                     0, 1, 0 };

FocusScoreFeature::FocusScoreFeature() : FeatureMethod("FocusScoreFeature") {
    provide_features(FocusScoreFeature::featureset);
}

void FocusScoreFeature::calculate(LR& r) {

    // Get ahold of the ROI image matrix
    const ImageMatrix& Im0 = r.aux_image_matrix;

    std::vector<double> laplacian_vec(Im0.height * Im0.width, 0);

    this->laplacian(Im0.ReadablePixels(), laplacian_vec, Im0.height, Im0.width);

    focus_score_ = this->variance(laplacian_vec);

    local_focus_score_ = this->get_local_focus_score(Im0.ReadablePixels(), Im0.height, Im0.width);

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
    return fs.anyEnabled (FocusScoreFeature::featureset); 
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

void osized_calculate(LR& r, ImageLoader& imloader) {

    // Skip calculation in case of noninformative data
    if (r.aux_max == r.aux_min) return;

    WriteImageMatrix_nontriv Im0 ("FocusScoreFeature-osized_calculate-Im0", r.label);
    Im0.allocate_from_cloud (r.raw_pixels_NT, r.aabb, false);

}

void FocusScoreFeature::save_value(std::vector<std::vector<double>>& feature_vals) {
    
    feature_vals[(int)FeatureIMQ::FOCUS_SCORE][0] = focus_score_;
    feature_vals[(int)FeatureIMQ::LOCAL_FOCUS_SCORE][0] = local_focus_score_;

}

double FocusScoreFeature::get_local_focus_score_NT(WriteImageMatrix_nontriv& Im, int ksize) {

    int n = 3; // size of kernel nxn

    if (ksize != 1) {

        kernel[0] = 2; 
        kernel[1] = 0;
        kernel[2] = 2;

        kernel[3] = 0;
        kernel[4] = -8;
        kernel[5] = 0;

        kernel[6] = 2;
        kernel[7] = 0;
        kernel[8] = 2;
    }

    auto width = Im.get_width(),
        height = Im.get_height(); 
    auto xy0 = (int)ceil(double(n) / 2.);

    // Window (N-fold kernel size)
    int winX = n * 10, 
        winY = n * 10;
    std::vector<unsigned int> W (winY * winX);

    // Convolution result buffer
    std::vector<double> conv_buffer ((winY + n - 1) * (winX + n - 1) * 2);

    // Iterate the image window by window
    int n_winHor = width / winX, //ceil (float(width) / float(win)), 
        n_winVert = height / winY; //ceil (float(height) / float(win));

    
    // ROI smaller than kernel?
    if (n_winVert == 0 || n_winHor == 0)
    {
        // Fill the window with data
        for (int row=0; row < height; row++) {
            for (int col = 0; col < width; col++)
            {
                size_t idx = row * width + col;
                W[idx] = Im.get_at(idx);
            }    
        }

        // Convolve
        laplacian (W, conv_buffer, width, height, ksize);
        
        return variance(conv_buffer);
    }

    double total_abs_sum = 0;

    std::vector<std::tuple<double, double, int>> tile_variance; // vector of tuples containing 0: abs sum of tile, 1: variance of tile, 2: buffer size
    // ROI larger than kernel
    for (int winVert = 0; winVert < n_winVert; winVert++)
    {
        for (int winHor = 0; winHor < n_winHor; winHor++)
        {
            int count = 0;
            // Fill the window with data
            for (int row=0; row<winY; row++)
                for (int col = 0; col < winX; col++)
                {
                    size_t imIdx = winVert * n_winHor * winY * winX // skip whole windows above
                        + row * n_winHor * winX  // skip 'row' image-wide horizontal lines
                        + winHor * n_winHor * winX   // skip winHor-wide line
                        + col;
                    W[row * winX + col] = Im.get_at(imIdx);
                }

            // Convolve
            laplacian (W, conv_buffer, winY, winX, ksize);

            double abs_sum = std::transform_reduce(conv_buffer.begin(), conv_buffer.end(), 0.0, std::plus<>(), [](double val) { 
                return val; 
            });

            total_abs_sum += abs_sum;

            tile_variance.emplace_back(std::make_tuple(abs_sum, variance(conv_buffer), conv_buffer.size()));
        }
    }

    double abs_mean = total_abs_sum / (width * height); // get median of absolute sum

    double total_variance = 0;
    for (auto& sum_pair: tile_variance) {
        
        auto mu_i = std::get<0>(sum_pair);
        auto s_i = std::get<1>(sum_pair);
        auto n_i = std::get<2>(sum_pair);

        total_variance += ((n_i - 1) * s_i) + (n_i * (mu_i - total_abs_sum) * (mu_i - abs_mean));
    }
    
    return total_variance / (width * height);

}

double FocusScoreFeature::get_local_focus_score(const std::vector<PixIntens>& image, int height, int width, int ksize, int scale) {

    double local_focus_score = 0;

    int M = height / scale;
    int N = width / scale;

    std::vector<double> laplacian_vec(M*N);
    std::vector<PixIntens> image_tile(M*N);
    for (int y = 0; y < height - M; y += M) {
        for (int x = 0; x < width - N; x += N) {

            // Extract image tile
            for (int i = y; i < y + M; i++) {
                for (int j = x; j < x + N; j++) {
                    image_tile[(i-y) * N + (j-x)] = image[i * width + j];
                }
            }
            
            std::fill(laplacian_vec.begin(), laplacian_vec.end(), 0.);
            laplacian(image_tile, laplacian_vec, M, N, ksize);

            // calculate focus score for tile
            local_focus_score += variance(laplacian_vec);
        }

    }

    return local_focus_score / (scale * scale); // average scores
}

void FocusScoreFeature::laplacian(const std::vector<PixIntens>& image, std::vector<double>& out, int m_image, int n_image, int ksize) {

    int m_kernel = 3;
    int n_kernel = 3;

    if (ksize != 1) {
        kernel[0] = 2; 
        kernel[1] = 0;
        kernel[2] = 2;

        kernel[3] = 0;
        kernel[4] = -8;
        kernel[5] = 0;

        kernel[6] = 2;
        kernel[7] = 0;
        kernel[8] = 2;
    }

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
}

double FocusScoreFeature::variance(const std::vector<double>& image) {

    double abs_image_mean = std::transform_reduce(image.begin(), image.end(), 0.0, std::plus<>(), [](double val) { 
        return std::abs(val); 
    }) / image.size();

    return std::transform_reduce(image.begin(), image.end(), 0.0, std::plus<>(), [abs_image_mean](double pix) {
        return std::pow(std::abs(pix) - abs_image_mean, 2);
    }) / image.size();
}
