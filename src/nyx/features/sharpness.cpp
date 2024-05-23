#include <vector>
#include <iostream>
#include <numeric>

#include "sharpness.h"
#include "../parallel.h"
#include "../helpers/helpers.h"

using namespace Nyxus;

SharpnessFeature::SharpnessFeature() : FeatureMethod("SharpnessFeature") {
    provide_features({FeatureIMQ::SHARPNESS});
}

void SharpnessFeature::calculate(LR& r) {

    // Get ahold of the ROI image matrix
    const ImageMatrix& Im0 = r.aux_image_matrix;
    sharpness_ = sharpness(Im0);
}

void SharpnessFeature::parallel_process(std::vector<int>& roi_labels, std::unordered_map <int, LR>& roiData, int n_threads)
{
	size_t jobSize = roi_labels.size(),
		workPerThread = jobSize / n_threads;

	runParallel(SharpnessFeature::parallel_process_1_batch, n_threads, workPerThread, jobSize, &roi_labels, &roiData);
}

void SharpnessFeature::parallel_process_1_batch(size_t firstitem, size_t lastitem, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData)
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
		SharpnessFeature f;
		f.calculate(r);
		f.save_value(r.fvals);
	}
}

bool SharpnessFeature::required(const FeatureSet& fs) 
{ 
    return fs.isEnabled (FeatureIMQ::SHARPNESS); 
}

void SharpnessFeature::reduce (size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData)
{
    for (auto i = start; i < end; i++)
    {
        int lab = (*ptrLabels)[i];
        LR& r = (*ptrLabelData)[lab];

        SharpnessFeature sf;

        sf.calculate (r);

        sf.save_value (r.fvals);
    }
}

void SharpnessFeature::save_value(std::vector<std::vector<double>>& feature_vals) {
    feature_vals[(int)FeatureIMQ::SHARPNESS][0] = sharpness_;
}

 void SharpnessFeature::remove_padding(std::vector<double>& img, int img_row, int img_col, int row_padding, int col_padding) {

    int new_col_size = img_col - 2*col_padding;

    for (int i = 0; i < img_row - 2*row_padding; ++i) {
        for (int j = 0; j < new_col_size; ++j) {
            img[i * new_col_size + j] = img[(i+row_padding) * img_col + (j+col_padding)];            
        } 
    }
    
    img.erase(img.begin() + img_row*img_col, img.end());    
}


void SharpnessFeature::pad_array(const std::vector<unsigned int>& array, std::vector<unsigned int>& out, int rows, int cols, int padRows, int padCols) {
    int paddedRows = rows + 2 * padRows;
    int paddedCols = cols + 2 * padCols;

    out.resize(paddedRows * paddedCols, 0);

    // Copy the original array to the center of the padded array
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            out[(i + padRows) * paddedCols + (j + padCols)] = array[i * cols + j];
        }
    }

    // Duplicate border elements horizontally
    for (int i = 0; i < rows; ++i) {
        for (int k = 1; k <= padCols; ++k) {
            out[(i + padRows) * paddedCols + k - 1] = array[i * cols]; // Left border
            out[(i + padRows) * paddedCols + paddedCols - k] = array[i * cols + cols - 1]; // Right border
        }
    }

    // Duplicate border elements vertically
    for (int i = 0; i < padRows; ++i){
        for (int j = 0; j < paddedCols; ++j) {
            out[(i*paddedCols) + j] = out[(padRows*paddedCols) + j];
            out[(i*paddedCols + (paddedCols*padRows + paddedCols*rows)) + j] = out[((paddedRows* paddedCols) - (padRows*paddedCols) - paddedCols) + j];
        }
    }
}

void SharpnessFeature::median_blur(const std::vector<unsigned int>& img, std::vector<double>& blurred_img_out, int rows, int cols, int ksize) {

    int pad = (ksize-1) / 2;

    int row_padding = rows;
    int col_padding = cols;
    
    int padded_rows = rows + 2 * row_padding;
    int padded_cols = cols + 2 * col_padding;

    std::vector<unsigned int> padded_img;
    pad_array(img, padded_img, rows, cols, row_padding, col_padding);

    blurred_img_out.resize(padded_rows * padded_cols, 0);

    for (int i = 0; i < padded_rows; ++i) {
        for (int j = 0; j < padded_cols; ++j) {
            std::vector<int> window;
            // Iterate over the neighborhood
            for (int x = -pad; x <= pad; ++x) {
                for (int y = -pad; y <= pad; ++y) {

                    int nx = i + x;
                    int ny = j + y;

                    // Check boundary conditions
                    if (nx >= 0 && nx < padded_rows && ny >= 0 && ny < padded_cols) {
                        window.push_back(padded_img[nx * padded_cols + ny]);
                    }
                }
            }

            // Sort the window values
            std::sort(window.begin(), window.end());

            // Get the median value
            int median_index = std::floor((double)window.size() / 2);

            blurred_img_out[i * padded_cols + j] = window[median_index];
        }
    }

    remove_padding(blurred_img_out, padded_rows, padded_cols, rows, cols);
}


std::vector<double> SharpnessFeature::convolve_1d(const std::vector<double>& img, std::vector<double>& kernel) {

    int input_size = img.size();
    int kernel_size = kernel.size();
    int pad_size = kernel_size / 2;

    std::vector<double> result(input_size);

    auto image = img;

    for (int i = 0; i < input_size; ++i) {
        double sum = 0.0;
        for (int j = 0; j < kernel_size; ++j) {
            int index = i - pad_size + j;
            if (index >= 0 && index < input_size) {
                sum += image[index] * kernel[j];
            }
        }
        result[i] = sum;
    }

    return result;
}

void SharpnessFeature::smooth_image(const std::vector<unsigned int>& image, std::vector<double>& smoothed, std::vector<double>& smoothed_transposed, int rows, int cols, double epsilon) {
    
    std::vector<double> kernel {-0.5, 0, 0.5};

    smoothed = std::vector<double>(image.begin(), image.end());
    smoothed_transposed.resize(image.size());

    // Transpose the vector
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            smoothed_transposed[j * rows + i] = image[i * cols + j];
        }
    }

    std::vector<double> conv, conv_transposed;
    for (int i = 0; i < rows; ++i) {

        conv = convolve_1d(std::vector<double>(smoothed.begin() + i*cols, smoothed.begin() + (i+1)*cols), kernel);
        
        for (int j = 0; j < conv.size(); ++j) {
            smoothed[i * cols + j] = conv[j];
        }
    }

    for (int i = 0; i < cols; ++i) {
        conv_transposed = convolve_1d(std::vector<double>(smoothed_transposed.begin() + i*rows, smoothed_transposed.begin() + (i+1)*rows), kernel);

        for (int j=0; j < conv_transposed.size(); ++j) {
            smoothed_transposed[i * rows + j] = conv_transposed[j];
        }
    }

    //  Transpose the vector
    for (int i = 0; i < cols; ++i) {
        for (int j = 0; j < rows; ++j) {
            std::swap(smoothed_transposed[j * cols + i], smoothed_transposed[i * rows + j]);
        }
    }

    auto max = *std::max_element(smoothed.begin(), smoothed.end());

    for (int i=0; i < smoothed.size(); ++i) {
        smoothed[i] = std::abs(smoothed[i])/(max+epsilon);
        smoothed_transposed[i] = std::abs(smoothed_transposed[i])/(max+epsilon);
    }
}

void SharpnessFeature::edges(const std::vector<unsigned int>& image, std::vector<double>& edge_x_out, std::vector<double>& edge_y_out, int rows, int cols, double edge_threshold) {

    std::vector<double> smooth_x, smooth_y;
    smooth_image(image, smooth_x, smooth_y, rows, cols);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {

            if (smooth_x[i * cols + j] > edge_threshold) {
                edge_x_out.emplace_back(1);
            } else {
                edge_x_out.emplace_back(0);
            }

            if (smooth_y[i * cols + j] > edge_threshold) {
                edge_y_out.emplace_back(1);
            } else {
                edge_y_out.emplace_back(0);
            }
        }
    }
}

void SharpnessFeature::dom(const std::vector<double>& Im, std::vector<double>& dom_x_out, std::vector<double>& dom_y_out, int rows, int cols) {

    // Calculate domx
    double median_shift_up, median_shift_down;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            median_shift_up = (i >= 2) ? Im[(i - 2) * cols + j] : 0;
            median_shift_down = (i < rows - 2) ? Im[(i + 2) * cols + j] : 0;
            dom_x_out[i * cols + j] = std::abs(median_shift_up - 2 * Im[i * cols + j] + median_shift_down);
        }
    }

    // Calculate domy
    double median_shift_left, median_shift_right;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            median_shift_left = (j >= 2) ? Im[i * cols + (j - 2)] : 0;
            median_shift_right = (j < cols - 2) ? Im[i * cols + (j + 2)] : 0;
            dom_y_out[i * cols + j] = std::abs(median_shift_left - 2 * Im[i * cols + j] + median_shift_right);
        }
    }
}

void SharpnessFeature::contrast(const std::vector<double>& Im, std::vector<double>&cx_out, std::vector<double>&cy_out, int rows, int cols) {

    cx_out.resize(rows*cols);
    cy_out.resize(rows*cols);

    double value;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            value = (i+1 < rows) ? Im[(i + 1) * cols + j] : 0;
            cx_out[i * cols + j] = std::abs(value - Im[i * cols + j]);

            value = (j+1 < cols) ? Im[i * cols + (j + 1)] : 0;
            cy_out[i * cols + j] = std::abs(value - Im[i * cols + j]);
        }
    }
}


double SharpnessFeature::sharpness(const ImageMatrix& Im, int width) {

    readOnlyPixels image = Im.ReadablePixels();

    int rows = Im.height;
    int cols = Im.width;

    std::vector<double> blurred;
    median_blur(image, blurred, rows, cols, 3);

    for (auto& pix: blurred) {
        pix /= 255.;
    }
    
    std::vector<double> edge_x, edge_y;

    edges(image, edge_x, edge_y, rows, cols);

    std::vector<double> dom_x(rows*cols), dom_y(rows*cols);
    dom_x.resize(rows*cols);
    dom_y.resize(rows*cols);

    dom(blurred, dom_x, dom_y, rows, cols);

    std::vector<double> cx, cy;
    contrast(blurred, cx, cy, rows, cols);

    for (int i = 0; i < rows*cols; ++i) {
        cx[i] *= edge_x[i];
        cy[i] *= edge_y[i];
    }

    std::vector<double> sx(rows*cols, 0), sy(rows*cols, 0);
    std::vector<double> num(cols, 0.), dn(cols, 0.);
    for (int i = width; i < rows-width; ++i) {
        
        std::fill(num.begin(), num.end(), 0.);
        std::fill(dn.begin(), dn.end(), 0.);

        // Calculate num and dn
        for (int j = -width; j < width; ++j) {
            for (int k = 0; k < cols; ++k) {
                num[k] += std::abs(dom_x[(i + j) * cols +k]);
                dn[k] += cx[(i + j) * cols + k];
            }
        }

        // Calculate Sx
        for (int k = 0; k < cols-width; ++k) {
            sx[i * cols + k] = (dn[k] > 1e-3) ? (num[k] / dn[k]) : 0;
        }

        std::fill(num.begin(), num.end(), 0.);
        std::fill(dn.begin(), dn.end(), 0.);

        // Calculate num and dn
        for (int j = -width; j < width; ++j) {
            for (int k = 0; k < cols; ++k) {
                num[k] += std::abs(dom_y[(i + j) * cols +k]);
                dn[k] += cy[(i + j) * cols + k];
            }
        }

        // Calculate Sx
        for (int k = 0; k < cols-width; ++k) {
            sy[i * cols + k] = (dn[k] > 1e-3) ? (num[k] / dn[k]) : 0;
        }
    }

    auto n_sharpx = std::accumulate(sx.begin(), sx.end(), 0.);
    auto n_sharpy = std::accumulate(sy.begin(), sy.end(), 0.);

    auto n_edgex = std::accumulate(edge_x.begin(), edge_x.end(), 0.);
    auto n_edgey = std::accumulate(edge_y.begin(), edge_y.end(), 0.);

    double rx = n_sharpx / (n_edgex + EPSILON);
    double ry = n_sharpy / (n_edgey + EPSILON);

    return std::sqrt((rx * rx) + (ry * ry));
}