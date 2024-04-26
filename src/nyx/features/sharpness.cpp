#include <vector>
#include <iostream>
#include <numeric>

#include "sharpness.h"
#include "../parallel.h"
#include "../helpers/helpers.h"

using namespace Nyxus;

SharpnessFeature::SharpnessFeature() : FeatureMethod("SharpnessFeature") {
    provide_features({Feature2D::SHARPNESS});
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
    return fs.isEnabled (Feature2D::SHARPNESS); 
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
    feature_vals[(int)Feature2D::SHARPNESS].resize(1);
    feature_vals[(int)Feature2D::SHARPNESS][0] = sharpness_;

}

std::vector<double> SharpnessFeature::remove_padding(std::vector<double> img, int img_row, int img_col, int row_padding, int col_padding) {
    std::vector<double> out;

    for (int i = row_padding; i < img_row-row_padding; ++i) {
        for (int j = col_padding; j < img_col-col_padding; ++j) {
            out.push_back(img[i * img_col + j]);
        }
    }
    return out;
}


std::vector<unsigned int> SharpnessFeature::pad_array(const std::vector<unsigned int>& array, int rows, int cols, int padRows, int padCols) {
    int paddedRows = rows + 2 * padRows;
    int paddedCols = cols + 2 * padCols;

    std::vector<unsigned int> paddedArray(paddedRows * paddedCols, 0);

    // Copy the original array to the center of the padded array
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            paddedArray[(i + padRows) * paddedCols + (j + padCols)] = array[i * cols + j];
        }
    }

    // Duplicate border elements horizontally
    for (int i = 0; i < rows; ++i) {
        for (int k = 1; k <= padCols; ++k) {
            paddedArray[(i + padRows) * paddedCols + k - 1] = array[i * cols]; // Left border
            paddedArray[(i + padRows) * paddedCols + paddedCols - k] = array[i * cols + cols - 1]; // Right border
        }
    }

    // Duplicate border elements vertically
    for (int i = 0; i < padRows; ++i){
        for (int j = 0; j < paddedCols; ++j) {
            paddedArray[(i*paddedCols) + j] = paddedArray[(padRows*paddedCols) + j];
            paddedArray[(i*paddedCols + (paddedCols*padRows + paddedCols*rows)) + j] = paddedArray[((paddedRows* paddedCols) - (padRows*paddedCols) - paddedCols) + j];
        }
    }

    return paddedArray;
}

std::vector<double> SharpnessFeature::median_blur(const std::vector<unsigned int>& img, int rows, int cols, int ksize) {

    int pad = (ksize-1) / 2;

    int row_padding = rows;
    int col_padding = cols;
    
    int padded_rows = rows + 2 * row_padding;
    int padded_cols = cols + 2 * col_padding;

    auto padded_img = pad_array(img, rows, cols, row_padding, col_padding);

    std::vector<double> temp (padded_rows * padded_cols, 0);

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

            temp[i * padded_cols + j] = window[median_index];
        }
    }

    auto out = remove_padding(temp, padded_rows, padded_cols, rows, cols);

    return out;
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

std::vector<double> SharpnessFeature::smooth_image(const std::vector<unsigned int>& image, int rows, int cols, bool transpose, double epsilon) {

    std::vector<double> kernel {-0.5, 0, 0.5};

    std::vector<double> result;

    auto in = transpose ? transpose_vector(image, rows, cols) : image;

    for (int i = 0; i < rows; ++i) {
        auto conv = convolve_1d(std::vector<double>(in.begin() + i*cols, in.begin() + (i+1)*cols), kernel);
        for (auto& pix: conv) {
            result.emplace_back(pix);
        }
    }
    

    if (transpose) {
        result = transpose_vector(result, rows, cols);
    }

    auto max = *std::max_element(result.begin(), result.end());

    for (auto& pix: result) {
        pix = std::abs(pix)/ (max+epsilon);
    }

    return result;
}

std::tuple<std::vector<double>, std::vector<double>> SharpnessFeature::edges(const std::vector<unsigned int>& image, int rows, int cols, double edge_threshold) {

    std::vector<double> edge_x, edge_y; 

    auto smooth_x = smooth_image(image, rows, cols, true);
    auto smooth_y = smooth_image(image, rows, cols);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {

            if (smooth_x[i * cols + j] > edge_threshold) {
                edge_x.emplace_back(1);
            } else {
                edge_x.emplace_back(0);
            }

            if (smooth_y[i * cols + j] > edge_threshold) {
                edge_y.emplace_back(1);
            } else {
                edge_y.emplace_back(0);
            }
        }
    }

    return std::make_tuple(edge_x, edge_y);
}

// Function to calculate the absolute difference of matrices
std::vector<double> SharpnessFeature::absolute_difference(const std::vector<double>& mat1, const std::vector<double>& mat2, int numRows, int numCols) {

    std::vector<double> result(numRows * numCols, 0);
    
    for (int i = 0; i < numRows; ++i) {
        for (int j = 0; j < numCols; ++j) {
            result[i * numCols + j] = mat1[i * numCols + j] + mat2[i * numCols + j];
        }
    }

    return result;
}

std::tuple<std::vector<double>, std::vector<double>> SharpnessFeature::dom(std::vector<double>& Im, int rows, int cols) {

    // Padding for median_shift_up and median_shift_down
    std::vector<double> median_shift_up = Im;
    for (int i = 0; i < 2*rows; ++i) {
        median_shift_up.emplace_back(0);
    }
    median_shift_up.erase(median_shift_up.begin(), median_shift_up.begin() + 2*cols);

    std::vector<double> median_shift_down = Im;
    for (int i = 0; i < 2*rows; ++i) {
        median_shift_down.insert(median_shift_down.begin(), 0);
    }
    median_shift_down.erase(median_shift_down.end() - (2*cols), median_shift_down.end());

    
    // Calculate domx
    std::vector<double> dom_x(cols * rows);
    for (int i = 0; i < rows * cols; ++i) {
        dom_x[i] = std::abs(median_shift_up[i] - 2 * Im[i] + median_shift_down[i]);
    }
    
    // Padding for median_shift_left and median_shift_right
    std::vector<double> median_shift_left = Im;
    std::vector<double> median_shift_right = Im;

    for (int i = 0; i < rows; ++i) {
        median_shift_left.insert(median_shift_left.begin() + i * cols + cols + 2*i, 0);
        median_shift_left.insert(median_shift_left.begin() + i * cols + cols + 2*i, 0);

        median_shift_right.insert(median_shift_right.begin() + i * cols + 2*i, 0);
        median_shift_right.insert(median_shift_right.begin() + i * cols + 2*i, 0);
    }

    for (int i = 0; i < rows; ++i) {
        median_shift_left.erase(median_shift_left.begin() + cols*i);
        median_shift_left.erase(median_shift_left.begin() + cols*i);

        median_shift_right.erase(median_shift_right.begin() + i * cols + cols);
        median_shift_right.erase(median_shift_right.begin() + i * cols + cols);
    }

    // Calculate domx]
    std::vector<double> dom_y(cols * rows);
    for (int i = 0; i < rows * cols; ++i) {
        dom_y[i] = std::abs(median_shift_left[i] - 2 * Im[i] + median_shift_right[i]);
    }  
    
    return std::make_tuple(dom_x, dom_y);
}

std::tuple<std::vector<double>, std::vector<double>> SharpnessFeature::contrast(const std::vector<double>& Im, int rows, int cols) {

    std::vector<double> cy_shift = Im;
    for (int i = 0; i < rows; ++i) {
        cy_shift.emplace_back(0);
    }
    cy_shift.erase(cy_shift.begin(), cy_shift.begin() + cols);

    std::vector<double> cx_shift = Im;
    for (int i = 0; i < rows; ++i) {
        cx_shift.insert(cx_shift.begin(), 0);
    }
    cx_shift.erase(cx_shift.end() - (cols), cx_shift.end());

    

    std::vector<double> cx(rows*cols), cy(rows*cols);

    for (int i = 0; i < rows*cols; ++i) {
        cx[i] = std::abs(Im[i] - cx_shift[i]);
        cy[i] = std::abs(Im[i] - cy_shift[i]);
    }

    return std::make_tuple(cx, cy);
}


double SharpnessFeature::sharpness(const ImageMatrix& Im, int width) {

    readOnlyPixels image = Im.ReadablePixels();

    int rows = Im.height;
    int cols = Im.width;

    auto blurred = median_blur(image, rows, cols, 3);

    for (auto& pix: blurred) {
        pix /= 255.;
    }
    
    std::vector<double> edge_x, edge_y;

    std::tie(edge_x, edge_y) = edges(image, rows, cols);

    std::vector<double> dom_x, dom_y;
    std::tie(dom_x, dom_y) = dom(blurred, rows, cols);

    std::vector<double> cx, cy;
    std::tie(cx, cy) = contrast(blurred, rows, cols);

    for (int i = 0; i < rows*cols; ++i) {
        cx[i] *= edge_x[i];
        cy[i] *= edge_y[i];
    }

    std::vector<double> sx(rows*cols, 0), sy(rows*cols, 0);

    for (int i = width; i < rows-width; ++i) {
        
        std::vector<double> num(cols, 0.0);
        std::vector<double> dn(cols, 0.0);

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
    }

    for (int i = width; i < rows-width; ++i) {
        
        std::vector<double> num(cols, 0.0);
        std::vector<double> dn(cols, 0.0);

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