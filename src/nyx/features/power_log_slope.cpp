#include "power_log_slope.h"
#include "../helpers/helpers.h"

using namespace Nyxus;

PowerLogSlopeFeature::PowerLogSlopeFeature() : FeatureMethod("PowerLogSlopeFeature") {
    provide_features({Feature2D::POWER_LOG_SLOPE});
}

void PowerLogSlopeFeature::calculate(LR& r) {


}

bool PowerLogSlopeFeature::required(const FeatureSet& fs) 
{ 
    return fs.isEnabled (Feature2D::POWER_LOG_SLOPE); 
}

void PowerLogSlopeFeature::reduce (size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData)
{
    for (auto i = start; i < end; i++)
    {
        int lab = (*ptrLabels)[i];
        LR& r = (*ptrLabelData)[lab];

        PowerLogSlopeFeature plsf;

        plsf.calculate (r);

        plsf.save_value (r.fvals);
    }
}

void PowerLogSlopeFeature::save_value(std::vector<std::vector<double>>& feature_vals) {
    
    feature_vals[(int)Feature2D::POWER_LOG_SLOPE][0] = fvals[0];

}


void rps(std::vector<unsigned int> image, int rows, int cols) {

    auto rows_arrange = arrange(0, rows, 1); 
    auto cols_arrange = arrange(0, cols, 1); 

    std::vector<std::vector<int>> added_vecs (rows, std::vector<int>(cols));

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            added_vecs[i][j] = rows_arrange[i] + cols_arrange[j];
        }
    }

    std::cout << "added vec: " << std::endl;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << added_vecs[i][j] << " ";
        }
        std::cout << std::endl;
    }

}
