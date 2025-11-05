#pragma once

#include <vector>
#include <unordered_map>
#include "../dataset.h"
#include "../roi_cache.h"
#include "aabb.h"
#include "pixel.h"
#include "../feature_method.h"
#include "../feature_settings.h"

/// @brief Fractal dimension determined by the box counting and the perimeter methods according to DIN ISO 9276-6 (evenly structured gait).
class FractalDimensionFeature: public FeatureMethod
{
public:
	
	const constexpr static std::initializer_list<Nyxus::Feature2D> featureset = 
	{ 
		Nyxus::Feature2D::FRACT_DIM_BOXCOUNT,
		Nyxus::Feature2D::FRACT_DIM_PERIMETER
	};

	FractalDimensionFeature();
	void calculate (LR& r, const Fsettings& s);
	void osized_add_online_pixel(size_t x, size_t y, uint32_t intensity);
	void osized_calculate (LR& r, const Fsettings& s, ImageLoader& ldr);
	void save_value(std::vector<std::vector<double>>& feature_vals);
	static void extract (LR& roi, const Fsettings& s);
	static void parallel_process_1_batch (size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData, const Fsettings & s, const Dataset & ds);

	static bool required(const FeatureSet& fs) { return fs.anyEnabled({ Nyxus::Feature2D::FRACT_DIM_BOXCOUNT, Nyxus::Feature2D::FRACT_DIM_PERIMETER }); }

private:
	void calculate_boxcount_fdim (LR& r);
	void calculate_boxcount_fdim_oversized (LR& r);
	void calculate_perimeter_fdim (LR& r);
	double calc_lyapunov_slope (const std::vector<std::pair<int, int>>& coverage_stats);
	double box_count_fd = 0, perim_fd = 0;
};