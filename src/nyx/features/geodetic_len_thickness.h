#pragma once

#include <unordered_map>
#include "../dataset.h"
#include "../roi_cache.h"
#include <tuple>
#include "pixel.h"
#include "../feature_method.h"
#include "../feature_settings.h"

/// @brief The geodetic lengths and thickness are approximated by a rectangle with the same area and perimeter: area = geodeticlength * thickness; perimeter = 2 * (geodetic_length + thickness).
class GeodeticLengthThicknessFeature:public FeatureMethod
{
public:

	const constexpr static std::initializer_list<Nyxus::Feature2D> featureset =
	{
		Nyxus::Feature2D::GEODETIC_LENGTH,
		Nyxus::Feature2D::THICKNESS
	};

	GeodeticLengthThicknessFeature();
	
	void calculate (LR& r, const Fsettings& s);
	void osized_add_online_pixel(size_t x, size_t y, uint32_t intensity);
	void osized_calculate (LR& r, const Fsettings& s, ImageLoader& ldr);
	void save_value(std::vector<std::vector<double>>& feature_vals);
	static void extract (LR& roi, const Fsettings& s);
	static void parallel_process_1_batch (size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData, const Fsettings& s, const Dataset & ds);

	static bool required(const FeatureSet& fs);
private:
	double geodetic_length = 0, thickness = 0;
};

