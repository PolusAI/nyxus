#pragma once

#include <unordered_map>
#include <vector>
#include "../roi_cache.h"
#include "pixel.h"
#include "../feature_method.h"

class CaliperNassensteinFeature : public FeatureMethod
{
public:
	const constexpr static std::initializer_list<Nyxus::Feature2D> featureset =
	{
		Nyxus::Feature2D::STAT_NASSENSTEIN_DIAM_MIN,
		Nyxus::Feature2D::STAT_NASSENSTEIN_DIAM_MAX,
		Nyxus::Feature2D::STAT_NASSENSTEIN_DIAM_MEAN,
		Nyxus::Feature2D::STAT_NASSENSTEIN_DIAM_MEDIAN,
		Nyxus::Feature2D::STAT_NASSENSTEIN_DIAM_STDDEV,
		Nyxus::Feature2D::STAT_NASSENSTEIN_DIAM_MODE
	};

	CaliperNassensteinFeature();
	void calculate (LR& r);
	void osized_add_online_pixel (size_t x, size_t y, uint32_t intensity) {};		// No online mode for this feature
	void osized_calculate (LR& r, ImageLoader& imloader);
	void save_value (std::vector<std::vector<double>>& feature_vals);
	static void parallel_process_1_batch (size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData);
	void parallel_process (std::vector<int>& roi_labels, std::unordered_map <int, LR>& roiData, int n_threads);

	// Compatibility with manual reduce
	static bool required(const FeatureSet& fs) 
	{
		return fs.anyEnabled (CaliperNassensteinFeature::featureset);
	}

private:
	void calculate_imp(const std::vector<Pixel2>& convex_hull, std::vector<double>& all_D);

	// Results instance cache
	double _min = 0, _max = 0, _mean = 0, _median = 0, _stdev = 0, _mode = 0;

	// Implementation constant
	const float rot_angle_increment = 10.f;	// degrees
	const int n_steps = 10;
};

class CaliperFeretFeature : public FeatureMethod
{
public:
	const constexpr static std::initializer_list<Nyxus::Feature2D> featureset = 
	{
		Nyxus::Feature2D::MIN_FERET_DIAMETER,
		Nyxus::Feature2D::MAX_FERET_DIAMETER,
		Nyxus::Feature2D::MIN_FERET_ANGLE,
		Nyxus::Feature2D::MAX_FERET_ANGLE,
		Nyxus::Feature2D::STAT_FERET_DIAM_MIN,
		Nyxus::Feature2D::STAT_FERET_DIAM_MAX,
		Nyxus::Feature2D::STAT_FERET_DIAM_MEAN,
		Nyxus::Feature2D::STAT_FERET_DIAM_MEDIAN,
		Nyxus::Feature2D::STAT_FERET_DIAM_STDDEV,
		Nyxus::Feature2D::STAT_FERET_DIAM_MODE
	};

	CaliperFeretFeature();
	void calculate(LR& r);
	void osized_add_online_pixel(size_t x, size_t y, uint32_t intensity) {};		// No online mode for this feature
	void osized_calculate(LR& r, ImageLoader& imloader);
	void save_value(std::vector<std::vector<double>>& feature_vals);
	static void parallel_process_1_batch(size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData);
	void parallel_process(std::vector<int>& roi_labels, std::unordered_map <int, LR>& roiData, int n_threads);

	// Compatibility with manual reduce
	static bool required(const FeatureSet& fs) 
	{
		return fs.anyEnabled (CaliperFeretFeature::featureset);
	}

private:	
	void calculate_angled_caliper_measurements (const std::vector<Pixel2>& cnovhull, std::vector<float>& angles, std::vector<double>& feret_measurements);

	// Results instance cache
	double 
		minFeretAngle = 0,
		maxFeretAngle = 0, 
		_min = 0, 
		_max = 0, 
		_mean = 0, 
		_median = 0, 
		_stdev = 0, 
		_mode = 0;

	// Implementation constant
	const double rot_angle_increment = 10.f;	// degrees
	const int n_steps = 10;
};

class CaliperMartinFeature : public FeatureMethod
{
public:
	const constexpr static std::initializer_list<Nyxus::Feature2D> featureset =
	{
		Nyxus::Feature2D::STAT_MARTIN_DIAM_MIN,
		Nyxus::Feature2D::STAT_MARTIN_DIAM_MAX,
		Nyxus::Feature2D::STAT_MARTIN_DIAM_MEAN,
		Nyxus::Feature2D::STAT_MARTIN_DIAM_MEDIAN,
		Nyxus::Feature2D::STAT_MARTIN_DIAM_STDDEV,
		Nyxus::Feature2D::STAT_MARTIN_DIAM_MODE
	};

	CaliperMartinFeature();
	void calculate(LR& r);
	void osized_add_online_pixel(size_t x, size_t y, uint32_t intensity) {};		
	void osized_calculate(LR& r, ImageLoader& imloader);
	void save_value(std::vector<std::vector<double>>& feature_vals);
	static void parallel_process_1_batch(size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData);
	void parallel_process(std::vector<int>& roi_labels, std::unordered_map <int, LR>& roiData, int n_threads);

	// Compatibility with manual reduce
	static bool required(const FeatureSet& fs) 
	{
		return fs.anyEnabled (CaliperMartinFeature::featureset);
	}

private:
	void calculate_imp(const std::vector<Pixel2>& convex_hull, std::vector<double>& all_D);

	// Results instance cache
	double _min = 0, _max = 0, _mean = 0, _median = 0, _stdev = 0, _mode = 0;

	// Implementation constant
	const float rot_angle_increment = 10.f;	// degrees
	const int n_steps = 10;
};

