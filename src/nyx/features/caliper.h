#pragma once

#include <unordered_map>
#include <vector>
#include "../roi_cache.h"
#include "pixel.h"
#include "../feature_method.h"

/// @brief Longest chord. Feret, Martin, Nassenstein diameters.
class ParticleMetrics_features
{
public:
	static bool feret_required(const FeatureSet& fs)
	{
		return (fs.anyEnabled({ MIN_FERET_DIAMETER, MAX_FERET_DIAMETER, MIN_FERET_ANGLE, MAX_FERET_ANGLE }) ||
			fs.anyEnabled({
				STAT_FERET_DIAM_MIN,
				STAT_FERET_DIAM_MAX,
				STAT_FERET_DIAM_MEAN,
				STAT_FERET_DIAM_MEDIAN,
				STAT_FERET_DIAM_STDDEV,
				STAT_FERET_DIAM_MODE })
				);
	}
	static bool martin_required(const FeatureSet& fs)
	{
		return fs.anyEnabled({ STAT_MARTIN_DIAM_MIN,
				STAT_MARTIN_DIAM_MAX,
				STAT_MARTIN_DIAM_MEAN,
				STAT_MARTIN_DIAM_MEDIAN,
				STAT_MARTIN_DIAM_STDDEV,
				STAT_MARTIN_DIAM_MODE });
	}
	static bool nassenstein_required(const FeatureSet& fs)
	{
		return fs.anyEnabled({ STAT_NASSENSTEIN_DIAM_MIN,
				STAT_NASSENSTEIN_DIAM_MAX,
				STAT_NASSENSTEIN_DIAM_MEAN,
				STAT_NASSENSTEIN_DIAM_MEDIAN,
				STAT_NASSENSTEIN_DIAM_STDDEV,
				STAT_NASSENSTEIN_DIAM_MODE });
	}

	ParticleMetrics_features(std::vector<Pixel2>& _convex_hull);

	void calc_ferret(
		// output:
		double& minFeretDiameter,
		double& minFeretAngle,
		double& maxFeretDiameter,
		double& maxFeretAngle,
		std::vector<double>& all_D);
	void calc_martin(std::vector<double>& D);
	void calc_nassenstein(std::vector<double>& D);

	static void reduce_feret (size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData);
	static void reduce_martin (size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData);
	static void reduce_nassenstein (size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData);

private:
	std::vector<Pixel2>& convex_hull;
	const int NY = 10;
	const int rot_angle_increment = 10;	// degrees
};

class CaliperNassensteinFeature : public FeatureMethod
{
public:
	CaliperNassensteinFeature();

	void calculate (LR& r);

	void osized_add_online_pixel (size_t x, size_t y, uint32_t intensity) {};		// No online mode for this feature
	void osized_calculate (LR& r, ImageLoader& imloader);

	// Makes results CSV-ready
	void save_value (std::vector<std::vector<double>>& feature_vals);

	static bool required(const FeatureSet& fs) { return fs.anyEnabled({ 
		STAT_NASSENSTEIN_DIAM_MIN,
		STAT_NASSENSTEIN_DIAM_MAX,
		STAT_NASSENSTEIN_DIAM_MEAN,
		STAT_NASSENSTEIN_DIAM_MEDIAN,
		STAT_NASSENSTEIN_DIAM_STDDEV,
		STAT_NASSENSTEIN_DIAM_MODE }); }

	// Parallel processing (trivial ROI)
	static void parallel_process_1_batch (size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData);
	void parallel_process (std::vector<int>& roi_labels, std::unordered_map <int, LR>& roiData, int n_threads);

private:
	// Implements feature calculation for a trivial ROI, saves result in instance cache
	void calculate_imp (const std::vector<Pixel2>& cnovhull, std::vector<double>& D);

	// Results instance cache
	double _min = 0, _max = 0, _mean = 0, _median = 0, _stdev = 0, _mode = 0;

	// Implementation constant
	const float rot_angle_increment = 10.f;	// degrees
	const int n_steps = 10;
};

class CaliperFeretFeature : public FeatureMethod
{
public:
	CaliperFeretFeature();
	void calculate(LR& r);
	void osized_add_online_pixel(size_t x, size_t y, uint32_t intensity) {};
	void save_value(std::vector<std::vector<double>>& feature_vals);
	static void reduce(size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData);
	static bool required(const FeatureSet& fs) {
		return fs.anyEnabled({ 
			MIN_FERET_DIAMETER,
			MAX_FERET_DIAMETER,
			MIN_FERET_ANGLE,
			MAX_FERET_ANGLE,
			STAT_FERET_DIAM_MIN,
			STAT_FERET_DIAM_MAX,
			STAT_FERET_DIAM_MEAN,
			STAT_FERET_DIAM_MEDIAN,
			STAT_FERET_DIAM_STDDEV,
			STAT_FERET_DIAM_MODE });
	}

protected:
	void osized_calculate(LR& r, ImageLoader& imloader);

private:
};

class CaliperMartinFeature : public FeatureMethod
{
public:
	CaliperMartinFeature();
	void calculate(LR& r);
	void osized_add_online_pixel(size_t x, size_t y, uint32_t intensity) {};
	void save_value(std::vector<std::vector<double>>& feature_vals);
	static void reduce(size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData);
	static bool required(const FeatureSet& fs) {
		return fs.anyEnabled({ 
			STAT_MARTIN_DIAM_MIN,
			STAT_MARTIN_DIAM_MAX,
			STAT_MARTIN_DIAM_MEAN,
			STAT_MARTIN_DIAM_MEDIAN,
			STAT_MARTIN_DIAM_STDDEV,
			STAT_MARTIN_DIAM_MODE });
	}
protected:
	void osized_calculate(LR& r, ImageLoader& imloader);

private:
};

