#include "caliper.h"
#include "../helpers/helpers.h"
#include "../parallel.h"
#include "rotation.h"

CaliperFeretFeature::CaliperFeretFeature() : FeatureMethod("CaliperFeretFeature")
{
	// Letting the feature dependency manager know 
	provide_features({
			MIN_FERET_ANGLE,
			MAX_FERET_ANGLE,
			STAT_FERET_DIAM_MIN,
			STAT_FERET_DIAM_MAX,
			STAT_FERET_DIAM_MEAN,
			STAT_FERET_DIAM_MEDIAN,
			STAT_FERET_DIAM_STDDEV,
			STAT_FERET_DIAM_MODE});

	add_dependencies({ CONVEX_HULL_AREA });
}

void CaliperFeretFeature::calculate(LR& r)
{
	if (r.has_bad_data())
		return;

	std::vector<float> angles;
	std::vector<double> ferets;
	calculate_angled_caliper_measurements (r.convHull_CH, angles, ferets);

	// Statistics of Feret diameters
	auto s = ComputeCommonStatistics2 (ferets);
	_min = s.min;
	_max = s.max;
	_mean = s.mean;
	_median = s.median;
	_stdev = s.stdev;
	_mode = s.mode;

	// Angles
	size_t min_i = Nyxus::get_min_idx (ferets);
	size_t max_i = Nyxus::get_max_idx (ferets);
	minFeretAngle = angles [min_i];
	maxFeretAngle = angles [max_i];
}

void CaliperFeretFeature::save_value(std::vector<std::vector<double>>& fvals)
{
	fvals[MIN_FERET_ANGLE][0] = minFeretAngle;
	fvals[MAX_FERET_ANGLE][0] = maxFeretAngle;
	fvals[STAT_FERET_DIAM_MIN][0] = _min;
	fvals[STAT_FERET_DIAM_MAX][0] = _max;
	fvals[STAT_FERET_DIAM_MEAN][0] = _mean;
	fvals[STAT_FERET_DIAM_MEDIAN][0] = _median;
	fvals[STAT_FERET_DIAM_STDDEV][0] = _stdev;
	fvals[STAT_FERET_DIAM_MODE][0] = _mode;
}

//void CaliperFeretFeature::calculate_imp (const std::vector<Pixel2>& convex_hull, std::vector<double>& D_min, std::vector<double>& D_max)
void CaliperFeretFeature::calculate_angled_caliper_measurements (const std::vector<Pixel2>& convex_hull, std::vector<float>& angles, std::vector<double>& ferets)
{
	// Rotated convex hull
	std::vector<Pixel2> CH_rot;
	CH_rot.reserve (convex_hull.size());

	// Rotate and calculate the diameter
	angles.clear();
	ferets.clear();
	for (float theta = 0.f; theta <= 180.f; theta += rot_angle_increment)
	{
		Rotation::rotate_around_center (convex_hull, theta, CH_rot);
		auto [minX, minY, maxX, maxY] = AABB::from_pixelcloud (CH_rot);

		// Save a caliper measurement orthogonal to X
		double feret = maxX - minX;
		if (feret > 0)
		{
			angles.push_back (theta);
			ferets.push_back (feret);
		}
	}
}

void CaliperFeretFeature::osized_calculate(LR& r, ImageLoader&)
{
	// Calculating this feature does not require access to the massive ROI pixel cloud, 
	// so we can reuse the trivial calculate()
	calculate(r);
}

void CaliperFeretFeature::parallel_process(std::vector<int>& roi_labels, std::unordered_map <int, LR>& roiData, int n_threads)
{
	size_t jobSize = roi_labels.size(),
		workPerThread = jobSize / n_threads;

	runParallel(CaliperFeretFeature::parallel_process_1_batch, n_threads, workPerThread, jobSize, &roi_labels, &roiData);
}

void CaliperFeretFeature::parallel_process_1_batch(size_t firstitem, size_t lastitem, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData)
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
		CaliperFeretFeature f;
		f.calculate(r);
		f.save_value(r.fvals);
	}
}
