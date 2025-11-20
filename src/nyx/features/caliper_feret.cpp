#include "caliper.h"
#include "../environment.h"
#include "../helpers/helpers.h"
#include "rotation.h"

using namespace Nyxus;

CaliperFeretFeature::CaliperFeretFeature() : FeatureMethod("CaliperFeretFeature")
{
	// Letting the feature dependency manager know 
	provide_features (CaliperFeretFeature::featureset);
	add_dependencies ({ Feature2D::CONVEX_HULL_AREA });
}

void CaliperFeretFeature::calculate (LR& r, const Fsettings& s)
{
	// intercept void ROIs
	if (r.convHull_CH.size() == 0)
	{
		minFeretAngle =
		maxFeretAngle =
		_min =
		_max =
		_mean =
		_median =
		_stdev =
		_mode = s[(int)NyxSetting::SOFTNAN].rval; // former theEnvironment.resultOptions.noval()

		return;
	}

	std::vector<float> angles;
	std::vector<double> ferets;
	calculate_angled_caliper_measurements (r.convHull_CH, angles, ferets);

	// Statistics of Feret diameters
	if (ferets.size())
	{
		// angles (find them before 'ferets' gets sorted by the statistics calculator)
		std::tuple<size_t, size_t> minmax = Nyxus::get_minmax_idx(ferets);
		minFeretAngle = angles[std::get<0>(minmax)];
		maxFeretAngle = angles[std::get<1>(minmax)];		
		
		// diameters
		auto s = ComputeCommonStatistics2 (ferets); 
		_min = s.min;
		_max = s.max;
		_mean = s.mean;
		_median = s.median;
		_stdev = s.stdev;
		_mode = s.mode;
	}
	else
	{
		// measurement rotations went wrong, so report failure feture values
		minFeretAngle =
		maxFeretAngle =
		_min =
		_max =
		_mean =
		_median =
		_stdev =
		_mode = s[(int)NyxSetting::SOFTNAN].rval;	// former theEnvironment.resultOptions.noval()
	}
}

void CaliperFeretFeature::save_value(std::vector<std::vector<double>>& fvals)
{
	fvals[(int)Feature2D::MIN_FERET_ANGLE][0] = minFeretAngle;
	fvals[(int)Feature2D::MAX_FERET_ANGLE][0] = maxFeretAngle;
	fvals[(int)Feature2D::STAT_FERET_DIAM_MIN][0] = _min;
	fvals[(int)Feature2D::STAT_FERET_DIAM_MAX][0] = _max;
	fvals[(int)Feature2D::STAT_FERET_DIAM_MEAN][0] = _mean;
	fvals[(int)Feature2D::STAT_FERET_DIAM_MEDIAN][0] = _median;
	fvals[(int)Feature2D::STAT_FERET_DIAM_STDDEV][0] = _stdev;
	fvals[(int)Feature2D::STAT_FERET_DIAM_MODE][0] = _mode;
}

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

void CaliperFeretFeature::osized_calculate (LR& r, const Fsettings& s, ImageLoader&)
{
	// Calculating this feature does not require access to the massive ROI pixel cloud, 
	// so we can reuse the trivial calculate()
	calculate (r, s);
}

void CaliperFeretFeature::extract (LR& r, const Fsettings& s)
{
		CaliperFeretFeature f;
		f.calculate (r, s);
		f.save_value (r.fvals);
}

void CaliperFeretFeature::parallel_process_1_batch (size_t firstitem, size_t lastitem, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData, const Fsettings & s, const Dataset & _)
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
