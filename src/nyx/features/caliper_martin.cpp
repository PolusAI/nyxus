#include <algorithm>	// FIX (caliper reimpl): std::min/std::max used by the analytic chord helper
#include <cmath>		// FIX (caliper reimpl): std::abs
#include <vector>
#include "caliper.h"
#include "../environment.h"
#include "rotation.h"

using namespace Nyxus;

// FIX (caliper reimpl): width of the convex hull at a horizontal cut y = span of the
// x-coordinates where the line y intersects the hull edges. The hull (convHull_CH) is
// stored OPEN (no duplicated closing vertex), so we must include the wrap-around edge
// last->first — the previous implementation omitted it. Inclusive edge test + min/max of
// the intersection x is robust to a line passing exactly through a shared vertex.
static double hull_width_at_y (const std::vector<Pixel2>& poly, double y)
{
	bool have = false;
	double xlo = 0.0, xhi = 0.0;
	size_t n = poly.size();
	for (size_t i = 0; i < n; i++)
	{
		const Pixel2& a = poly[i];
		const Pixel2& b = poly[(i + 1) % n];
		double ay = a.y, by = b.y, lo = std::min(ay, by), hi = std::max(ay, by);
		if (y < lo || y > hi)
			continue;
		double e0, e1;
		if (by != ay)
		{
			double x = a.x + (b.x - a.x) * (y - ay) / (by - ay);
			e0 = e1 = x;
		}
		else	// horizontal edge lying on the cut: it spans [min x, max x]
		{
			e0 = std::min ((double)a.x, (double)b.x);
			e1 = std::max ((double)a.x, (double)b.x);
		}
		if (!have) { xlo = e0; xhi = e1; have = true; }
		else { xlo = std::min (xlo, e0); xhi = std::max (xhi, e1); }
	}
	return have ? (xhi - xlo) : 0.0;
}

CaliperMartinFeature::CaliperMartinFeature() : FeatureMethod("CaliperMartinFeature")
{
	// Letting the feature dependency manager know
	provide_features (CaliperMartinFeature::featureset);
}

void CaliperMartinFeature::calculate (LR& r, const Fsettings& settings)
{
	// intercept void ROIs
	if (r.convHull_CH.size() == 0)
	{
		_min =
		_max =
		_mean =
		_median =
		_stdev =
		_mode = settings[(int)NyxSetting::SOFTNAN].rval;	// former theEnvironment.resultOptions.noval()

		return;
	}

	std::vector<double> allD;	// Diameters at 0-180 degrees rotation
	calculate_imp(r.convHull_CH, allD);

	auto s = ComputeCommonStatistics2(allD);

	_min = (double)s.min;
	_max = (double)s.max;
	_mean = s.mean;
	_median = s.median;
	_stdev = s.stdev;
	_mode = (double)s.mode;
}

void CaliperMartinFeature::save_value(std::vector<std::vector<double>>& fvals)
{
	fvals[(int)Feature2D::STAT_MARTIN_DIAM_MIN][0] = _min;
	fvals[(int)Feature2D::STAT_MARTIN_DIAM_MAX][0] = _max;
	fvals[(int)Feature2D::STAT_MARTIN_DIAM_MEAN][0] = _mean;
	fvals[(int)Feature2D::STAT_MARTIN_DIAM_MEDIAN][0] = _median;
	fvals[(int)Feature2D::STAT_MARTIN_DIAM_STDDEV][0] = _stdev;
	fvals[(int)Feature2D::STAT_MARTIN_DIAM_MODE][0] = _mode;
}

void CaliperMartinFeature::calculate_imp(const std::vector<Pixel2>& convex_hull, std::vector<double>& all_D)
{
	// FIX (caliper reimpl): the previous code was NOT the Martin diameter at all — it pushed
	// BOTH the shortest and longest of a Y-grid of horizontal chords per angle, so the per-angle
	// shortest (a spurious near-apex ~0-length chord) corrupted MIN/MEAN/MEDIAN/MODE/STDDEV.
	// The Martin diameter (Pahl/Rumpf 1973; imea reference) is a SINGLE chord per angle: the
	// horizontal chord at the level that bisects the projected area (50%/50%). We reproduce that
	// analytically on the rotated convex hull — one diameter per rotation angle.
	std::vector<Pixel2> CH_rot;
	CH_rot.reserve(convex_hull.size());

	all_D.clear();
	const int NGRID = 100;	// FIX: fine Y-grid for the area integral (converges well before this; see morph_oracle/caliper_proto.py)
	for (float theta = 0.f; theta < 180.f; theta += rot_angle_increment)
	{
		Rotation::rotate_around_center(convex_hull, theta, CH_rot);
		auto [minX, minY, maxX, maxY] = AABB::from_pixelcloud(CH_rot);
		if (maxY <= minY)	// FIX: degenerate (collinear) hull at this angle — skip
			continue;

		// FIX: sample the hull width at NGRID horizontal levels (midpoint rule) and integrate area
		double stepY = (double(maxY) - double(minY)) / NGRID;
		std::vector<double> widths(NGRID);
		double total = 0.0;
		for (int i = 0; i < NGRID; i++)
		{
			double y = double(minY) + (i + 0.5) * stepY;
			widths[i] = hull_width_at_y(CH_rot, y);
			total += widths[i];
		}
		if (total <= 0.0)	// FIX: empty cut set — skip
			continue;

		// FIX: walk the cumulative area from one side; the width at the 50% level is the Martin diameter
		double half = 0.5 * total, cum = 0.0, martin = widths[NGRID - 1];
		for (int i = 0; i < NGRID; i++)
		{
			cum += widths[i];
			if (cum >= half)
			{
				martin = widths[i];
				break;
			}
		}
		all_D.push_back(martin);	// FIX: one Martin diameter per angle (was two spurious values)
	}
}

void CaliperMartinFeature::osized_calculate (LR& r, const Fsettings& settings, ImageLoader&)
{
	// FIX (caliper reimpl): the out-of-RAM path duplicated the same defective min+max chord logic.
	// Route it through the corrected area-bisecting calculate_imp so both paths agree.
	std::vector<double> all_D;
	calculate_imp(r.convHull_CH, all_D);

	// Process the stats
	auto s = ComputeCommonStatistics2(all_D);

	_min = (double)s.min;
	_max = (double)s.max;
	_mean = s.mean;
	_median = s.median;
	_stdev = s.stdev;
	_mode = (double)s.mode;
}

void CaliperMartinFeature::extract (LR& r, const Fsettings& s)
{
	CaliperMartinFeature f;
	f.calculate (r, s);
	f.save_value (r.fvals);
}

void CaliperMartinFeature::parallel_process_1_batch (size_t firstitem, size_t lastitem, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData, const Fsettings & s, const Dataset & _)
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
		CaliperMartinFeature f;
		f.calculate (r, s);
		f.save_value (r.fvals);
	}
}
