#include <algorithm>	// FIX (caliper reimpl): std::min/std::max used by the analytic chord helper
#include <cmath>		// FIX (caliper reimpl): std::abs
#include <vector>
#include "caliper.h"
#include "../environment.h"
#include "rotation.h"

using namespace Nyxus;

// FIX (caliper reimpl): height of the convex hull at a vertical cut x = span of the
// y-coordinates where the line x intersects the hull edges. Hull is stored OPEN, so the
// wrap-around edge (last->first) is included. Inclusive edge test + min/max is robust to a
// line passing exactly through a shared vertex (the Nassenstein contact column is a vertex).
// FIX (caliper float-precision): operate on Point2f (float-precision rotated hull) so the chord height is not
// quantized by the old integer-Pixel2 truncation.
static double hull_height_at_x (const std::vector<Point2f>& poly, double x)
{
	bool have = false;
	double ylo = 0.0, yhi = 0.0;
	size_t n = poly.size();
	for (size_t i = 0; i < n; i++)
	{
		const Point2f& a = poly[i];
		const Point2f& b = poly[(i + 1) % n];
		double ax = a.x, bx = b.x, lo = std::min(ax, bx), hi = std::max(ax, bx);
		if (x < lo || x > hi)
			continue;
		double e0, e1;
		if (bx != ax)
		{
			double yv = a.y + (b.y - a.y) * (x - ax) / (bx - ax);
			e0 = e1 = yv;
		}
		else	// vertical edge lying on the cut: it spans [min y, max y]
		{
			e0 = std::min ((double)a.y, (double)b.y);
			e1 = std::max ((double)a.y, (double)b.y);
		}
		if (!have) { ylo = e0; yhi = e1; have = true; }
		else { ylo = std::min (ylo, e0); yhi = std::max (yhi, e1); }
	}
	return have ? (yhi - ylo) : 0.0;
}

CaliperNassensteinFeature::CaliperNassensteinFeature() : FeatureMethod("CaliperNassensteinFeature")
{
	// Letting the feature dependency manager know
	provide_features (CaliperNassensteinFeature::featureset);
}

void CaliperNassensteinFeature::calculate (LR& r, const Fsettings& settings)
{
	// intercept void ROIs
	if (r.convHull_CH.size() == 0)
	{
		_min =
		_max =
		_mean =
		_median =
		_stdev =
		_mode = settings [(int)NyxSetting::SOFTNAN].rval;	// former theEnvironment.resultOptions.noval()
		
		return;
	}

	std::vector<double> allD;	// Diameters at 0-180 degrees rotation
	calculate_imp (r.convHull_CH, allD);

	auto s = ComputeCommonStatistics2(allD);

	_min = (double)s.min;
	_max = (double)s.max;
	_mean = s.mean;
	_median = s.median;
	_stdev = s.stdev;
	_mode = (double)s.mode;
}

void CaliperNassensteinFeature::save_value (std::vector<std::vector<double>>& fvals)
{
	fvals[(int)Feature2D::STAT_NASSENSTEIN_DIAM_MIN][0] = _min;
	fvals[(int)Feature2D::STAT_NASSENSTEIN_DIAM_MAX][0] = _max;
	fvals[(int)Feature2D::STAT_NASSENSTEIN_DIAM_MEAN][0] = _mean;
	fvals[(int)Feature2D::STAT_NASSENSTEIN_DIAM_MEDIAN][0] = _median;
	fvals[(int)Feature2D::STAT_NASSENSTEIN_DIAM_STDDEV][0] = _stdev;
	fvals[(int)Feature2D::STAT_NASSENSTEIN_DIAM_MODE][0] = _mode;
}

void CaliperNassensteinFeature::calculate_imp (const std::vector<Pixel2>& convex_hull, std::vector<double>& all_D)
{
	// FIX (caliper reimpl): the previous code was byte-identical to the (also wrong) Martin loop —
	// two diameters cannot share one algorithm. It pushed both the shortest and longest horizontal
	// chord per angle, so the per-angle shortest (a near-apex ~0-length chord) drove MIN and MODE to
	// 0.0 — impossible for a solid shape. The Nassenstein diameter (Pahl/Rumpf 1973; imea reference)
	// is a SINGLE chord per angle: the vertical chord measured at the bottom-tangent contact column.
	// We reproduce that on the rotated convex hull: the contact is the extreme (max-y) vertex/edge,
	// and the diameter is the hull's vertical extent at that contact column.
	std::vector<Point2f> CH_rot;	// FIX (caliper float-precision): float-precision rotated hull (was integer Pixel2)
	CH_rot.reserve(convex_hull.size());

	all_D.clear();
	for (float theta = 0.f; theta < 180.f; theta += rot_angle_increment)
	{
		Rotation::rotate_around_center_fp(convex_hull, theta, CH_rot);	// FIX (caliper float-precision): no integer truncation
		if (CH_rot.size() < 3)	// FIX: degenerate hull — no measurable tangent chord
			continue;

		// FIX: bottom tangent = the max-y extreme; the contact column is the mean x of the max-y
		// vertices (a single vertex generically, or the midpoint of a flat bottom edge)
		double ymax = CH_rot[0].y;
		for (auto& p : CH_rot)
			ymax = std::max (ymax, (double)p.y);
		double xsum = 0.0;
		int cnt = 0;
		for (auto& p : CH_rot)
			if (std::abs((double)p.y - ymax) < 1e-3)
			{
				xsum += p.x;
				cnt++;
			}
		double xc = xsum / std::max(cnt, 1);

		// FIX: Nassenstein diameter = vertical extent of the hull at the contact column
		all_D.push_back (hull_height_at_x(CH_rot, xc));
	}
}

void CaliperNassensteinFeature::osized_calculate (LR& r, const Fsettings& settings, ImageLoader&)
{
	// FIX (caliper reimpl): the out-of-RAM path duplicated the same defective min+max chord logic.
	// Route it through the corrected bottom-tangent calculate_imp so both paths agree.
	std::vector<double> all_D;
	calculate_imp(r.convHull_CH, all_D);

	// Process the stats
	auto s = ComputeCommonStatistics2 (all_D);

	_min = (double)s.min;
	_max = (double)s.max;
	_mean = s.mean;
	_median = s.median;
	_stdev = s.stdev;
	_mode = (double)s.mode;
}

void CaliperNassensteinFeature::extract (LR& r, const Fsettings& s)
{
	CaliperNassensteinFeature f;
	f.calculate (r, s);
	f.save_value (r.fvals);
}

void CaliperNassensteinFeature::parallel_process_1_batch (size_t firstitem, size_t lastitem, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData, const Fsettings& s, const Dataset & _)
{
	// Calculate the feature for each batch ROI item 
	for (auto i = firstitem; i < lastitem; i++)
	{
		// Get ahold of ROI's label and cache
		int roiLabel = (*ptrLabels) [i];
		LR& r = (*ptrLabelData) [roiLabel];

		// Skip the ROI if its data is invalid to prevent nans and infs in the output
		if (r.has_bad_data())
			continue;

		// Calculate the feature and save it in ROI's csv-friendly buffer 'fvals'
		CaliperNassensteinFeature f;
		f.calculate (r, s);
		f.save_value (r.fvals);
	}
}
