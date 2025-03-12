#include "caliper.h"
#include "../environment.h"
#include "../parallel.h"
#include "rotation.h"

using namespace Nyxus;

CaliperMartinFeature::CaliperMartinFeature() : FeatureMethod("CaliperMartinFeature")
{
	// Letting the feature dependency manager know
	provide_features (CaliperMartinFeature::featureset);
}

void CaliperMartinFeature::calculate(LR& r)
{
	// intercept void ROIs
	if (r.convHull_CH.size() == 0)
	{
		_min =
		_max =
		_mean =
		_median =
		_stdev =
		_mode = theEnvironment.nan_substitute;

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
	// Rotated convex hull
	std::vector<Pixel2> CH_rot;
	CH_rot.reserve(convex_hull.size());

	// Rotate and calculate the diameter
	all_D.clear();
	for (float theta = 0.f; theta < 180.f; theta += rot_angle_increment)
	{
		Rotation::rotate_around_center(convex_hull, theta, CH_rot);
		auto [minX, minY, maxX, maxY] = AABB::from_pixelcloud(CH_rot);

		std::vector<float> DA;	// Diameters at this angle

		// Iterate the Y-grid
		float stepY = (maxY - minY) / float(n_steps);
		for (int iy = 1; iy <= n_steps; iy++)
		{
			float chord_y = minY + iy * stepY;

			// Find convex hull segments intersecting 'y'
			std::vector<std::pair<float, float>> X;	// intersection points
			for (int iH = 1; iH < CH_rot.size(); iH++)
			{
				// The convex hull points are guaranteed to be consecutive
				auto& a = CH_rot[iH - 1],
					& b = CH_rot[iH];

				// Chord's Y is between segment AB's Ys ?
				if ((a.y >= chord_y && b.y < chord_y) || (b.y >= chord_y && a.y < chord_y))
				{
					float chord_x = b.y != a.y ?
						(b.x - a.x) * (chord_y - a.y) / (b.y - a.y) + a.x
						: (b.x + a.x) / 2;
					auto tup1 = std::make_pair(chord_x, chord_y);
					X.push_back (tup1);

					// find the 2nd intersection
					for (int kH = iH+1; kH < CH_rot.size(); kH++)
					{
						auto & a = CH_rot [kH-1],
							& b = CH_rot [kH];

						// chord chord_y crosses hull's segment (a,b) ?
						if ((a.y >= chord_y && b.y < chord_y) || (b.y >= chord_y && a.y < chord_y))
						{
							float chord_x = b.y != a.y ?
								(b.x - a.x) * (chord_y - a.y) / (b.y - a.y) + a.x
								: (b.x + a.x) / 2;
							auto tup2 = std::make_pair(chord_x, chord_y);

							// save point #2 only if it's different from #1, otherwise the chord's length will be ==0
							// (#1==#2 happens when chord's Y is exactly the Y of the contact of 2 convex hull segments.)
							if (tup1 != tup2)
								X.push_back (tup2);

							break; // done with chord point #2
						}
					}

					break;	// done with chord point #1
				}
			}

			// Save the length of this chord. There must be 2 items in 'chordEnds' because we don't allow uniformative chords of zero length
			if (X.size() == 2)	// we ignore special 1-vertex segments because their lengths are ==0
			{
				// for N segments
				auto compareFunc = [](const std::pair<float, float>& p1, const std::pair<float, float>& p2) { return p1.first < p2.first; };
				auto idx_minX = std::distance(X.begin(), std::min_element(X.begin(), X.end(), compareFunc));
				auto idx_maxX = std::distance(X.begin(), std::max_element(X.begin(), X.end(), compareFunc));
				// left X and right X segments
				auto& e1 = X[idx_minX], & e2 = X[idx_maxX];
				auto x1 = e1.first, y1 = e1.second, x2 = e2.first, y2 = e2.second;
				// save this chord
				auto dist = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2);	// Squared distance
				DA.push_back(dist);
			}
		}

		if (DA.size() > 0)
		{
			// Find the shortest and longest chords (diameters)
			double minD2 = *std::min_element(DA.begin(), DA.end()),
				maxD2 = *std::max_element(DA.begin(), DA.end()),
				min_ = sqrt(minD2),
				max_ = sqrt(maxD2);

			// Save them
			all_D.push_back(min_);
			all_D.push_back(max_);
		}
	}
}

void CaliperMartinFeature::osized_calculate(LR& r, ImageLoader&)
{
	// Rotated convex hull
	std::vector<Pixel2> CH_rot;
	CH_rot.reserve(r.convHull_CH.size());

	// Rotate and calculate the diameter
	std::vector<double> all_D;
	for (float theta = 0.f; theta < 180.f; theta += rot_angle_increment)
	{
		Rotation::rotate_around_center(r.convHull_CH, theta, CH_rot);
		auto [minX, minY, maxX, maxY] = AABB::from_pixelcloud(CH_rot);

		std::vector<float> DA;	// Diameters at this angle

		// Iterate y-grid
		float stepY = (maxY - minY) / float(n_steps);
		for (int iy = 1; iy <= n_steps; iy++)
		{
			float chord_y = minY + iy * stepY;

			// Find convex hull segments intersecting 'y'
			std::vector<std::pair<float, float>> X;	// intersection points
			for (int iH = 1; iH < CH_rot.size(); iH++)
			{
				// The convex hull points are guaranteed to be consecutive
				auto& a = CH_rot[iH - 1],
					& b = CH_rot[iH];

				// Chord's Y is between segment AB's Ys ?
				if ((a.y >= chord_y && b.y <= chord_y) || (b.y >= chord_y && a.y <= chord_y))
				{
					auto chord_x = b.y != a.y ?
						(b.x - a.x) * (chord_y - a.y) / (b.y - a.y) + a.x
						: (b.y + a.y) / 2;
					auto tup = std::make_pair(chord_x, chord_y);
					X.push_back(tup);
				}
			}

			// Save the length of this chord. There must be 2 items in 'chordEnds' because we don't allow uniformative chords of zero length
			if (X.size() >= 2)
			{
				// for N segments
				auto compareFunc = [](const std::pair<float, float>& p1, const std::pair<float, float>& p2) { return p1.first < p2.first; };
				auto idx_minX = std::distance(X.begin(), std::min_element(X.begin(), X.end(), compareFunc));
				auto idx_maxX = std::distance(X.begin(), std::max_element(X.begin(), X.end(), compareFunc));
				// left X and right X segments
				auto& e1 = X[idx_minX], & e2 = X[idx_maxX];
				auto x1 = e1.first, y1 = e1.second, x2 = e2.first, y2 = e2.second;
				// save this chord
				auto dist = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2);	// Squared distance
				DA.push_back(dist);
			}
		}

		if (DA.size() > 0)
		{
			// Find the shortest and longest chords (diameters)
			double minD2 = *std::min_element(DA.begin(), DA.end()),
				maxD2 = *std::max_element(DA.begin(), DA.end()),
				min_ = sqrt(minD2),
				max_ = sqrt(maxD2);

			// Save them
			all_D.push_back(min_);
			all_D.push_back(max_);
		}
	}

	// Process the stats
	auto s = ComputeCommonStatistics2(all_D);

	_min = (double)s.min;
	_max = (double)s.max;
	_mean = s.mean;
	_median = s.median;
	_stdev = s.stdev;
	_mode = (double)s.mode;
}

void CaliperMartinFeature::parallel_process(std::vector<int>& roi_labels, std::unordered_map <int, LR>& roiData, int n_threads)
{
	size_t jobSize = roi_labels.size(),
		workPerThread = jobSize / n_threads;

	runParallel(CaliperMartinFeature::parallel_process_1_batch, n_threads, workPerThread, jobSize, &roi_labels, &roiData);
}

void CaliperMartinFeature::extract (LR& r)
{
	CaliperMartinFeature f;
	f.calculate(r);
	f.save_value(r.fvals);
}

void CaliperMartinFeature::parallel_process_1_batch(size_t firstitem, size_t lastitem, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData)
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
		f.calculate(r);
		f.save_value(r.fvals);
	}
}
