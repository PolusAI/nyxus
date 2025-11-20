#define _USE_MATH_DEFINES	// For M_PI, etc.
#include <cmath>
#include "../feature_method.h"
#include "chords.h"
#include "histogram.h"
#include "image_matrix_nontriv.h"
#include "rotation.h"

ChordsFeature::ChordsFeature() : FeatureMethod("ChordsFeature")
{
	provide_features({ 
		Nyxus::Feature2D::MAXCHORDS_MAX,
		Nyxus::Feature2D::MAXCHORDS_MAX_ANG,
		Nyxus::Feature2D::MAXCHORDS_MIN,
		Nyxus::Feature2D::MAXCHORDS_MIN_ANG,
		Nyxus::Feature2D::MAXCHORDS_MEDIAN,
		Nyxus::Feature2D::MAXCHORDS_MEAN,
		Nyxus::Feature2D::MAXCHORDS_MODE,
		Nyxus::Feature2D::MAXCHORDS_STDDEV,
		Nyxus::Feature2D::ALLCHORDS_MAX,
		Nyxus::Feature2D::ALLCHORDS_MAX_ANG,
		Nyxus::Feature2D::ALLCHORDS_MIN,
		Nyxus::Feature2D::ALLCHORDS_MIN_ANG,
		Nyxus::Feature2D::ALLCHORDS_MEDIAN,
		Nyxus::Feature2D::ALLCHORDS_MEAN,
		Nyxus::Feature2D::ALLCHORDS_MODE,
		Nyxus::Feature2D::ALLCHORDS_STDDEV });
}

void ChordsFeature::osized_calculate (LR& r, const Fsettings& stng, ImageLoader& imloader)
{
	// Center
	double cenx = (r.aabb.get_xmin() + r.aabb.get_xmax()) / 2.0,
		ceny = (r.aabb.get_ymin() + r.aabb.get_ymax()) / 2.0;

	// All chords and max chords
	std::vector<HistoItem> AC, MC; // lengths
	std::vector<double> ACang, MCang; // angles

	// Gather chord lengths at various angles
	double angStep = M_PI / double(n_angle_segments);
	for (double ang = 0; ang < M_PI; ang += angStep)
	{
		// Chords at angle theta
		std::vector<int> TC;

		// Container for rotated pixel cloud
		OutOfRamPixelCloud R;
		R.init (r.label, "ChordsFeature-osized_calculate-R");

		// Rotate the cloud and save as image matrix
		Rotation::rotate_cloud_NT (
			// input
			r.raw_pixels_NT, cenx, ceny, ang, 
			// output
			R);
		WriteImageMatrix_nontriv im ("im", r.label);
		im.allocate_from_cloud (R, r.aabb, true);	// The subsequent analysis is not AABB-critical so we are good to use the unrotated r.aabb

		// Explore column chords, with step to keep the timing under control at huge ROIs
		auto rotW = im.get_width();
		int step = rotW >= 2 * n_side_segments ? rotW / n_side_segments : 1;
		for (int col = 0; col < rotW; col += step)
		{
			int chlen = im.get_chlen(col);
			if (chlen > 0)
			{
				TC.push_back(chlen);
				AC.push_back(chlen);
				ACang.push_back(ang);
			}
		}

		// Save the longest chord's length and angle
		if (TC.size() > 0)
		{
			auto maxChlen = *(std::max_element(TC.begin(), TC.end()));
			MC.push_back(maxChlen);
			MCang.push_back(ang);
		}
	}

	// Analyze max chords
	if (MC.size() == 0)
		return;	// Nothing to analyze, return

	Moments2 mom2;
	for (auto chlen : MC)
		mom2.add(chlen);

	maxchords_max = mom2.max__();
	maxchords_min = mom2.min__();
	maxchords_mean = mom2.mean();
	maxchords_stddev = mom2.std();

	TrivialHistogram histo;
	histo.initialize_uniques(MC);
	maxchords_mode = histo.get_mode();
	maxchords_median = histo.get_median();

	auto iteMin = std::min_element(MC.begin(), MC.end());
	auto idxmin = std::distance(MC.begin(), iteMin);
	maxchords_min_angle = MCang[idxmin];

	auto iteMax = std::max_element(MC.begin(), MC.end());
	auto idxmax = std::distance(MC.begin(), iteMin);
	maxchords_max_angle = MCang[idxmax];

	// Analyze all chords
	mom2.reset();
	for (auto chlen : AC)
		mom2.add(chlen);

	allchords_max = mom2.max__();
	allchords_min = mom2.min__();
	allchords_mean = mom2.mean();
	allchords_stddev = mom2.std();

	histo.initialize_uniques(MC);
	allchords_mode = histo.get_mode();
	allchords_median = histo.get_median();

	iteMin = std::min_element(AC.begin(), AC.end());
	idxmin = std::distance(AC.begin(), iteMin);
	allchords_min_angle = ACang[idxmin];

	iteMax = std::max_element(AC.begin(), AC.end());
	idxmax = std::distance(AC.begin(), iteMin);
	allchords_max_angle = ACang[idxmax];
}

void ChordsFeature::save_value (std::vector<std::vector<double>>& feature_vals)
{
	feature_vals[(int)Nyxus::Feature2D::MAXCHORDS_MAX][0] = maxchords_max;
	feature_vals[(int)Nyxus::Feature2D::MAXCHORDS_MAX_ANG][0] = maxchords_max_angle;
	feature_vals[(int)Nyxus::Feature2D::MAXCHORDS_MIN][0] = maxchords_min;
	feature_vals[(int)Nyxus::Feature2D::MAXCHORDS_MIN_ANG][0] = maxchords_min_angle;
	feature_vals[(int)Nyxus::Feature2D::MAXCHORDS_MEDIAN][0] = maxchords_median;
	feature_vals[(int)Nyxus::Feature2D::MAXCHORDS_MEAN][0] = maxchords_mean;
	feature_vals[(int)Nyxus::Feature2D::MAXCHORDS_MODE][0] = maxchords_mode;
	feature_vals[(int)Nyxus::Feature2D::MAXCHORDS_STDDEV][0] = maxchords_stddev;
	feature_vals[(int)Nyxus::Feature2D::ALLCHORDS_MAX][0] = allchords_max;
	feature_vals[(int)Nyxus::Feature2D::ALLCHORDS_MAX_ANG][0] = allchords_max_angle;
	feature_vals[(int)Nyxus::Feature2D::ALLCHORDS_MIN][0] = allchords_min;
	feature_vals[(int)Nyxus::Feature2D::ALLCHORDS_MIN_ANG][0] = allchords_min_angle;
	feature_vals[(int)Nyxus::Feature2D::ALLCHORDS_MEDIAN][0] = allchords_median;
	feature_vals[(int)Nyxus::Feature2D::ALLCHORDS_MEAN][0] = allchords_mean;
	feature_vals[(int)Nyxus::Feature2D::ALLCHORDS_MODE][0] = allchords_mode;
	feature_vals[(int)Nyxus::Feature2D::ALLCHORDS_STDDEV][0] = allchords_stddev;
}
