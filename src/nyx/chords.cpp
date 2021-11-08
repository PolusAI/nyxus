#define _USE_MATH_DEFINES	// For M_PI, etc.
#include <cmath>

#include <algorithm>
#include "chords.h"
#include "histogram.h"
#include "image_matrix.h"
#include "rotation.h"

void Chords::initialize(const std::vector<Pixel2>& raw_pixels, const AABB& bb, const double cenx, const double ceny)
{
	auto n = raw_pixels.size();
	std::vector<Pixel2> R;	// raw_pixels rotated 
	R.resize(n, {0,0,0});

	std::vector<HistoItem> AC, MC; // all chords and max chords
	std::vector<double> ACang, MCang; // corresponding angles

	// Gather chord lengths at various angles
	double angStep = M_PI / 20.;
	for (double ang = 0; ang < M_PI; ang += angStep)
	{
		std::vector<int> TC; // chords at angle theta

		Rotation::rotate_cloud (raw_pixels, cenx, ceny, ang, R);
		ImageMatrix im (R);
		for (int c = 0; c < im.width; c++)
		{
			int chlen = im.get_chlen(c);
			if (chlen > 0)
			{
				TC.push_back(chlen);
				AC.push_back(chlen);
				ACang.push_back(ang);
			}
		}

		if (TC.size() > 0)
		{
			auto maxChlen = *(std::max_element(TC.begin(), TC.end()));
			MC.push_back(maxChlen);
			MCang.push_back(ang);
		}
	}

	// Analyze max chords
	if (MC.size() == 0)
		return;

	Moments2 mom2;
	for (auto chlen : MC)
		mom2.add(chlen);
	
	maxchords_max = mom2.max__();
	maxchords_min = mom2.min__();
	maxchords_mean = mom2.mean();
	maxchords_stddev = mom2.std();

	TrivialHistogram histo;
	histo.initialize ((HistoItem)maxchords_min, (HistoItem)maxchords_max, MC);
	auto [median_, mode_, p01_, p10_, p25_, p75_, p90_, p99_, iqr_, rmad_, entropy_, uniformity_] = histo.get_stats();
	maxchords_mode = mode_;
	maxchords_median = median_;
	
	auto iteMin = std::min_element (MC.begin(), MC.end());
	auto idxmin = std::distance (MC.begin(), iteMin);
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

	histo.initialize ((HistoItem)allchords_min, (HistoItem)allchords_max, MC);
	std::tie(median_, mode_, p01_, p10_, p25_, p75_, p90_, p99_, iqr_, rmad_, entropy_, uniformity_) = histo.get_stats();
	allchords_mode = mode_;
	allchords_median = median_;

	iteMin = std::min_element(AC.begin(), AC.end());
	idxmin = std::distance(AC.begin(), iteMin);
	allchords_min_angle = ACang[idxmin];

	iteMax = std::max_element(AC.begin(), AC.end());
	idxmax = std::distance(AC.begin(), iteMin);
	allchords_max_angle = ACang[idxmax];
}

// Returns
// --------
//	max
//	min
//	median
//	mean
//	mode
//	stddev
//	min_angle
//	max_angle
//
std::tuple<double, double, double, double, double, double, double, double> Chords::get_maxchords_stats()
{
	return { maxchords_max,
		maxchords_min,
		maxchords_median,
		maxchords_mean,
		maxchords_mode,
		maxchords_stddev,
		maxchords_min_angle,
		maxchords_max_angle };
}

// Returns
// --------
//	max
//	min
//	median
//	mean
//	mode
//	stddev
//	min_angle
//	max_angle
//
std::tuple<double, double, double, double, double, double, double, double> Chords::get_allchords_stats()
{
	return { allchords_max,
		allchords_min,
		allchords_median,
		allchords_mean,
		allchords_mode,
		allchords_stddev,
		allchords_min_angle,
		allchords_max_angle };
}
