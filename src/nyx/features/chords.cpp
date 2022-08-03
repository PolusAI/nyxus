#define _USE_MATH_DEFINES	// For M_PI, etc.
#include <cmath>

#include <algorithm>
#include "aabb.h"
#include "chords.h"
#include "histogram.h"
#include "image_matrix.h"
#include "rotation.h"

void ChordsFeature::calculate (LR & r)
{
	const std::vector<Pixel2>& raw_pixels = r.raw_pixels;
	const AABB& bb = r.aabb;
	double cenx = (bb.get_xmin() + bb.get_xmax()) / 2.0,
		ceny = (bb.get_ymin() + bb.get_ymax()) / 2.0;

	auto n = raw_pixels.size();
	std::vector<Pixel2> R;	// raw_pixels rotated 
	R.resize(n, {0,0,0});

	std::vector<HistoItem> AC, MC; // all chords and max chords
	std::vector<double> ACang, MCang; // corresponding angles

	// Gather chord lengths at various angles
	double angStep = M_PI / 20.0;
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
	histo.initialize_uniques (MC);
	maxchords_mode = histo.get_mode();
	maxchords_median = histo.get_median();
	
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

void ChordsFeature::process_1_batch (size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData)
{
	for (auto i = start; i < end; i++)
	{
		int lab = (*ptrLabels)[i];
		LR& r = (*ptrLabelData)[lab];

		if (r.has_bad_data())
			continue;

		ChordsFeature f;
		f.calculate(r);
		f.save_value(r.fvals);
	}
}

