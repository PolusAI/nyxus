#define _USE_MATH_DEFINES	// For M_PI, etc.
#include <cmath>

#include <algorithm>
#include "aabb.h"
#include "chords.h"
#include "histogram.h"
#include "image_matrix.h"
#include "rotation.h"

void ChordsFeature::calculate (LR & r, const Fsettings& s)
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
		std::vector<Pixel2> R;
		R.resize(r.raw_pixels.size(), { 0,0,0 });

		// Rotate the cloud and save as image matrix
		Rotation::rotate_cloud (r.raw_pixels, cenx, ceny, ang, R);
		ImageMatrix im (R);

		// Explore column chords, with step to keep the timing under control at huge ROIs
		int step = im.width >= 2 * n_side_segments ? im.width / n_side_segments : 1;
		for (int col = 0; col < im.width; col += step)
		{
			int chlen = im.get_chlen (col);
			if (chlen > 0)
			{
				TC.push_back (chlen);
				AC.push_back (chlen);
				ACang.push_back (ang);
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

void ChordsFeature::extract (LR& r, const Fsettings& s)
{
	ChordsFeature f;
	f.calculate (r, s);
	f.save_value (r.fvals);
}

void ChordsFeature::process_1_batch (size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData, const Fsettings & s, const Dataset & _)
{
	for (auto i = start; i < end; i++)
	{
		int lab = (*ptrLabels)[i];
		LR& r = (*ptrLabelData)[lab];

		if (r.has_bad_data())
			continue;

		ChordsFeature f;
		f.calculate (r, s);
		f.save_value (r.fvals);
	}
}

