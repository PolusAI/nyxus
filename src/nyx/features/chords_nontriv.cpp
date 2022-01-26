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
		MAXCHORDS_MAX,
		MAXCHORDS_MAX_ANG,
		MAXCHORDS_MIN,
		MAXCHORDS_MIN_ANG,
		MAXCHORDS_MEDIAN,
		MAXCHORDS_MEAN,
		MAXCHORDS_MODE,
		MAXCHORDS_STDDEV,
		ALLCHORDS_MAX,
		ALLCHORDS_MAX_ANG,
		ALLCHORDS_MIN,
		ALLCHORDS_MIN_ANG,
		ALLCHORDS_MEDIAN,
		ALLCHORDS_MEAN,
		ALLCHORDS_MODE,
		ALLCHORDS_STDDEV });
}

void ChordsFeature::osized_calculate (LR& r, ImageLoader& imloader)
{
	if (r.osized_pixel_cloud.get_size() == 0)
		return;

	// The center that we'll rotate the ROI around
	size_t cenx = (r.aabb.get_xmin() + r.aabb.get_xmax()) / 2,
		ceny = (r.aabb.get_ymin() + r.aabb.get_ymax()) / 2;

	std::vector<HistoItem> AC, MC; // all chords and max chords
	std::vector<double> ACang, MCang; // corresponding angles

	// Pixel cloud to store rotated ROI
	OutOfRamPixelCloud R;
	R.init (r.label, "rotatedPixCloud");

	// Gather chord lengths at various angles
	double angStep = M_PI / 20.0;
	for (double ang = 0; ang < M_PI; ang += angStep)
	{
		std::vector<int> TC; // chords at angle theta

		AABB aabbRot;	//  bounding box of the rotated cloud
		Rotation::rotate_cloud (
			// inputs
			r.osized_pixel_cloud, cenx, ceny, ang,
			// outputs
			R, aabbRot);

		//ImageMatrix_nontriv im(R);
		WriteImageMatrix_nontriv imRot ("imRot", r.label);
		imRot.allocate (aabbRot.get_width(), aabbRot.get_height());
		imRot.init_with_cloud (R, aabbRot);

		for (int c = 0; c < imRot.get_width(); c++)
		{
			int chlen = imRot.get_chlen (c);
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
}

void ChordsFeature::save_value (std::vector<std::vector<double>>& feature_vals)
{
	feature_vals[MAXCHORDS_MAX][0] = maxchords_max;
	feature_vals[MAXCHORDS_MAX_ANG][0] = maxchords_max_angle;
	feature_vals[MAXCHORDS_MIN][0] = maxchords_min;
	feature_vals[MAXCHORDS_MIN_ANG][0] = maxchords_min_angle;
	feature_vals[MAXCHORDS_MEDIAN][0] = maxchords_median;
	feature_vals[MAXCHORDS_MEAN][0] = maxchords_mean;
	feature_vals[MAXCHORDS_MODE][0] = maxchords_mode;
	feature_vals[MAXCHORDS_STDDEV][0] = maxchords_stddev;
	feature_vals[ALLCHORDS_MAX][0] = allchords_max;
	feature_vals[ALLCHORDS_MAX_ANG][0] = allchords_max_angle;
	feature_vals[ALLCHORDS_MIN][0] = allchords_min;
	feature_vals[ALLCHORDS_MIN_ANG][0] = allchords_min_angle;
	feature_vals[ALLCHORDS_MEDIAN][0] = allchords_median;
	feature_vals[ALLCHORDS_MEAN][0] = allchords_mean;
	feature_vals[ALLCHORDS_MODE][0] = allchords_mode;
	feature_vals[ALLCHORDS_STDDEV][0] = allchords_stddev;
}
