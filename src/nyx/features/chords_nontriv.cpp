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

void ChordsFeature::osized_calculate (LR& r, const Fsettings& stng, ImageLoader&)
{
	// Materialize the ROI's pixel cloud from the disk-backed one and reuse the identical in-RAM
	// calculate(). The previous bespoke streaming re-implementation stepped over columns rather
	// than scanning every one ("to keep the timing under control at huge ROIs"), so it sampled
	// shorter chords and reported max/min/median chord lengths that disagreed with the trivial
	// path. Delegating guarantees trivial == out-of-core.
	r.rebuild_raw_pixels_from_cloud();

	calculate (r, stng);
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
