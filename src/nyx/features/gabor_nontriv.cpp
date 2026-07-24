#define _USE_MATH_DEFINES	// For M_PI, etc.
#include <cmath>
#include "gabor.h"
#include "image_matrix_nontriv.h"

using namespace Nyxus;

GaborFeature::GaborFeature() : FeatureMethod("GaborFeature") 
{
    provide_features ({ Feature2D::GABOR });
}

void GaborFeature::osized_calculate (LR& r, const Fsettings& s, ImageLoader&)
{
	// The oversized ROI is fully materialized here regardless, so rebuild its dense image from the
	// disk-backed pixel cloud and reuse the identical in-RAM calculate(). The previous bespoke
	// streaming re-implementation never assigned its 'originalScore' baseline (it stayed 0), so
	// every frequency was divided by the tiny-number floor and came back astronomically large
	// instead of a ratio in [0,1]. Delegating guarantees trivial == out-of-core.
	r.rebuild_aux_image_matrix_from_cloud();

	calculate (r, s);
}
