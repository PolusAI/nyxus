#include "../environment.h"
#include "glcm.h"
#include "image_matrix_nontriv.h"

using namespace Nyxus;

void GLCMFeature::osized_calculate (LR& r, const Fsettings& s, ImageLoader&)
{
	// The oversized ROI is fully materialized here regardless, so rebuild its dense image from the
	// disk-backed pixel cloud and reuse the identical in-RAM calculate(). The previous bespoke
	// streaming re-implementation had drifted from calculate() -- it binned intensities with
	// to_grayscale() instead of TextureFeature::bin_intensities(), so co-occurrence indices could
	// exceed the matrix sized by the trivial binning and the lookup threw "invalid vector
	// subscript". Delegating guarantees trivial == out-of-core.
	r.rebuild_aux_image_matrix_from_cloud();

	calculate (r, s);
}
