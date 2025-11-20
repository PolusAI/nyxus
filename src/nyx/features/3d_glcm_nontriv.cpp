#include "../environment.h"
#include "3d_glcm.h"
#include "image_matrix_nontriv.h"

using namespace Nyxus;

void D3_GLCM_feature::osized_calculate (LR& r, const Fsettings& s, ImageLoader&)
{
	calculate (r, s);
}

