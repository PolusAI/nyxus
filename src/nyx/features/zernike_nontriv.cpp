//
//	Adaptation of Wind-Charm's adaptation of Ilya Goldberg's adaptation 
// of Michael Boland's mb_Znl.c Zernike polynomial based feature extraction (09 Dec 1998)
//                                                                          
//  Revisions:                                                              
//  9-1-04 Tom Macura <tmacura@nih.gov> modified to make the code ANSI C    
//         and work with included complex arithmetic library from           
//         Numerical Recepies in C instead of using the system's C++ STL    
//         Libraries.                                                       
//                                                                          
//  1-29-06 Lior Shamir <shamirl (-at-) mail.nih.gov> modified "factorial"  
//  to a loop, replaced input structure with ImageMatrix class.             
//  2011-04-25 Ilya Goldberg. Optimization due to this function accounting  
//    for 35% of the total wndchrm run-time.  Now ~4x faster.               
//  2012-12-13 Ilya Goldberg. Added 10x faster mb_zernike2D_2               
//    the feature values this algorithm produces are not the same as before 
//    however, the weights assigned to these features in classification     
//    are as good or better than mb_zernike2D                               
//

#define _USE_MATH_DEFINES 	// For M_PI, etc.
#include <complex>
#include <cmath>
#include <cfloat> // Has definition of DBL_EPSILON
#include <assert.h>
#include <stdio.h>
#include "specfunc.h" 

#include "image_matrix.h" 

#include <unordered_map>
#include "../roi_cache.h"
#include "zernike.h"

void ZernikeFeature::osized_calculate (LR& r, const Fsettings& s, ImageLoader&)
{
	// The oversized ROI is fully materialized here regardless, so rebuild its dense image from the
	// disk-backed pixel cloud and reuse the identical in-RAM calculate(). The previous bespoke
	// streaming re-implementation agreed on ordinary ROIs but lacked calculate()'s constant-ROI
	// guard, so on a degenerate ROI it produced Zernike moments where the in-RAM path reports the
	// soft-NAN sentinel. Delegating guarantees trivial == out-of-core.
	r.rebuild_aux_image_matrix_from_cloud();

	calculate (r, s);
}
