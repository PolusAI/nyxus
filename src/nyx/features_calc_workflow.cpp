#define _USE_MATH_DEFINES	// For M_PI, etc.
#include <algorithm>
#include <cmath>
#include <future>
#include <iostream>
#include <memory>
#include <thread>
#include <sstream>
#include <unordered_map>
#include <unordered_set> 
#include "environment.h"
#include "globals.h"
#include "features/chords.h"
#include "features/ellipse_fitting.h"
#include "features/euler_number.h"
#include "features/circle.h"
#include "features/extrema.h"
#include "features/fractal_dim.h"
#include "features/erosion.h"
#include "features/radial_distribution.h"
#include "features/gabor.h"
#include "features/geodetic_len_thickness.h"
#include "features/glcm.h"
#include "features/glrlm.h"
#include "features/glszm.h"
#include "features/gldm.h"
#include "features/hexagonality_polygonality.h"
#include "features/ngtdm.h"
#include "features/2d_geomoments.h"
#include "features/moments.h"
#include "features/neighbors.h"
#include "features/caliper.h"
#include "features/roi_radius.h"
#include "features/zernike.h"
#include "helpers/timing.h"

namespace Nyxus
{
	// Preallocates the intensely accessed main containers
	void init_slide_rois (Uniqueids & L, Roidata & D)
	{
		L.reserve (100 * 1000);
		D.reserve (100 * 1000);
	}

	// Resets the main containers
	void clear_slide_rois (Uniqueids & L, Roidata & D)
	{
		// Reset per-image buffers
		L.clear();
		D.clear();
	}

	void init_label_record_hierarchical (LR& r, const std::string& segFile, const std::string& intFile, int x, int y, int label, PixIntens intensity)
	{
		// Initialize basic counters
		r.aux_area = 1;
		r.aux_min = r.aux_max = intensity;
		r.init_aabb (x,y);

		// Cache the ROI label
		r.label = label;
	}

	void init_label_record_3 (LR& r, int x, int y, PixIntens intensity)
	{
		// Initialize basic counters
		r.aux_area = 1;
		r.aux_min = r.aux_max = intensity;
		r.init_aabb(x, y);
	}

	void init_label_record_3D (
		/*out*/ 
		LR & r, 
		/*in*/
		int x, int y, int z, int label, PixIntens intensity)
	{
		// Initialize basic counters
		r.aux_area = 1;
		r.aux_min = r.aux_max = intensity;
		r.init_aabb_3D (x, y, z);

		// Cache the ROI label
		r.label = label;
	}

	void update_label_record_2 (LR& lr, int x, int y, int label, PixIntens intensity)
	{
		// Per-ROI 
		lr.aux_area++;

		lr.aux_min = std::min(lr.aux_min, intensity);
		lr.aux_max = std::max(lr.aux_max, intensity);

		lr.update_aabb (x,y);
	}

	void update_label_record_3 (LR& lr, int x, int y, PixIntens intensity)
	{
		lr.aux_area++;
		lr.aux_min = std::min (lr.aux_min, intensity);
		lr.aux_max = std::max (lr.aux_max, intensity);
		lr.update_aabb (x,y);
	}

	void update_label_record_3D (LR& lr, int x, int y, int z, int label, PixIntens intensity)
	{
		// Per-ROI 
		lr.aux_area++;

		lr.aux_min = std::min(lr.aux_min, intensity);
		lr.aux_max = std::max(lr.aux_max, intensity);

		lr.update_aabb_3D (x, y, z);
	}
}