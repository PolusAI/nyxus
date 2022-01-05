#define _USE_MATH_DEFINES	// For M_PI, etc.
#include <algorithm>
#include <cmath>
#include <memory>
#include <unordered_map>
#include <unordered_set> 
#include <algorithm>
#include <iostream>
#include <chrono>
#include <thread>
#include <future>
#include <sstream>
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
#include "features/hexagonality_and_polygonality.h"
#include "features/ngtdm.h"
#include "features/image_moments.h"
#include "features/moments.h"
#include "features/neighbors.h"
#include "features/particle_metrics.h"
#include "features/roi_radius.h"
#include "features/zernike.h"
#include "helpers/timing.h"
#include "parallel.h"
#include "feature_method.h"

namespace Nyxus
{
	// This function should be called once after scanning ROI pixels is finished
	void reduce_by_roi (int nThr, int min_online_roi_size)
	{
		// Prepare requested features as a vector sorted by number of references
		theFeatureMgr.sort_by_num_dependencies();
		const std::vector<FeatureMethod*>& RF = theFeatureMgr.get_requested_features();

		for (auto& ld : roiData)
		{
			auto l = ld.first;		// ROI label code
			LR& r = ld.second;	// ROI info cache structure

			// Calculate features for this ROI
			for (FeatureMethod* f : RF)
			{
				// Calculate the feature by either scanning its pixels from the image (large ROIs) or using the pixels cache (small ROIs)
				f->calculate (r);

				// Dispose helper objects in ROI's cache that can be displsed
				for (auto & cacheItem : r.CachedObjects)
					if (theFeatureMgr.roi_cache_item_needed (cacheItem) == false)
						r.recycle_aux_obj(cacheItem);
			}
		}

		//==== Neighbors are reduced
		if (Neighbor_features::required(theFeatureSet))
		{
			STOPWATCH("Neighbors/Neighbors/N/#FF69B4", "\t=");
			Neighbor_features::reduce(theEnvironment.get_pixel_distance());
		}
	}
}

