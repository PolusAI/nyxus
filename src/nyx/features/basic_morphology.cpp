#include "../environment.h"
#include "../parallel.h"
#include "histogram.h"
#include "basic_morphology.h"
#include "pixel.h"

BasicMorphologyFeatures::BasicMorphologyFeatures()
{
	provide_features({
		AREA_PIXELS_COUNT,
		AREA_UM2,
		ASPECT_RATIO,
		BBOX_XMIN,
		BBOX_YMIN,
		BBOX_WIDTH,
		BBOX_HEIGHT,		
		CENTROID_X,
		CENTROID_Y,
		COMPACTNESS,
		EXTENT,
		MASS_DISPLACEMENT,		
		WEIGHTED_CENTROID_X,
		WEIGHTED_CENTROID_Y
		});
}

void BasicMorphologyFeatures::calculate(LR& r)
{
	double n = r.aux_area;

	// --AREA
	val_AREA_PIXELS_COUNT = n;
	if (theEnvironment.xyRes > 0.0)
			val_AREA_UM2  = n * std::pow(theEnvironment.pixelSizeUm, 2);

	// --CENTROID_XY
	double cen_x = 0.0,
		cen_y = 0.0;
	for (auto& px : r.raw_pixels)
	{
		cen_x += px.x;
		cen_y += px.y;
	}

	val_CENTROID_X = cen_x;
	val_CENTROID_Y = cen_y;
	
	// --COMPACTNESS
	Moments2 mom2;
	for (auto& px : r.raw_pixels)
	{
		double dst = std::sqrt(px.sqdist(cen_x, cen_y));
		mom2.add(dst);
	}
	val_COMPACTNESS = mom2.std() / n;

	//==== Basic morphology :: Bounding box
	val_BBOX_XMIN = r.aabb.get_xmin();
	val_BBOX_YMIN = r.aabb.get_ymin();
	val_BBOX_WIDTH = r.aabb.get_width();
	val_BBOX_HEIGHT = r.aabb.get_height();

	//==== Basic morphology :: Centroids
	val_CENTROID_X = val_CENTROID_Y = 0;
	for (auto& px : r.raw_pixels)
	{
		val_CENTROID_X += px.x;
		val_CENTROID_Y += px.y;
	}
	val_CENTROID_X /= n;
	val_CENTROID_Y /= n;

	//==== Basic morphology :: Weighted centroids
	double x_mass = 0, y_mass = 0, mass = 0;

	for (auto& px : r.raw_pixels)
	{
		// the "+1" is only for compatability with matlab code (where index starts from 1) 
		x_mass = x_mass + (px.x + 1) * px.inten;
		y_mass = y_mass + (px.y + 1) * px.inten;
		mass += px.inten;
	}

	if (mass > 0)
	{
		val_WEIGHTED_CENTROID_X = x_mass / mass;
		val_WEIGHTED_CENTROID_Y = y_mass / mass;
	}
	else
	{
		val_WEIGHTED_CENTROID_X = 0.0;
		val_WEIGHTED_CENTROID_Y = 0.0;
	}

	// --Mass displacement (The distance between the centers of gravity in the gray-level representation of the object and the binary representation of the object.)
	double dx = val_WEIGHTED_CENTROID_X - val_CENTROID_X,
		dy = val_WEIGHTED_CENTROID_Y - val_CENTROID_Y,
		dist = std::sqrt(dx * dx + dy * dy);
	val_MASS_DISPLACEMENT = dist;

	//==== Basic morphology :: Extent
	val_EXTENT = n / r.aabb.get_area();

	//==== Basic morphology :: Aspect ratio
	val_ASPECT_RATIO = r.aabb.get_width() / r.aabb.get_height();
}

void BasicMorphologyFeatures::osized_add_online_pixel(size_t x, size_t y, uint32_t intensity)
{}

void BasicMorphologyFeatures::osized_calculate(LR& r, ImageLoader& imloader)
{
	double n = r.aux_area;

	// --AREA
	val_AREA_PIXELS_COUNT = n;
	if (theEnvironment.xyRes > 0.0)
		val_AREA_UM2 = n * std::pow(theEnvironment.pixelSizeUm, 2);

	// --CENTROID_XY
	double cen_x = 0.0,
		cen_y = 0.0;
	
	for (size_t i = 0; i < r.osized_pixel_cloud.get_size(); i++)	// for (auto& px : r.raw_pixels)
	{
		auto px = r.osized_pixel_cloud.get_at(i);
		cen_x += px.x;
		cen_y += px.y;
	}

	val_CENTROID_X = cen_x;
	val_CENTROID_Y = cen_y;

	// --COMPACTNESS
	Moments2 mom2;
	for (size_t i = 0; i < r.osized_pixel_cloud.get_size(); i++)	// for (auto& px : r.raw_pixels)
	{
		auto px = r.osized_pixel_cloud.get_at(i);
		double dst = std::sqrt(px.sqdist(cen_x, cen_y));
		mom2.add(dst);
	}
	val_COMPACTNESS = mom2.std() / n;

	//==== Basic morphology :: Bounding box
	val_BBOX_XMIN = r.aabb.get_xmin();
	val_BBOX_YMIN = r.aabb.get_ymin();
	val_BBOX_WIDTH = r.aabb.get_width();
	val_BBOX_HEIGHT = r.aabb.get_height();

	//==== Basic morphology :: Centroids
	val_CENTROID_X = val_CENTROID_Y = 0;
	for (size_t i = 0; i < r.osized_pixel_cloud.get_size(); i++)	// for (auto& px : r.raw_pixels)
	{
		auto px = r.osized_pixel_cloud.get_at(i);
		val_CENTROID_X += px.x;
		val_CENTROID_Y += px.y;
	}
	val_CENTROID_X /= n;
	val_CENTROID_Y /= n;

	//==== Basic morphology :: Weighted centroids
	double x_mass = 0, y_mass = 0, mass = 0;

	for (size_t i = 0; i < r.osized_pixel_cloud.get_size(); i++)	// for (auto& px : r.raw_pixels)
	{
		auto px = r.osized_pixel_cloud.get_at(i);
		// the "+1" is only for compatability with matlab code (where index starts from 1) 
		x_mass = x_mass + (px.x + 1) * px.inten;
		y_mass = y_mass + (px.y + 1) * px.inten;
		mass += px.inten;
	}

	if (mass > 0)
	{
		val_WEIGHTED_CENTROID_X = x_mass / mass;
		val_WEIGHTED_CENTROID_Y = y_mass / mass;
	}
	else
	{
		val_WEIGHTED_CENTROID_X = 0.0;
		val_WEIGHTED_CENTROID_Y = 0.0;
	}

	// --Mass displacement (The distance between the centers of gravity in the gray-level representation of the object and the binary representation of the object.)
	double dx = val_WEIGHTED_CENTROID_X - val_CENTROID_X,
		dy = val_WEIGHTED_CENTROID_Y - val_CENTROID_Y,
		dist = std::sqrt(dx * dx + dy * dy);
	val_MASS_DISPLACEMENT = dist;

	//==== Basic morphology :: Extent
	val_EXTENT = n / r.aabb.get_area();

	//==== Basic morphology :: Aspect ratio
	val_ASPECT_RATIO = r.aabb.get_width() / r.aabb.get_height();
}

void BasicMorphologyFeatures::save_value(std::vector<std::vector<double>>& fvals)
{
	fvals[AREA_PIXELS_COUNT][0] = val_AREA_PIXELS_COUNT;
	fvals[AREA_UM2][0] = val_AREA_UM2;
	fvals[ASPECT_RATIO][0] = val_ASPECT_RATIO;
	fvals[BBOX_XMIN][0] = val_BBOX_XMIN;
	fvals[BBOX_YMIN][0] = val_BBOX_YMIN;
	fvals[BBOX_WIDTH][0] = val_BBOX_WIDTH;
	fvals[BBOX_HEIGHT][0] = val_BBOX_HEIGHT;
	fvals[CENTROID_X][0] = val_CENTROID_X;
	fvals[CENTROID_Y][0] = val_CENTROID_Y;
	fvals[COMPACTNESS][0] = val_COMPACTNESS;
	fvals[EXTENT][0] = val_EXTENT;
	fvals[MASS_DISPLACEMENT][0] = val_MASS_DISPLACEMENT;
	fvals[WEIGHTED_CENTROID_X][0] = val_WEIGHTED_CENTROID_X;
	fvals[WEIGHTED_CENTROID_Y][0] = val_WEIGHTED_CENTROID_Y;
}

void BasicMorphologyFeatures::parallel_process(std::vector<int>& roi_labels, std::unordered_map <int, LR>& roiData, int n_threads)
{
	size_t jobSize = roi_labels.size(),
		workPerThread = jobSize / n_threads;

	runParallel(BasicMorphologyFeatures::parallel_process_1_batch, n_threads, workPerThread, jobSize, &roi_labels, &roiData);
}

void BasicMorphologyFeatures::parallel_process_1_batch(size_t firstitem, size_t lastitem, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData)
{
	// Calculate the feature for each batch ROI item 
	for (auto i = firstitem; i < lastitem; i++)
	{
		// Get ahold of ROI's label and cache
		int roiLabel = (*ptrLabels)[i];
		LR& r = (*ptrLabelData)[roiLabel];

		// Skip the ROI if its data is invalid to prevent nans and infs in the output
		if (r.has_bad_data())
			continue;

		// Calculate the feature and save it in ROI's csv-friendly buffer 'fvals'
		BasicMorphologyFeatures f;
		f.calculate(r);
		f.save_value(r.fvals);
	}
}

void BasicMorphologyFeatures::cleanup_instance()
{
	val_AREA_PIXELS_COUNT = 0;
	val_AREA_UM2 = 0;
	val_CENTROID_X = 0;
	val_CENTROID_Y = 0;
	val_COMPACTNESS = 0;
	val_BBOX_XMIN = 0;
	val_BBOX_YMIN = 0;
	val_BBOX_WIDTH = 0;
	val_BBOX_HEIGHT = 0;
	val_WEIGHTED_CENTROID_X = 0;
	val_WEIGHTED_CENTROID_Y = 0;
	val_MASS_DISPLACEMENT = 0;
	val_EXTENT = 0;
	val_ASPECT_RATIO = 0;
}

