#define _USE_MATH_DEFINES	// For M_PI, etc.
#include "../environment.h"
#include "../parallel.h"
#include "histogram.h"
#include "basic_morphology.h"
#include "pixel.h"

using namespace Nyxus;

bool BasicMorphologyFeatures::required(const FeatureSet& fs) 
{
	return fs.anyEnabled (BasicMorphologyFeatures::featureset);
}

BasicMorphologyFeatures::BasicMorphologyFeatures(): FeatureMethod("BasicMorphologyFeatures")
{
	provide_features (BasicMorphologyFeatures::featureset);
}

void BasicMorphologyFeatures::calculate(LR& r)
{
	double n = r.aux_area;

	// --AREA
	val_AREA_PIXELS_COUNT = n;
	if (theEnvironment.xyRes > 0.0)
			val_AREA_UM2  = n * std::pow(theEnvironment.pixelSizeUm, 2);

	// --DIAMETER_EQUAL_AREA
	val_DIAMETER_EQUAL_AREA = double(val_AREA_PIXELS_COUNT) / M_PI * 4.0;

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
		// the "+1" is only for compatibility with matlab code (where index starts from 1) 
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

void BasicMorphologyFeatures::osized_add_online_pixel(size_t x, size_t y, uint32_t intensity) {} // Not providing online calculation for these group of features

void BasicMorphologyFeatures::osized_calculate(LR& r, ImageLoader& imloader)
{
	double n = r.aux_area;

	// --AREA
	val_AREA_PIXELS_COUNT = n;
	if (theEnvironment.xyRes > 0.0)
		val_AREA_UM2 = n * std::pow(theEnvironment.pixelSizeUm, 2);

	// --DIAMETER_EQUAL_AREA
	val_DIAMETER_EQUAL_AREA = double(val_AREA_PIXELS_COUNT) / M_PI * 4.0;

	// --CENTROID_XY
	double cen_x = 0.0,
		cen_y = 0.0;
	
	for (size_t i = 0; i < r.raw_pixels_NT.size(); i++)	// for (auto& px : r.raw_pixels)
	{
		auto px = r.raw_pixels_NT.get_at(i);
		cen_x += px.x;
		cen_y += px.y;
	}

	val_CENTROID_X = cen_x;
	val_CENTROID_Y = cen_y;

	// --COMPACTNESS
	Moments2 mom2;
	for (size_t i = 0; i < r.raw_pixels_NT.size(); i++)	// for (auto& px : r.raw_pixels)
	{
		auto px = r.raw_pixels_NT.get_at(i);
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
	for (size_t i = 0; i < r.raw_pixels_NT.size(); i++)	// for (auto& px : r.raw_pixels)
	{
		auto px = r.raw_pixels_NT.get_at(i);
		val_CENTROID_X += px.x;
		val_CENTROID_Y += px.y;
	}
	val_CENTROID_X /= n;
	val_CENTROID_Y /= n;

	//==== Basic morphology :: Weighted centroids
	double x_mass = 0, y_mass = 0, mass = 0;

	for (size_t i = 0; i < r.raw_pixels_NT.size(); i++)	// for (auto& px : r.raw_pixels)
	{
		auto px = r.raw_pixels_NT.get_at(i);
		// the "+1" is only for compatibility with matlab code (where index starts from 1) 
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
	fvals[(int)Feature2D::AREA_PIXELS_COUNT][0] = val_AREA_PIXELS_COUNT;
	fvals[(int)Feature2D::AREA_UM2][0] = val_AREA_UM2;
	fvals[(int)Feature2D::ASPECT_RATIO][0] = val_ASPECT_RATIO;
	fvals[(int)Feature2D::BBOX_XMIN][0] = val_BBOX_XMIN;
	fvals[(int)Feature2D::BBOX_YMIN][0] = val_BBOX_YMIN;
	fvals[(int)Feature2D::BBOX_WIDTH][0] = val_BBOX_WIDTH;
	fvals[(int)Feature2D::BBOX_HEIGHT][0] = val_BBOX_HEIGHT;
	fvals[(int)Feature2D::CENTROID_X][0] = val_CENTROID_X;
	fvals[(int)Feature2D::CENTROID_Y][0] = val_CENTROID_Y;
	fvals[(int)Feature2D::COMPACTNESS][0] = val_COMPACTNESS;
	fvals[(int)Feature2D::DIAMETER_EQUAL_AREA][0] = val_DIAMETER_EQUAL_AREA;
	fvals[(int)Feature2D::EXTENT][0] = val_EXTENT;
	fvals[(int)Feature2D::MASS_DISPLACEMENT][0] = val_MASS_DISPLACEMENT;
	fvals[(int)Feature2D::WEIGHTED_CENTROID_X][0] = val_WEIGHTED_CENTROID_X;
	fvals[(int)Feature2D::WEIGHTED_CENTROID_Y][0] = val_WEIGHTED_CENTROID_Y;
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
		//if (r.has_bad_data())
		//	continue;

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
	val_DIAMETER_EQUAL_AREA = 0;
}

