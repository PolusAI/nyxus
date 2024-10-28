#include <algorithm>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>
#include "environment.h"
#include "featureset.h"
#include "features/basic_morphology.h"
#include "features/chords.h"
#include "features/convex_hull.h"
#include "features/erosion.h"
#include "features/caliper.h"
#include "features/circle.h"
#include "features/ellipse_fitting.h"
#include "features/euler_number.h"
#include "features/extrema.h"
#include "features/fractal_dim.h"
#include "features/geodetic_len_thickness.h"
#include "features/gabor.h"
#include "features/glcm.h"
#include "features/gldm.h"
#include "features/gldzm.h"
#include "features/glrlm.h"
#include "features/glszm.h"
#include "features/2d_geomoments.h"
#include "features/intensity.h"
#include "features/neighbors.h"
#include "features/ngldm.h"
#include "features/ngtdm.h"
#include "features/roi_radius.h"
#include "helpers/helpers.h"
#include "helpers/system_resource.h"
#include "helpers/timing.h"
#include "version.h"

using namespace Nyxus;

bool Environment::spellcheck_raw_featurelist(const std::string& comma_separated_fnames, std::vector<std::string>& fnames)
{
	fnames.clear();

	// Missing user input about desired features ?
	if (comma_separated_fnames.length() == 0)
	{
		if (dim() == 2)
		{
			auto gname = Nyxus::theFeatureSet.findGroupNameByCode(Fgroup2D::FG2_ALL);
			std::cout << "Warning: no features specified, defaulting to " << gname << "\n";
			fnames.push_back(gname);
			return true;
		}

		if (dim() == 3)
		{
			auto gname = Nyxus::theFeatureSet.findGroupNameByCode(Fgroup3D::FG3_ALL);
			std::cout << "Warning: no features specified, defaulting to " << gname << "\n";
			fnames.push_back(gname);
			return true;
		}

		// unexpected dimensionality
		std::cout << "Error: unsupported dimensionality " << dim() << " @ " << __FILE__ << ':' << __LINE__ << "\n";
		return false;

	}

	// Chop the CS-list
	bool success = true;
	std::vector<std::string> strings;
	parse_delimited_string(comma_separated_fnames, ",", strings);

	// Check names of features and feature-groups
	for (const std::string& s : strings)
	{
		// Forgive user's typos of consecutive commas e.g. MIN,MAX,,MEDIAN
		if (s.empty())
			continue;

		auto s_uppr = Nyxus::toupper(s);
		if (dim() == 2)
		{
			// Is feature found among 2D features?
			int fg; // signed Fgroup2D
			bool gnameExists = theFeatureSet.find_2D_GroupByString (s_uppr, fg);

			// Intercept an error: 2D feature group exists but requested in the non-2D mode
			if (gnameExists && dim() != 2)
			{
				success = false;
				std::cerr << "Error: 2D feature group '" << s << "' in non-2D mode\n";
				continue;
			}

			// If a group is found, register it
			if (gnameExists)
			{
				fnames.push_back(s_uppr);
				continue;
			}

			int fcode;	// signed Feature2D
			bool fnameExists = theFeatureSet.find_2D_FeatureByString (s_uppr, fcode);

			// 2D feature group requested on a non-2D mode ?
			if (fnameExists && dim() != 2)
			{
				success = false;
				std::cerr << "Error: 2D feature '" << s << "' in non-2D mode\n";
				continue;
			}

			if (!fnameExists)
			{
				success = false;
				std::cerr << "Error: expecting '" + s + "' to be a proper 2D feature name or feature file path\n";
			}
			else
				fnames.push_back(s_uppr);
		} // 2D

		if (dim() == 3)
		{
			// Is feature found among 2D features?
			Fgroup3D afg;
			bool gnameExists = theFeatureSet.find_3D_GroupByString (s_uppr, afg);

			// Intercept an error: 3D feature group exists but requested in the non-3D mode
			if (gnameExists && dim() != 3)
			{
				success = false;
				std::cerr << "Error: 3D feature group '" << s << "' in non-3D mode\n";
				continue;
			}

			// If a group is found, register it
			if (gnameExists)
			{
				fnames.push_back(s_uppr);
				continue;
			}

			Feature3D af;
			bool fnameExists = theFeatureSet.find_3D_FeatureByString (s_uppr, af);

			// 3D feature group requested on a non-3D mode ?
			if (fnameExists && dim() != 3)
			{
				success = false;
				std::cerr << "Error: 3D feature '" << s << "' in non-3D mode\n";
				continue;
			}

			if (!fnameExists)
			{
				success = false;
				std::cerr << "Error: expecting '" << s << "' to be a proper 3D feature name or feature file path\n";
			}
			else
				fnames.push_back(s_uppr);
		} // 3D
	}

	// Missing user input due to commas and no features specified ?
	if (success && fnames.size() == 0)
	{
		success = false;
		std::cerr << "Error: no features specified\n";
	}

	// Show help on available features if necessary
	if (!success)
		theEnvironment.show_featureset_help();

	return success;
}

// Returns:
//		true if s is recognized as a group name
//		false if not recognized (s may be an individual feature name)
//
bool Environment::expand_2D_featuregroup (const std::string & s)
{
	int fgcode;
	if (! Nyxus::theFeatureSet.find_2D_GroupByString(s, fgcode))
		return false; // 's' is a feature name
	bool enable = true;
	if (fgcode < 0)
	{
		fgcode = -fgcode;
		enable = false;
	}

	if ((Fgroup2D)fgcode == Fgroup2D::FG2_ALL)
	{
		Nyxus::theFeatureSet.enableAll (enable);
		return true; 
	}

	if ((Fgroup2D)fgcode == Fgroup2D::FG2_WHOLESLIDE)
	{
		Nyxus::theFeatureSet.enableAll(enable);

		// Handle whole-slide mode differently: disable features irrelevant to this mode (shape, neighbors, etc)
		std::cout << box_text(
			"Activating whole slide (aka single-ROI) mode\n"
			"Using GPU is advised!\n"
			"ATTENTION: disabling inappplicable and time-sonsuming features:\n"
			" - morphological features\n"
			" - neighbor features\n"
			" - GLDZM");

		theFeatureSet.disableFeatures (BasicMorphologyFeatures::featureset);
		theFeatureSet.disableFeatures (EnclosingInscribingCircumscribingCircleFeature::featureset);
		// enabling ContourFeature (builds a special trivial wholeslide contour)
		theFeatureSet.disableFeatures (ConvexHullFeature::featureset);				// depends on ContourFeature
		theFeatureSet.disableFeatures (FractalDimensionFeature::featureset);		// depends on ContourFeature
		theFeatureSet.disableFeatures (GeodeticLengthThicknessFeature::featureset);	// depends on ContourFeature
		theFeatureSet.disableFeatures (NeighborsFeature::featureset);				// no neighbors for whole slide; depends on ContourFeature
		theFeatureSet.disableFeatures (RoiRadiusFeature::featureset);				// depends on ContourFeature
		theFeatureSet.disableFeatures (EllipseFittingFeature::featureset);
		theFeatureSet.disableFeatures (EulerNumberFeature::featureset);
		theFeatureSet.disableFeatures (ExtremaFeature::featureset);
		theFeatureSet.disableFeatures (ErosionPixelsFeature::featureset);
		theFeatureSet.disableFeatures (CaliperFeretFeature::featureset);
		theFeatureSet.disableFeatures (CaliperMartinFeature::featureset);
		theFeatureSet.disableFeatures (CaliperNassensteinFeature::featureset);
		theFeatureSet.disableFeatures (ChordsFeature::featureset);

		// enabling GaborFeature
		// enabling ImageMomentsFeature
		// 
		// enabling GLCMFeature
		// enabling GLDMFeature
		theFeatureSet.disableFeatures(GLDZMFeature::featureset);	// costs about 82 %
		// enabling GLRLMFeature 
		// enabling GLSZMFeature
		// enabling NGLDMfeature
		// enabling NGTDMFeature

		return true;
	}

	if ((Fgroup2D) fgcode == Fgroup2D::FG2_EASY)
	{
		theFeatureSet.enableAll();
		theFeatureSet.disableFeatures(GaborFeature::featureset);
		theFeatureSet.disableFeatures(GLCMFeature::featureset);
		theFeatureSet.disableFeatures(Imoms2D_feature::featureset);
		theFeatureSet.disableFeatures(Smoms2D_feature::featureset);
		return true;
	}

	if ((Fgroup2D) fgcode == Fgroup2D::FG2_NEIG)
	{
		Nyxus::theFeatureSet.enableFeatures (NeighborsFeature::featureset, enable);
		return true;
	}

	if ((Fgroup2D) fgcode == Fgroup2D::FG2_INTENSITY)
	{
		Nyxus::theFeatureSet.enableFeatures (PixelIntensityFeatures::featureset, enable);
		return true;
	}
	if ((Fgroup2D) fgcode == Fgroup2D::FG2_MORPHOLOGY)
	{
		auto F = {
			Feature2D::AREA_PIXELS_COUNT,
			Feature2D::AREA_UM2,
			Feature2D::CENTROID_X,
			Feature2D::CENTROID_Y,
			Feature2D::DIAMETER_EQUAL_AREA,
			Feature2D::WEIGHTED_CENTROID_Y,
			Feature2D::WEIGHTED_CENTROID_X,
			Feature2D::COMPACTNESS,
			Feature2D::BBOX_YMIN,
			Feature2D::BBOX_XMIN,
			Feature2D::BBOX_HEIGHT,
			Feature2D::BBOX_WIDTH,
			Feature2D::MAJOR_AXIS_LENGTH,
			Feature2D::MINOR_AXIS_LENGTH,
			Feature2D::ECCENTRICITY,
			Feature2D::ORIENTATION,
			Feature2D::ROUNDNESS,
			Feature2D::EXTENT,
			Feature2D::ASPECT_RATIO,
			Feature2D::DIAMETER_EQUAL_PERIMETER,
			Feature2D::CONVEX_HULL_AREA,
			Feature2D::SOLIDITY,
			Feature2D::PERIMETER,
			Feature2D::EDGE_MEAN_INTENSITY,
			Feature2D::EDGE_STDDEV_INTENSITY,
			Feature2D::EDGE_MAX_INTENSITY,
			Feature2D::EDGE_MIN_INTENSITY,
			Feature2D::CIRCULARITY,
			Feature2D::MASS_DISPLACEMENT };
		Nyxus::theFeatureSet.enableFeatures (F, enable);
		return true;
	}
	if ((Fgroup2D)fgcode == Fgroup2D::FG2_BASIC_MORPHOLOGY)
	{
		auto F = {
			Feature2D::AREA_PIXELS_COUNT,
			Feature2D::AREA_UM2,
			Feature2D::CENTROID_X,
			Feature2D::CENTROID_Y,
			Feature2D::BBOX_YMIN,
			Feature2D::BBOX_XMIN,
			Feature2D::BBOX_HEIGHT,
			Feature2D::BBOX_WIDTH };
		Nyxus::theFeatureSet.enableFeatures (F, enable);
		return true;
	}
	if ((Fgroup2D)fgcode == Fgroup2D::FG2_GLCM)
	{
		Nyxus::theFeatureSet.enableFeatures (GLCMFeature::featureset, enable);
		return true;
	}
	if ((Fgroup2D)fgcode == Fgroup2D::FG2_GLRLM)
	{
		Nyxus::theFeatureSet.enableFeatures (GLRLMFeature::featureset, enable);
		return true;
	}
	if ((Fgroup2D)fgcode == Fgroup2D::FG2_GLDZM)
	{
		Nyxus::theFeatureSet.enableFeatures (GLDZMFeature::featureset, enable);
		return true;
	}
	if ((Fgroup2D)fgcode == Fgroup2D::FG2_GLSZM)
	{
		Nyxus::theFeatureSet.enableFeatures (GLSZMFeature::featureset, enable);
		return true;
	}
	if ((Fgroup2D)fgcode == Fgroup2D::FG2_GLDM)
	{
		Nyxus::theFeatureSet.enableFeatures(GLDMFeature::featureset, enable);
		return true;
	}
	if ((Fgroup2D)fgcode == Fgroup2D::FG2_NGLDM)
	{
		Nyxus::theFeatureSet.enableFeatures (NGLDMfeature::featureset, enable);
		return true;
	}
	if ((Fgroup2D)fgcode == Fgroup2D::FG2_NGTDM)
	{
		Nyxus::theFeatureSet.enableFeatures (NGTDMFeature::featureset, enable);
		return true;
	}

	if ((Fgroup2D)fgcode == Fgroup2D::FG2_GEOMOMENTS)
	{
		Nyxus::theFeatureSet.enableFeatures (Smoms2D_feature::featureset, enable);
		Nyxus::theFeatureSet.enableFeatures (Imoms2D_feature::featureset, enable);
		return true;
	}

	if ((Fgroup2D)fgcode == Fgroup2D::FG2_GEOMOMENTS_I)
	{
		Nyxus::theFeatureSet.enableFeatures (Imoms2D_feature::featureset, enable);
		return true;
	}

	if ((Fgroup2D)fgcode == Fgroup2D::FG2_GEOMOMENTS_S)
	{
		Nyxus::theFeatureSet.enableFeatures (Smoms2D_feature::featureset, enable);
		return true;
	}

	// unrecognized feature group
	return false;
}

// Returns:
//		true if s is recognized as a group name
//		false if not recognized (s may be an individual feature name)
//
bool Environment::expand_3D_featuregroup (const std::string& s)
{
	// mutually exclusive groups:
	if (s == Nyxus::theFeatureSet.findGroupNameByCode(Fgroup3D::FG3_ALL))
	{
		theFeatureSet.enableAll(false);

		auto F =
		{
			Feature3D::COV,
			Feature3D::COVERED_IMAGE_INTENSITY_RANGE,
			Feature3D::ENERGY,
			Feature3D::ENTROPY,
			Feature3D::EXCESS_KURTOSIS,
			Feature3D::HYPERFLATNESS,
			Feature3D::HYPERSKEWNESS,
			Feature3D::INTEGRATED_INTENSITY,
			Feature3D::INTERQUARTILE_RANGE,
			Feature3D::KURTOSIS,
			Feature3D::MAX,
			Feature3D::MEAN,
			Feature3D::MEAN_ABSOLUTE_DEVIATION,
			Feature3D::MEDIAN,
			Feature3D::MEDIAN_ABSOLUTE_DEVIATION,
			Feature3D::MIN,
			Feature3D::MODE,
			Feature3D::P01, Feature3D::P10, Feature3D::P25, Feature3D::P75, Feature3D::P90, Feature3D::P99,
			Feature3D::QCOD,
			Feature3D::RANGE,
			Feature3D::ROBUST_MEAN,
			Feature3D::ROBUST_MEAN_ABSOLUTE_DEVIATION,
			Feature3D::ROOT_MEAN_SQUARED,
			Feature3D::SKEWNESS,
			Feature3D::STANDARD_DEVIATION,
			Feature3D::STANDARD_DEVIATION_BIASED,
			Feature3D::STANDARD_ERROR,
			Feature3D::VARIANCE,
			Feature3D::VARIANCE_BIASED,
			Feature3D::UNIFORMITY,
			Feature3D::UNIFORMITY_PIU,

			// 3D features planned for a future PR
#if 0
			Feature3D::AREA,
			Feature3D::MESH_VOLUME,
			Feature3D::VOLUME_CONVEXHULL,
			Feature3D::DIAMETER_EQUAL_AREA,
			Feature3D::DIAMETER_EQUAL_VOLUME,

			Feature3D::VOLUME_PIXELS,
			Feature3D::CENTROID_X,
			Feature3D::CENTROID_Y,
			Feature3D::CENTROID_Z,
			Feature3D::BBOX_XMIN,
			Feature3D::BBOX_YMIN,
			Feature3D::BBOX_ZMIN,
			Feature3D::BBOX_HEIGHT,
			Feature3D::BBOX_WIDTH,
			Feature3D::BBOX_DEPTH,

			// Neighbor features
			Feature3D::NUM_NEIGHBORS,
			Feature3D::PERCENT_TOUCHING,
			Feature3D::CLOSEST_NEIGHBOR1_DIST,
			Feature3D::CLOSEST_NEIGHBOR1_ANG,
			Feature3D::CLOSEST_NEIGHBOR2_DIST,
			Feature3D::CLOSEST_NEIGHBOR2_ANG,
			Feature3D::ANG_BW_NEIGHBORS_MEAN,
			Feature3D::ANG_BW_NEIGHBORS_STDDEV,
			Feature3D::ANG_BW_NEIGHBORS_MODE,

			// Spatial (raw) moments
			Feature3D::SPAT_MOMENT_00,
			Feature3D::SPAT_MOMENT_01,
			Feature3D::SPAT_MOMENT_02,
			Feature3D::SPAT_MOMENT_03,
			Feature3D::SPAT_MOMENT_10,
			Feature3D::SPAT_MOMENT_11,
			Feature3D::SPAT_MOMENT_12,
			Feature3D::SPAT_MOMENT_13,
			Feature3D::SPAT_MOMENT_20,
			Feature3D::SPAT_MOMENT_21,
			Feature3D::SPAT_MOMENT_22,
			Feature3D::SPAT_MOMENT_23,
			Feature3D::SPAT_MOMENT_30
#endif
		};

		theFeatureSet.enableFeatures(F);
		return true;
	}

	if (s == Nyxus::theFeatureSet.findGroupNameByCode(Fgroup3D::FG3_GLDZM))
	{
		theFeatureSet.enableAll(false);

		auto F =
		{
			Feature3D::GLDZM_SDE,
			Feature3D::GLDZM_LDE,
			Feature3D::GLDZM_LGLZE,
			Feature3D::GLDZM_HGLZE,
			Feature3D::GLDZM_SDLGLE,
			Feature3D::GLDZM_SDHGLE,
			Feature3D::GLDZM_LDLGLE,
			Feature3D::GLDZM_LDHGLE,
			Feature3D::GLDZM_GLNU,
			Feature3D::GLDZM_GLNUN,
			Feature3D::GLDZM_ZDNU,
			Feature3D::GLDZM_ZDNUN,
			Feature3D::GLDZM_ZP,
			Feature3D::GLDZM_GLM,
			Feature3D::GLDZM_GLV,
			Feature3D::GLDZM_ZDM,
			Feature3D::GLDZM_ZDV,
			Feature3D::GLDZM_ZDE
		};

		theFeatureSet.enableFeatures(F);
		return true;
	}



	return false;
}

// Returns:
//		true if s is recognized as a group name
//		false if not recognized (s may be an individual feature name)
//
bool Environment::expand_IMQ_featuregroup (const std::string & s)
{
	if (s == Nyxus::theFeatureSet.findGroupNameByCode(FgroupIMQ::ALL_IMQ))
	{
		Nyxus::theFeatureSet.enableAllIMQ();
		return true; 
	}

	return false;
}

void Environment::expand_featuregroups()
{
	// initially, no features are enabled
	theFeatureSet.enableAll(false); 

	// enable/disable feature groups and individual features according to user's choice
	for (auto& s : recognizedFeatureNames)
	{
		if (is_imq()) {
			if (expand_IMQ_featuregroup (s)) 
				return;

			FeatureIMQ a;
			if (!theFeatureSet.find_IMQ_FeatureByString(s, a))
				throw std::invalid_argument("Error: '" + s + "' is not a valid Image Quality feature name \n");

			theFeatureSet.enableFeature (int(a));
			continue;
		}

		if (dim() == 2)
		{
			if (expand_2D_featuregroup (s))
				continue;
		}

		if (dim() == 3)
		{
			if (expand_3D_featuregroup (s))
				continue;
		}

		// 's' is an individual feature name, not feature group name. Process it now
		if (dim() == 2)
		{
			int fcode; // signed Feature2D
			if (!theFeatureSet.find_2D_FeatureByString (s, fcode))
				throw std::invalid_argument("Error: '" + s + "' is not a valid 2D feature name \n");

			theFeatureSet.enableFeature (fcode);

			continue;
		}

		if (dim() == 3)
		{
			Feature3D a;
			if (!theFeatureSet.find_3D_FeatureByString(s, a))
				throw std::invalid_argument("Error: '" + s + "' is not a valid 3D feature name \n");

			theFeatureSet.enableFeature (int(a));
			continue;
		}
	}
}

void Environment::show_featureset_help()
{
	const int W = 40;   // width

	if (dim() == 2)
	{
		std::cout << "\nAvailable 2D features: \n";

		for (auto f = Nyxus::UserFacingFeatureNames.begin(); f != Nyxus::UserFacingFeatureNames.end(); ++f) 
		{
			auto idx = std::distance(Nyxus::UserFacingFeatureNames.begin(), f);

			std::cout << std::setw(W) << f->first << " ";
			if ((idx + 1) % 4 == 0)
				std::cout << "\n";
		}
		std::cout << "\n";

		std::cout << "\nAvailable 2D feature groups:" << "\n";

		for (const auto& f : Nyxus::UserFacing2dFeaturegroupNames)
			std::cout << std::setw(W) << f.first << "\n";
		std::cout << "\n";
	}
	else
		if (dim() == 3)
		{
			std::cout << "\nAvailable 3D features: \n";

			for (auto f = Nyxus::UserFacing_3D_featureNames.begin(); f != Nyxus::UserFacing_3D_featureNames.end(); ++f) 
			{
				auto idx = std::distance(Nyxus::UserFacing_3D_featureNames.begin(), f);

				std::cout << std::setw(W) << f->first << " ";
				if ((idx + 1) % 4 == 0)
					std::cout << "\n";
			}
			std::cout << "\n";

			std::cout << "\nAvailable 3D feature groups:" << "\n";

			for (const auto& f : Nyxus::UserFacing3dFeaturegroupNames)
				std::cout << std::setw(W) << f.first << "\n";
			std::cout << "\n";
		}
		else
			std::cout << "No features for dimensionality " << dim() << '\n';
}


