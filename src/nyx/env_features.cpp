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
#include "features/gabor.h"
#include "features/glcm.h"
#include "features/gldm.h"
#include "features/gldzm.h"
#include "features/glrlm.h"
#include "features/glszm.h"
#include "features/image_moments.h"
#include "features/intensity.h"
#include "features/neighbors.h"
#include "features/ngldm.h"
#include "features/ngtdm.h"
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
			Fgroup2D afg;
			bool gnameExists = theFeatureSet.find_2D_GroupByString (s_uppr, afg);

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

			Feature2D af;
			bool fnameExists = theFeatureSet.find_2D_FeatureByString (s_uppr, af);

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
				std::cerr << "Error: expecting '" << s << "' to be a proper 2D feature name or feature file path\n";
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
	if (s == Nyxus::theFeatureSet.findGroupNameByCode(Fgroup2D::FG2_ALL))
	{
		Nyxus::theFeatureSet.enableAll();

		// disabled image quality features (will improve disabling if we add this as feature group)
		auto F = {
			Feature2D::FOCUS_SCORE,
			Feature2D::LOCAL_FOCUS_SCORE,
			Feature2D::POWER_SPECTRUM_SLOPE,
			Feature2D::MIN_SATURATION,
			Feature2D::MAX_SATURATION,
			Feature2D::SHARPNESS,
			Feature2D::BRISQUE
		};
		
		Nyxus::theFeatureSet.disableFeatures(F);
		return true; 
	}
	if (s == Nyxus::theFeatureSet.findGroupNameByCode(Fgroup2D::FG2_BUT_GABOR))
	{
		Nyxus::theFeatureSet.enableAll();
		auto F = { Feature2D::GABOR };
		Nyxus::theFeatureSet.disableFeatures(F);
		return true; 
	}
	if (s == Nyxus::theFeatureSet.findGroupNameByCode(Fgroup2D::FG2_ALL_BUT_GLCM))
	{
		Nyxus::theFeatureSet.enableAll();
		theFeatureSet.disableFeatures(GLCMFeature::featureset);
		return true;
	}

	if (s == Nyxus::theFeatureSet.findGroupNameByCode(Fgroup2D::FG2_EASY))
	{
		theFeatureSet.enableAll();
		theFeatureSet.disableFeatures(GaborFeature::featureset);
		theFeatureSet.disableFeatures(GLCMFeature::featureset);
		theFeatureSet.disableFeatures(ImageMomentsFeature::featureset);
		return true;
	}

	if (s == Nyxus::theFeatureSet.findGroupNameByCode(Fgroup2D::FG2_NEIG))
	{
		theFeatureSet.enableAll();
		theFeatureSet.enableFeatures(NeighborsFeature::featureset);
		return true;
	}

	if (s == Nyxus::theFeatureSet.findGroupNameByCode(Fgroup2D::FG2_INTENSITY))
	{
		theFeatureSet.enableFeatures(PixelIntensityFeatures::featureset);
		return true;
	}
	if (s == Nyxus::theFeatureSet.findGroupNameByCode(Fgroup2D::FG2_MORPHOLOGY))
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
		theFeatureSet.enableFeatures(F);
		return true;
	}
	if (s == Nyxus::theFeatureSet.findGroupNameByCode(Fgroup2D::FG2_BASIC_MORPHOLOGY))
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
		theFeatureSet.enableFeatures(F);
		return true;
	}
	if (s == Nyxus::theFeatureSet.findGroupNameByCode(Fgroup2D::FG2_GLCM))
	{
		theFeatureSet.enableFeatures(GLCMFeature::featureset);
		return true;
	}
	if (s == Nyxus::theFeatureSet.findGroupNameByCode(Fgroup2D::FG2_GLRLM))
	{
		theFeatureSet.enableFeatures(GLRLMFeature::featureset);
		return true;
	}
	if (s == Nyxus::theFeatureSet.findGroupNameByCode(Fgroup2D::FG2_GLDZM))
	{
		theFeatureSet.enableFeatures(GLDZMFeature::featureset);
		return true;
	}
	if (s == Nyxus::theFeatureSet.findGroupNameByCode(Fgroup2D::FG2_GLSZM))
	{
		theFeatureSet.enableFeatures(GLSZMFeature::featureset);
		return true;
	}
	if (s == Nyxus::theFeatureSet.findGroupNameByCode(Fgroup2D::FG2_GLDM))
	{
		theFeatureSet.enableFeatures(GLDMFeature::featureset);
		return true;
	}
	if (s == Nyxus::theFeatureSet.findGroupNameByCode(Fgroup2D::FG2_NGLDM))
	{
		theFeatureSet.enableFeatures(NGLDMfeature::featureset);
		return true;
	}
	if (s == Nyxus::theFeatureSet.findGroupNameByCode(Fgroup2D::FG2_NGTDM))
	{
		theFeatureSet.enableFeatures(NGTDMFeature::featureset);
		return true;
	}

	if (s == Nyxus::theFeatureSet.findGroupNameByCode(Fgroup2D::FG2_MOMENTS))
	{
		theFeatureSet.enableFeatures(ImageMomentsFeature::featureset);
		return true;
	}

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

// 3D features planned for a following PR
#if 0
			// Morphology:
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

	return false;
}

void Environment::expand_featuregroups()
{
	theFeatureSet.enableAll(false); 
	for (auto& s : recognizedFeatureNames) // Second, iterate uppercased feature names
	{
		// Enforce the feature names to be in uppercase
		s = Nyxus::toupper(s);

		if (dim() == 2)
		{
			if (expand_2D_featuregroup (s))
				return;
		}

		if (dim() == 3)
		{
			if (expand_3D_featuregroup (s))
				return;
		}

		// 's' is an individual feature name, not feature group name. Process it now
		if (dim() == 2)
		{
			Feature2D a;
			if (!theFeatureSet.find_2D_FeatureByString(s, a))
				throw std::invalid_argument("Error: '" + s + "' is not a valid 2D feature name \n");

			theFeatureSet.enableFeature (int(a));
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


