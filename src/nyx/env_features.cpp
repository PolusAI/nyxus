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
#include "features/3d_glcm.h"
#include "features/3d_gldm.h"
#include "features/3d_ngldm.h"
#include "features/3d_ngtdm.h"
#include "features/3d_gldzm.h"
#include "features/3d_glrlm.h"
#include "features/3d_glszm.h"
#include "features/radial_distribution.h"
#include "features/roi_radius.h"
#include "features/zernike.h"
#include "helpers/helpers.h"
#include "helpers/system_resource.h"
#include "helpers/timing.h"
#include "version.h"

using namespace Nyxus;

// The purpose of this methos is checking user's feature request but not changing the state of the Environment instance.
// Specifically, it:
// (1) splits 'comma_separated_fnames' into identifiers, 
// (2) checks if they are known feature and group names in the corresponding context (2D or 3D), and 
// (3) saves them in vector 'fnames'
bool Environment::spellcheck_raw_featurelist (const std::string & comma_separated_fnames, std::vector<std::string> & fnames)
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

	// Chop the comma-separated feature list
	bool success = true;
	std::vector<std::string> strings;
	parse_delimited_string (comma_separated_fnames, ",", strings);

	// Check names of features and feature-groups
	for (const std::string & s : strings)
	{
		// Forgive user's typos of consecutive commas e.g. MIN,MAX,,MEDIAN
		if (s.empty())
			continue;

		auto s_uppr = Nyxus::toupper(s);

		if (dim() == 2)
		{
			//==== feature group ?
			
			int _; // feature code, unused
			bool gfound1 = theFeatureSet.find_2D_GroupByString (s_uppr, _),
				gfound2 = theFeatureSet.find_IMQ_GroupByString (s_uppr, _);

			// if 's' is recognized as a group, register it and skip checking it as an individual feature name
			if (gfound1 || gfound2)
			{
				// set the IMQ flag if applicable
				if (gfound2)
					theEnvironment.set_imq (true);

				fnames.push_back (s_uppr);
				continue;
			}

			//==== individual feature ?

			bool ffound1 = theFeatureSet.find_2D_FeatureByString (s_uppr, _),
				ffound2 = theFeatureSet.find_IMQ_FeatureByString (s_uppr, _);

			// if a feature is found, register it
			if (! (ffound1 || ffound2))
			{
				success = false;
				std::cerr << "Error: expecting " + s + " to be a proper 2D feature name or feature file path\n";
			}
			else
			{
				// set the IMQ flag if applicable
				if (ffound2)
					theEnvironment.set_imq (true);

				fnames.push_back(s_uppr);
			}

		} // 2D

		if (dim() == 3)
		{
			// Is feature found among 3D features?
			int afg; // signed Fgroup3D
			bool gnameExists = theFeatureSet.find_3D_GroupByString (s_uppr, afg);

			// If a group is found, register it
			if (gnameExists)
			{
				fnames.push_back (s_uppr);
				continue;
			}

			int af; // signed Feature3D
			bool fnameExists = theFeatureSet.find_3D_FeatureByString (s_uppr, af);

			if (! fnameExists)
			{
				success = false;
				std::cerr << "Error: expecting '" << s << "' to be a proper 3D feature name or feature file path\n";
			}
			else
				fnames.push_back (s_uppr);
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
		// enable just the 2D part of the feature set
		for (int i = (int) Nyxus::Feature2D::_FIRST_; i < (int) Nyxus::Feature2D::_COUNT_; i++)
			Nyxus::theFeatureSet.enableFeature (i);
		return true; 
	}

	if ((Fgroup2D)fgcode == Fgroup2D::FG2_WHOLESLIDE)
	{
		theFeatureSet.enableFeatures (ContourFeature::featureset, enable);
		theFeatureSet.enableFeatures (PixelIntensityFeatures::featureset, enable);
		theFeatureSet.enableFeatures (GLCMFeature::featureset, enable);
		theFeatureSet.enableFeatures (GLDMFeature::featureset, enable);
		theFeatureSet.enableFeatures (GLRLMFeature::featureset, enable);
		theFeatureSet.enableFeatures (GLSZMFeature::featureset, enable);
		theFeatureSet.enableFeatures (NGLDMfeature::featureset, enable);
		theFeatureSet.enableFeatures (NGTDMFeature::featureset, enable);
		theFeatureSet.enableFeatures (GaborFeature::featureset, enable);
		theFeatureSet.enableFeatures(Imoms2D_feature::featureset, enable);
		theFeatureSet.enableFeatures (RadialDistributionFeature::featureset, enable);
		theFeatureSet.enableFeatures (ZernikeFeature::featureset, enable);
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
		theFeatureSet.enableFeatures (BasicMorphologyFeatures::featureset, enable);
		theFeatureSet.enableFeatures (EnclosingInscribingCircumscribingCircleFeature::featureset, enable);
		theFeatureSet.enableFeatures (ContourFeature::featureset, enable);
		theFeatureSet.enableFeatures (ConvexHullFeature::featureset, enable);	
		theFeatureSet.enableFeatures (FractalDimensionFeature::featureset, enable);
		theFeatureSet.enableFeatures (GeodeticLengthThicknessFeature::featureset, enable);	
		theFeatureSet.enableFeatures (NeighborsFeature::featureset, enable);
		theFeatureSet.enableFeatures (RoiRadiusFeature::featureset, enable);
		theFeatureSet.enableFeatures (EllipseFittingFeature::featureset, enable);
		theFeatureSet.enableFeatures (EulerNumberFeature::featureset, enable);
		theFeatureSet.enableFeatures (ExtremaFeature::featureset, enable);
		theFeatureSet.enableFeatures (ErosionPixelsFeature::featureset, enable);
		theFeatureSet.enableFeatures (CaliperFeretFeature::featureset, enable);
		theFeatureSet.enableFeatures (CaliperMartinFeature::featureset, enable);
		theFeatureSet.enableFeatures (CaliperNassensteinFeature::featureset, enable);
		theFeatureSet.enableFeatures (ChordsFeature::featureset, enable);

		return true;
	}
	if ((Fgroup2D)fgcode == Fgroup2D::FG2_BASIC_MORPHOLOGY)
	{
		theFeatureSet.enableFeatures (BasicMorphologyFeatures::featureset, enable);
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
	int fgcode;
	if (!Nyxus::theFeatureSet.find_3D_GroupByString(s, fgcode))
		return false; // 's' is a feature name
	bool enable = true;
	if (fgcode < 0)
	{
		fgcode = -fgcode;
		enable = false;
	}

	// mutually exclusive groups:
	if ((Fgroup3D)fgcode == Fgroup3D::FG3_ALL)
	{
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

		theFeatureSet.enableFeatures (F, enable);
		theFeatureSet.enableFeatures (D3_GLCM_feature::featureset, enable);
		theFeatureSet.enableFeatures (D3_GLDM_feature::featureset, enable);
		theFeatureSet.enableFeatures (D3_GLDZM_feature::featureset, enable);
		theFeatureSet.enableFeatures (D3_GLRLM_feature::featureset, enable);
		theFeatureSet.enableFeatures (D3_GLSZM_feature::featureset, enable);
		theFeatureSet.enableFeatures (D3_NGLDM_feature::featureset, enable);
		theFeatureSet.enableFeatures (D3_NGTDM_feature::featureset, enable);

		return true;
	}

	if ((Fgroup3D)fgcode == Fgroup3D::FG3_GLCM)
	{
		theFeatureSet.enableFeatures (D3_GLCM_feature::featureset, enable);
		return true;
	}

	if ((Fgroup3D)fgcode == Fgroup3D::FG3_GLDM)
	{
		theFeatureSet.enableFeatures (D3_GLDM_feature::featureset, enable);
		return true;
	}

	if ((Fgroup3D)fgcode == Fgroup3D::FG3_NGLDM)
	{
		theFeatureSet.enableFeatures (D3_NGLDM_feature::featureset, enable);
		return true;
	}

	if ((Fgroup3D)fgcode == Fgroup3D::FG3_NGTDM)
	{
		theFeatureSet.enableFeatures (D3_NGTDM_feature::featureset, enable);
		return true;
	}

	if ((Fgroup3D)fgcode == Fgroup3D::FG3_GLDZM)
	{
		theFeatureSet.enableFeatures (D3_GLDZM_feature::featureset, enable);
		return true;
	}

	if ((Fgroup3D)fgcode == Fgroup3D::FG3_GLSZM)
	{
		theFeatureSet.enableFeatures (D3_GLSZM_feature::featureset, enable);
		return true;
	}	
	
	if ((Fgroup3D)fgcode == Fgroup3D::FG3_GLRLM)
	{
		theFeatureSet.enableFeatures (D3_GLRLM_feature::featureset, enable);
		return true;
	}

	// unrecognized feature group
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

			int a;
			if (!theFeatureSet.find_IMQ_FeatureByString(s, a))
				throw std::invalid_argument("Error: '" + s + "' is not a valid Image Quality feature name \n");

			theFeatureSet.enableFeature (a);
			continue;
		}

		// try to interpret 's' as a group name

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
			if (!Nyxus::theFeatureSet.find_2D_FeatureByString (s, fcode))
				throw std::invalid_argument("Error: '" + s + "' is not a valid 2D feature name \n");

			Nyxus::theFeatureSet.enableFeature (fcode);

			continue;
		}

		if (dim() == 3)
		{
			int a; // signed Feature3D
			if (!theFeatureSet.find_3D_FeatureByString(s, a))
				throw std::invalid_argument("Error: '" + s + "' is not a valid 3D feature name \n");
			theFeatureSet.enableFeature (a);
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


