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
#include "features/hexagonality_polygonality.h"
#include "features/2d_geomoments.h"
#include "features/intensity.h"
#include "features/neighbors.h"
#include "features/ngldm.h"
#include "features/ngtdm.h"
#include "features/radial_distribution.h"
#include "features/roi_radius.h"
#include "features/zernike.h"

#include "features/3d_surface.h"
#include "features/3d_glcm.h"
#include "features/3d_gldm.h"
#include "features/3d_ngldm.h"
#include "features/3d_ngtdm.h"
#include "features/3d_gldzm.h"
#include "features/3d_glrlm.h"
#include "features/3d_glszm.h"
#include "features/3d_intensity.h"
#include "features/3d_surface.h"

#include "features/focus_score.h"
#include "features/power_spectrum.h"
#include "features/saturation.h"
#include "features/sharpness.h"

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
			auto gname = theFeatureSet.findGroupNameByCode(Fgroup2D::FG2_ALL);
			std::cout << "Warning: no features specified, defaulting to " << gname << "\n";
			fnames.push_back(gname);
			return true;
		}

		if (dim() == 3)
		{
			auto gname = theFeatureSet.findGroupNameByCode(Fgroup3D::FG3_ALL);
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
					set_imq (true);

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
					set_imq (true);

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
		show_featureset_help();

	return success;
}

// Returns:
//		true if s is recognized as a group name
//		false if not recognized (s may be an individual feature name)
//
bool Environment::expand_2D_featuregroup (const std::string & s)
{
	int fgcode;
	if (! theFeatureSet.find_2D_GroupByString(s, fgcode))
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
			theFeatureSet.enableFeature (i);
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
		theFeatureSet.enableFeatures (NeighborsFeature::featureset, enable);
		return true;
	}

	if ((Fgroup2D) fgcode == Fgroup2D::FG2_INTENSITY)
	{
		theFeatureSet.enableFeatures (PixelIntensityFeatures::featureset, enable);
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
		theFeatureSet.enableFeatures (GLCMFeature::featureset, enable);
		return true;
	}
	if ((Fgroup2D)fgcode == Fgroup2D::FG2_GLRLM)
	{
		theFeatureSet.enableFeatures (GLRLMFeature::featureset, enable);
		return true;
	}
	if ((Fgroup2D)fgcode == Fgroup2D::FG2_GLDZM)
	{
		theFeatureSet.enableFeatures (GLDZMFeature::featureset, enable);
		return true;
	}
	if ((Fgroup2D)fgcode == Fgroup2D::FG2_GLSZM)
	{
		theFeatureSet.enableFeatures (GLSZMFeature::featureset, enable);
		return true;
	}
	if ((Fgroup2D)fgcode == Fgroup2D::FG2_GLDM)
	{
		theFeatureSet.enableFeatures(GLDMFeature::featureset, enable);
		return true;
	}
	if ((Fgroup2D)fgcode == Fgroup2D::FG2_NGLDM)
	{
		theFeatureSet.enableFeatures (NGLDMfeature::featureset, enable);
		return true;
	}
	if ((Fgroup2D)fgcode == Fgroup2D::FG2_NGTDM)
	{
		theFeatureSet.enableFeatures (NGTDMFeature::featureset, enable);
		return true;
	}

	if ((Fgroup2D)fgcode == Fgroup2D::FG2_GEOMOMENTS)
	{
		theFeatureSet.enableFeatures (Smoms2D_feature::featureset, enable);
		theFeatureSet.enableFeatures (Imoms2D_feature::featureset, enable);
		return true;
	}

	if ((Fgroup2D)fgcode == Fgroup2D::FG2_GEOMOMENTS_I)
	{
		theFeatureSet.enableFeatures (Imoms2D_feature::featureset, enable);
		return true;
	}

	if ((Fgroup2D)fgcode == Fgroup2D::FG2_GEOMOMENTS_S)
	{
		theFeatureSet.enableFeatures (Smoms2D_feature::featureset, enable);
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
	if (!theFeatureSet.find_3D_GroupByString(s, fgcode))
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
		theFeatureSet.enableFeatures (D3_VoxelIntensityFeatures::featureset, enable);
		theFeatureSet.enableFeatures (D3_SurfaceFeature::featureset, enable);
		theFeatureSet.enableFeatures (D3_GLCM_feature::featureset, enable);
		theFeatureSet.enableFeatures (D3_GLDM_feature::featureset, enable);
		theFeatureSet.enableFeatures (D3_GLDZM_feature::featureset, enable);
		theFeatureSet.enableFeatures (D3_GLRLM_feature::featureset, enable);
		theFeatureSet.enableFeatures (D3_GLSZM_feature::featureset, enable);
		theFeatureSet.enableFeatures (D3_NGLDM_feature::featureset, enable);
		theFeatureSet.enableFeatures (D3_NGTDM_feature::featureset, enable);

		return true;
	}

	if ((Fgroup3D)fgcode == Fgroup3D::FG3_INTENSITY)
	{
		theFeatureSet.enableFeatures (D3_VoxelIntensityFeatures::featureset, enable);
		return true;
	}

	if ((Fgroup3D)fgcode == Fgroup3D::FG3_MORPHOLOGY)
	{
		theFeatureSet.enableFeatures (D3_SurfaceFeature::featureset, enable);
		return true;
	}

	if ((Fgroup3D)fgcode == Fgroup3D::FG3_TEXTURE)
	{
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
	if (s == theFeatureSet.findGroupNameByCode(FgroupIMQ::ALL_IMQ))
	{
		theFeatureSet.enableAllIMQ();
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
			if (!theFeatureSet.find_2D_FeatureByString (s, fcode))
				throw std::invalid_argument("Error: '" + s + "' is not a valid 2D feature name \n");

			theFeatureSet.enableFeature (fcode);

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

void Environment::compile_feature_settings()
{
	f_settings_.push_back (fsett_PixelIntensity);
		feature2settings_ [typeid(PixelIntensityFeatures).hash_code()] = f_settings_.size() - 1;

	f_settings_.push_back (fsett_BasicMorphology);
		feature2settings_ [typeid(BasicMorphologyFeatures).hash_code()] = f_settings_.size() - 1;

	f_settings_.push_back (fsett_Neighbors);
		feature2settings_ [typeid(NeighborsFeature).hash_code()] = f_settings_.size() - 1;

	f_settings_.push_back (fsett_Contour);
		feature2settings_ [typeid(ContourFeature).hash_code()] = f_settings_.size() - 1;

	f_settings_.push_back (fsett_ConvexHull);
		feature2settings_[typeid(ConvexHullFeature).hash_code()] = f_settings_.size() - 1;

	f_settings_.push_back(fsett_EllipseFitting);
		feature2settings_ [typeid(EllipseFittingFeature).hash_code()] = f_settings_.size() - 1;

	f_settings_.push_back(fsett_Extrema);
		feature2settings_ [typeid(ExtremaFeature).hash_code()] = f_settings_.size() - 1;

	f_settings_.push_back(fsett_EulerNumber);
		feature2settings_ [typeid(EulerNumberFeature).hash_code()] = f_settings_.size() - 1;

	f_settings_.push_back(fsett_CaliperFeret);
		feature2settings_ [typeid(CaliperFeretFeature).hash_code()] = f_settings_.size() - 1;

	f_settings_.push_back(fsett_CaliperMartin);
		feature2settings_ [typeid(CaliperMartinFeature).hash_code()] = f_settings_.size() - 1;

	f_settings_.push_back(fsett_CaliperNassenstein);
		feature2settings_ [typeid(CaliperNassensteinFeature).hash_code()] = f_settings_.size() - 1;

	f_settings_.push_back(fsett_Chords);
		feature2settings_ [typeid(ChordsFeature).hash_code()] = f_settings_.size() - 1;

	f_settings_.push_back(fsett_HexagonalityPolygonality);
		feature2settings_ [typeid(HexagonalityPolygonalityFeature).hash_code()] = f_settings_.size() - 1;

	f_settings_.push_back(fsett_EnclosingInscribingCircumscribingCircle);
		feature2settings_ [typeid(EnclosingInscribingCircumscribingCircleFeature).hash_code()] = f_settings_.size() - 1;

	f_settings_.push_back(fsett_GeodeticLengthThickness);
		feature2settings_ [typeid(GeodeticLengthThicknessFeature).hash_code()] = f_settings_.size() - 1;

	f_settings_.push_back(fsett_RoiRadius);
		feature2settings_ [typeid(RoiRadiusFeature).hash_code()] = f_settings_.size() - 1;

	f_settings_.push_back(fsett_ErosionPixels);
		feature2settings_ [typeid(ErosionPixelsFeature).hash_code()] = f_settings_.size() - 1;

	f_settings_.push_back(fsett_FractalDimension);
		feature2settings_ [typeid(FractalDimensionFeature).hash_code()] = f_settings_.size() - 1;

	f_settings_.push_back(fsett_GLCM);
		feature2settings_ [typeid(GLCMFeature).hash_code()] = f_settings_.size() - 1;

	f_settings_.push_back(fsett_GLRLM);
		feature2settings_ [typeid(GLRLMFeature).hash_code()] = f_settings_.size() - 1;

	f_settings_.push_back(fsett_GLDZM);
		feature2settings_ [typeid(GLDZMFeature).hash_code()] = f_settings_.size() - 1;

	f_settings_.push_back(fsett_GLSZM);
		feature2settings_ [typeid(GLSZMFeature).hash_code()] = f_settings_.size() - 1;

	f_settings_.push_back(fsett_GLDM);
		feature2settings_ [typeid(GLDMFeature).hash_code()] = f_settings_.size() - 1;

	f_settings_.push_back(fsett_NGLDM);
		feature2settings_ [typeid(NGLDMfeature).hash_code()] = f_settings_.size() - 1;

	f_settings_.push_back(fsett_NGTDM);
		feature2settings_ [typeid(NGTDMFeature).hash_code()] = f_settings_.size() - 1;

	f_settings_.push_back(fsett_Imoms2D);
		feature2settings_ [typeid(Imoms2D_feature).hash_code()] = f_settings_.size() - 1;

	f_settings_.push_back(fsett_Smoms2D);
		feature2settings_ [typeid(Smoms2D_feature).hash_code()] = f_settings_.size() - 1;

	f_settings_.push_back(fsett_Gabor);
		feature2settings_ [typeid(GaborFeature).hash_code()] = f_settings_.size() - 1;

	f_settings_.push_back(fsett_Zernike);
		feature2settings_ [typeid(ZernikeFeature).hash_code()] = f_settings_.size() - 1;

	f_settings_.push_back(fsett_RadialDistribution);
		feature2settings_ [typeid(RadialDistributionFeature).hash_code()] = f_settings_.size() - 1;

		// 3D
	f_settings_.push_back(fsett_D3_VoxelIntensity);
		feature2settings_ [typeid(D3_VoxelIntensityFeatures).hash_code()] = f_settings_.size() - 1;

	f_settings_.push_back(fsett_D3_Surface);
		feature2settings_ [typeid(D3_SurfaceFeature).hash_code()] = f_settings_.size() - 1;

	f_settings_.push_back(fsett_D3_GLCM);
		feature2settings_ [typeid(D3_GLCM_feature).hash_code()] = f_settings_.size() - 1;

	f_settings_.push_back(fsett_D3_GLDM);
		feature2settings_ [typeid(D3_GLDM_feature).hash_code()] = f_settings_.size() - 1;

	f_settings_.push_back(fsett_D3_GLDZM);
		feature2settings_ [typeid(D3_GLDZM_feature).hash_code()] = f_settings_.size() - 1;

	f_settings_.push_back(fsett_D3_NGLDM);
		feature2settings_ [typeid(D3_NGLDM_feature).hash_code()] = f_settings_.size() - 1;

	f_settings_.push_back(fsett_D3_NGTDM);
		feature2settings_ [typeid(D3_NGTDM_feature).hash_code()] = f_settings_.size() - 1;

	f_settings_.push_back(fsett_D3_GLSZM);
		feature2settings_ [typeid(D3_GLSZM_feature).hash_code()] = f_settings_.size() - 1;

	f_settings_.push_back(fsett_D3_GLRLM);
		feature2settings_ [typeid(D3_GLSZM_feature).hash_code()] = f_settings_.size() - 1;

		// 2D image quality
	f_settings_.push_back(fsett_FocusScore);
		feature2settings_ [typeid(FocusScoreFeature).hash_code()] = f_settings_.size() - 1;

	f_settings_.push_back(fsett_PowerSpectrum);
		feature2settings_ [typeid(PowerSpectrumFeature).hash_code()] = f_settings_.size() - 1;

	f_settings_.push_back(fsett_Saturation);
		feature2settings_ [typeid(SaturationFeature).hash_code()] = f_settings_.size() - 1;

	f_settings_.push_back(fsett_Sharpness);
		feature2settings_ [typeid(SharpnessFeature).hash_code()] = f_settings_.size() - 1;

		for (auto& wrapd_s : f_settings_)
		{
			auto& s = wrapd_s.get();
			s.clear();
			s.resize((int)NyxSetting::__COUNT__);
			s[(int)NyxSetting::SOFTNAN].rval = resultOptions.noval();
			s[(int)NyxSetting::TINY].rval = resultOptions.tiny();
			s[(int)NyxSetting::SINGLEROI].bval = singleROI;
			s[(int)NyxSetting::GREYDEPTH].ival = get_coarse_gray_depth();
			s[(int)NyxSetting::PIXELSIZEUM].rval = pixelSizeUm;
			s[(int)NyxSetting::PIXELDISTANCE].ival = get_pixel_distance();
			s[(int)NyxSetting::USEGPU].bval = using_gpu();
			s[(int)NyxSetting::VERBOSLVL].ival = get_verbosity_level();
			s[(int)NyxSetting::IBSI].bval = ibsi_compliance;
		}
}

const Fsettings& Environment::get_feature_settings (const std::type_info& ftype)
{
	size_t h = ftype.hash_code();
	int idx = feature2settings_[h];
	const Fsettings& s = f_settings_[idx];
	return s;
}


