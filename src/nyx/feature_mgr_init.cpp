#include "feature_mgr.h"
#include "features/basic_morphology.h"
#include "features/caliper.h"
#include "features/chords.h"
#include "features/convex_hull.h"
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
#include "features/gldzm.h"
#include "features/glszm.h"
#include "features/gldm.h"
#include "features/hexagonality_polygonality.h"
#include "features/ngldm.h"
#include "features/ngtdm.h"
#include "features/2d_geomoments.h"
#include "features/intensity.h"
#include "features/moments.h"
#include "features/neighbors.h"
#include "features/caliper.h"
#include "features/roi_radius.h"
#include "features/zernike.h"

#include "features/3d_intensity.h"
#include "features/3d_gldzm.h"

#include "features/focus_score.h"
#include "features/power_spectrum.h"
#include "features/saturation.h"
#include "features/sharpness.h"

FeatureManager::FeatureManager()
{
	// 2D
	register_feature (new PixelIntensityFeatures());
	register_feature (new BasicMorphologyFeatures());
	register_feature (new NeighborsFeature());
	register_feature (new ContourFeature());
	register_feature (new ConvexHullFeature());
	register_feature (new EllipseFittingFeature());
	register_feature (new ExtremaFeature());
	register_feature (new EulerNumberFeature());
	register_feature (new CaliperFeretFeature());
	register_feature (new CaliperMartinFeature());
	register_feature (new CaliperNassensteinFeature());
	register_feature (new ChordsFeature());
	register_feature (new HexagonalityPolygonalityFeature());
	register_feature (new EnclosingInscribingCircumscribingCircleFeature());	
	register_feature (new GeodeticLengthThicknessFeature());
	register_feature (new RoiRadiusFeature());
	register_feature (new ErosionPixelsFeature());	
	register_feature (new FractalDimensionFeature());
	register_feature (new GLCMFeature());	
	register_feature (new GLRLMFeature());	
	register_feature (new GLDZMFeature());
	register_feature (new GLSZMFeature());
	register_feature (new GLDMFeature());
	register_feature (new NGLDMfeature());
	register_feature (new NGTDMFeature());
	register_feature (new Imoms2D_feature());
	register_feature (new Smoms2D_feature());
	register_feature (new GaborFeature());
	register_feature (new ZernikeFeature());
	register_feature (new RadialDistributionFeature());
	// 3D
	register_feature (new D3_PixelIntensityFeatures());
	register_feature (new D3_GLDZM_feature());

	// image quality
	register_feature (new FocusScoreFeature());
	register_feature (new PowerSpectrumFeature());
	register_feature (new SaturationFeature());
	register_feature (new SharpnessFeature());
}bool FeatureManager::init_feature_classes()
{
	return GaborFeature::init_class();
}

