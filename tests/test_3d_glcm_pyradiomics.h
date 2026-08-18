#pragma once

#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>
#include "../src/nyx/environment.h"
#include "../src/nyx/featureset.h"
#include "../src/nyx/roi_cache.h"
#include "../src/nyx/features/3d_glcm.h"
#include "../src/nyx/helpers/fsystem.h"
#include "test_ref_vals.h"

// PyRadiomics goldens for the 3D GLCM family (SPEC 6.4 provenance).
// tool=pyradiomics 3.0.1 (SimpleITK 2.3.1, Python 3.8); env=nyxus_oracle (conda, needs Python <=3.9);
// recipe=glcm3d.pyradiomics_bincount20; generator=tests/vetting/oracles/gen_glcm3d_pyradiomics.py.
//
// Fixture: the COMPAT phantom -- data/nifti/compat_int/compat_int_mri.nii +
// compat_seg/compat_seg_liver.nii, label 1 -- which is what get_3d_compat_phantom() returns and what
// every assertion below uses. (The ut_ phantom named in the older version of this comment belongs to
// test_3d_glcm_regression.h, at a different bin count; the two benchmarks are not comparable.)
//
// Nyxus side: 100 grey levels, GLCM offset 1. BOTH SIDES ARE SYMMETRIC here: Nyxus symmetrises the
// cooccurrence matrix whenever radiomics grey binning is active -- 3d_glcm.cpp,
// `if (symmetric_glcm || radiomics_grey_binning(greyInfo) || ibsi_grey_binning(...)) GLCM.xy(b,a)++`
// -- and GLCM_GREYDEPTH=-20 is that path. An earlier version of this comment claimed Nyxus was
// asymmetric here and justified a 10% band with it; there is no such convention gap, and the
// measured residuals below are what the tolerances are now set from.
//
// TWO REFERENCES, because one scalar cannot vet 13 directions:
//
//   glcm_3d_pyradiomics_ref_vals            PyRadiomics' direction-set value -> the *_AVE features
//   glcm_3d_pyradiomics_ref_vals_by_angle   one value per direction         -> the base features
//
// The base (unsuffixed) features hold one value per 3D angle. Comparing their average against the
// direction-set scalar validates the average and nothing more: per-direction errors that cancel
// leave the mean untouched. The per-direction goldens come from PyRadiomics' own feature formulas
// (RadiomicsGLCM computes each feature per angle and averages last, so intercepting that average
// yields the per-angle vector), which is what makes the base assertions genuine.
//
// ANGLE ORDER: both tools walk the same 13 offsets in the same order, PyRadiomics as (dz,dy,dx) and
// Nyxus as (dx,dy,dz) (`shifts`, 3d_glcm.cpp), so a Nyxus triple reversed is a PyRadiomics row, up
// to a sign that is the same unordered pixel pair. The per-direction table below is stored in NYXUS
// slot order; gen_glcm3d_pyradiomics.py derives that mapping from the two offset lists.
//
// Getting Pyradiomics ground truth values (the generator does exactly this):
//      pyradiomics <intensity>.nii <mask>.nii --param settings1.yaml
// 
// where file "settings1.yaml" is:
// 
//  setting:
//  #disabled - binWidth: 25
//  binCount : 20
//  label : 1
//  interpolator : 'sitkBSpline'
//  resampledPixelSpacing :
//  weightingNorm: 
//
//  imageType :
//        Original : {} 
//  featureClass :
//      glcm:
//        - 'Autocorrelation'
//        - 'JointAverage'
//        - 'ClusterProminence'
//        - 'ClusterShade'
//        - 'ClusterTendency'
//        - 'Contrast'
//        - 'Correlation'
//        - 'DifferenceAverage'
//        - 'DifferenceEntropy'
//        - 'DifferenceVariance'
//        - 'JointEnergy'
//        - 'JointEntropy'
//        - 'Imc1'
//        - 'Imc2'
//        - 'Idm'
//        - 'Idmn'
//        - 'Id'
//        - 'Idn'
//        - 'InverseVariance'
//        - 'MaximumProbability'
//        - 'SumAverage'
//        - 'SumEntropy'
//        - 'SumSquares'
//

static ref_vals_map<double> glcm_3d_pyradiomics_ref_vals
{
    {"3GLCM_ACOR", 122.14708306342365},         // Case-1_original_glcm_Autocorrelation
    {"3GLCM_ASM", 0.0143339715631298},          // Case-1_original_glcm_JointEnergy
    {"3GLCM_CLUPROM", 1870.7687419551776},      // Case-1_original_glcm_ClusterProminence
    {"3GLCM_CLUSHADE", 8.755242780815239},      // Case-1_original_glcm_ClusterShade
    {"3GLCM_CLUTEND", 23.113911920055934},      // Case-1_original_glcm_ClusterTendency
    {"3GLCM_CONTRAST", 8.76143159022662},       // Case-1_original_glcm_Contrast
    {"3GLCM_CORRELATION", 0.43309121847659515},  // Case-1_original_glcm_Correlation
    {"3GLCM_DIFAVE", 2.2143984613019545},       // Case-1_original_glcm_DifferenceAverage
    {"3GLCM_DIFENTRO", 2.645537347146111},      // Case-1_original_glcm_DifferenceEntropy
    {"3GLCM_DIFVAR", 3.4395235149928194},       // Case-1_original_glcm_DifferenceVariance
    // No 3GLCM_DIS entry: PyRadiomics deprecates Dissimilarity as equivalent to DifferenceAverage
    // and does not report it, so it is vetted through the DIS == DIFAVE identity instead.
    {"3GLCM_ID", 0.4459415317170447},          // Case-1_original_glcm_Id
    {"3GLCM_IDN", 0.9067759330416398},          // Case-1_original_glcm_Idn
    {"3GLCM_IDM", 0.3726945904589868},          // Case-1_original_glcm_Idm
    {"3GLCM_IDMN", 0.9797065356412845},         // Case-1_original_glcm_Idmn
    {"3GLCM_INFOMEAS1", -0.09924883901268647},  // Case-1_original_glcm_Imc1
    {"3GLCM_INFOMEAS2", 0.5781205730305887},    // Case-1_original_glcm_Imc2
    {"3GLCM_IV", 0.36184532347527026},          // Case-1_original_glcm_InverseVariance
    {"3GLCM_JAVE", 10.888107083238083},         // Case-1_original_glcm_JointAverage
    {"3GLCM_JE", 6.701464036118752},            // Case-1_original_glcm_JointEntropy
    // only in pyRadiomics: Case-1_original_glcm_MCC
    {"3GLCM_JMAX", 0.036309525310650057},       // 1_original_glcm_MaximumProbability
    {"3GLCM_JVAR", 7.968835877570637},          // Case-1_original_glcm_SumSquares
    {"3GLCM_SUMAVERAGE", 21.776214166476173},   // Case-1_original_glcm_SumAverage
    {"3GLCM_SUMENTROPY", 4.27263829307018}      // Case-1_original_glcm_SumEntropy
};

// Per-feature tolerances, measured rather than assumed (SPEC 7). On this fixture and config Nyxus
// and PyRadiomics agree to <= 1.2e-15 per direction on 18 of the 23 features -- they compute the
// same quantity from the same symmetric matrix -- so the default band is rel=1e-9, tight enough
// that any change of definition or of the neighbourhood walk fails it.
//
// The five entropy-family features are the exception, and the cause is in Nyxus, not in a
// convention: every log2 in the GLCM code goes through Nyxus::fast_log10 (helpers/helpers.h), a
// float-precision quadratic approximation of log2, while PyRadiomics uses numpy.log2 in double.
// The bands below are the measured worst per-direction residual of that approximation, doubled and
// then rounded up to one significant figure, so each band sits 2.4x to 2.6x its measurement:
//
//   3GLCM_INFOMEAS1   1.7e-2      3GLCM_DIFENTRO    1.2e-3      3GLCM_JE          4.1e-4
//   3GLCM_INFOMEAS2   7.7e-3      3GLCM_SUMENTROPY  7.6e-4
//
// A band this wide is a statement about fast_log10's accuracy, not slack: tightening it means
// computing the information measures in double.
static const double GLCM_3D_PYRADIOMICS_DEFAULT_TOL = 1e-9;

static const ref_vals_map<double> glcm_3d_pyradiomics_ref_tols
{
    {"3GLCM_INFOMEAS1", 4e-2},
    {"3GLCM_INFOMEAS2", 2e-2},
    {"3GLCM_DIFENTRO", 3e-3},
    {"3GLCM_SUMENTROPY", 2e-3},
    {"3GLCM_JE", 1e-3}
};

static double glcm_3d_pyradiomics_tol (const std::string& fname)
{
    auto it = glcm_3d_pyradiomics_ref_tols.find(fname);
    return it == glcm_3d_pyradiomics_ref_tols.end() ? GLCM_3D_PYRADIOMICS_DEFAULT_TOL : it->second;
}

// One PyRadiomics value per 3D direction, in Nyxus angle-slot order. Regenerate with
// tests/vetting/oracles/gen_glcm3d_pyradiomics.py, which also re-verifies every entry here.
static const ref_vals_map_by_angle<double> glcm_3d_pyradiomics_ref_vals_by_angle
{
	{0, {   // Nyxus shift (1,1,1)
		{"3GLCM_ACOR", 118.99236641221376},
		{"3GLCM_ASM", 0.012430284698005887},
		{"3GLCM_CLUPROM", 1128.9114525251982},
		{"3GLCM_CLUSHADE", 11.979938582809126},
		{"3GLCM_CLUTEND", 18.501572251323292},
		{"3GLCM_CONTRAST", 11.159958362248435},
		{"3GLCM_CORRELATION", 0.24751298187274456},
		{"3GLCM_DIFAVE", 2.6096460791117284},
		{"3GLCM_DIFENTRO", 2.9080832111187984},
		{"3GLCM_DIFVAR", 4.349705704025224},
		{"3GLCM_ID", 0.3971436208944884},
		{"3GLCM_IDM", 0.31289232350671703},
		{"3GLCM_IDMN", 0.9742455423861098},
		{"3GLCM_IDN", 0.8916188345125778},
		{"3GLCM_INFOMEAS1", -0.028849320703047295},
		{"3GLCM_INFOMEAS2", 0.42649326176558594},
		{"3GLCM_IV", 0.32099033289502626},
		{"3GLCM_JAVE", 10.82390700902151},
		{"3GLCM_JE", 6.858755520111761},
		{"3GLCM_JMAX", 0.029840388619014575},
		{"3GLCM_JVAR", 7.415382653392931},
		{"3GLCM_SUMAVERAGE", 21.64781401804303},
		{"3GLCM_SUMENTROPY", 4.136853807670587},
	}},
	{1, {   // Nyxus shift (1,1,0)
		{"3GLCM_ACOR", 128.74671419024276},
		{"3GLCM_ASM", 0.017683730845544542},
		{"3GLCM_CLUPROM", 3320.925740996756},
		{"3GLCM_CLUSHADE", -1.3639074343548505},
		{"3GLCM_CLUTEND", 31.757448928027397},
		{"3GLCM_CONTRAST", 3.2143016261973685},
		{"3GLCM_CORRELATION", 0.8161772530538037},
		{"3GLCM_DIFAVE", 1.3746936957006015},
		{"3GLCM_DIFENTRO", 2.1053341438825215},
		{"3GLCM_DIFVAR", 1.3245188691983933},
		{"3GLCM_ID", 0.5329975919973693},
		{"3GLCM_IDM", 0.48140499271375853},
		{"3GLCM_IDMN", 0.9921750252094438},
		{"3GLCM_IDN", 0.9382765544881969},
		{"3GLCM_INFOMEAS1", -0.2273211435397601},
		{"3GLCM_INFOMEAS2", 0.8971732961849341},
		{"3GLCM_IV", 0.47393810314073054},
		{"3GLCM_JAVE", 11.027734462018268},
		{"3GLCM_JE", 6.3724144970612056},
		{"3GLCM_JMAX", 0.04633548674537759},
		{"3GLCM_JVAR", 8.742937638556189},
		{"3GLCM_SUMAVERAGE", 22.05546892403654},
		{"3GLCM_SUMENTROPY", 4.513791865105922},
	}},
	{2, {   // Nyxus shift (1,1,-1)
		{"3GLCM_ACOR", 120.04542326221619},
		{"3GLCM_ASM", 0.011706266409436495},
		{"3GLCM_CLUPROM", 1256.2668919774499},
		{"3GLCM_CLUSHADE", 10.01414042456447},
		{"3GLCM_CLUTEND", 19.89623196945447},
		{"3GLCM_CONTRAST", 11.587749483826569},
		{"3GLCM_CORRELATION", 0.2638955463100125},
		{"3GLCM_DIFAVE", 2.684101858224363},
		{"3GLCM_DIFENTRO", 2.931806609921414},
		{"3GLCM_DIFVAR", 4.383346698503086},
		{"3GLCM_ID", 0.388027945710327},
		{"3GLCM_IDM", 0.300244009664749},
		{"3GLCM_IDMN", 0.9732688884804859},
		{"3GLCM_IDN", 0.8887228796769687},
		{"3GLCM_INFOMEAS1", -0.02876182285260025},
		{"3GLCM_INFOMEAS2", 0.42817383315822377},
		{"3GLCM_IV", 0.30311238914289446},
		{"3GLCM_JAVE", 10.861321403991738},
		{"3GLCM_JE", 6.940143392817866},
		{"3GLCM_JMAX", 0.030282174810736407},
		{"3GLCM_JVAR", 7.870995363320259},
		{"3GLCM_SUMAVERAGE", 21.72264280798348},
		{"3GLCM_SUMENTROPY", 4.192880744021509},
	}},
	{3, {   // Nyxus shift (1,0,1)
		{"3GLCM_ACOR", 119.53260123541529},
		{"3GLCM_ASM", 0.012314465136239069},
		{"3GLCM_CLUPROM", 1119.7693580230175},
		{"3GLCM_CLUSHADE", 11.611060729655609},
		{"3GLCM_CLUTEND", 18.340187290758795},
		{"3GLCM_CONTRAST", 11.415579958819498},
		{"3GLCM_CORRELATION", 0.23271479689495947},
		{"3GLCM_DIFAVE", 2.6427590940288264},
		{"3GLCM_DIFENTRO", 2.924951721169867},
		{"3GLCM_DIFVAR", 4.431404329747429},
		{"3GLCM_ID", 0.39463557776014874},
		{"3GLCM_IDM", 0.3090358906014264},
		{"3GLCM_IDMN", 0.9737125274467848},
		{"3GLCM_IDN", 0.8904166916854648},
		{"3GLCM_INFOMEAS1", -0.026212275164827277},
		{"3GLCM_INFOMEAS2", 0.40843164469526927},
		{"3GLCM_IV", 0.3114347734347124},
		{"3GLCM_JAVE", 10.85363761153054},
		{"3GLCM_JE", 6.871184405464778},
		{"3GLCM_JMAX", 0.03088538091969801},
		{"3GLCM_JVAR", 7.438941812394575},
		{"3GLCM_SUMAVERAGE", 21.70727522306108},
		{"3GLCM_SUMENTROPY", 4.12707201790672},
	}},
	{4, {   // Nyxus shift (1,0,0)
		{"3GLCM_ACOR", 129.03225806451604},
		{"3GLCM_ASM", 0.02261722372372403},
		{"3GLCM_CLUPROM", 3567.623107907319},
		{"3GLCM_CLUSHADE", 4.755237375422198},
		{"3GLCM_CLUTEND", 33.038378542086775},
		{"3GLCM_CONTRAST", 1.9014582412726462},
		{"3GLCM_CORRELATION", 0.8911581497611208},
		{"3GLCM_DIFAVE", 1.0238621299160406},
		{"3GLCM_DIFENTRO", 1.8180811437621478},
		{"3GLCM_DIFVAR", 0.8531645801964356},
		{"3GLCM_ID", 0.6076210466510952},
		{"3GLCM_IDM", 0.5729550260365678},
		{"3GLCM_IDMN", 0.9953212690906778},
		{"3GLCM_IDN", 0.9530756741577313},
		{"3GLCM_INFOMEAS1", -0.3201181494470871},
		{"3GLCM_INFOMEAS2", 0.948580009637562},
		{"3GLCM_IV", 0.4882026071586389},
		{"3GLCM_JAVE", 11.011268228015899},
		{"3GLCM_JE", 6.036493321822255},
		{"3GLCM_JMAX", 0.05965532479010163},
		{"3GLCM_JVAR", 8.734959195839856},
		{"3GLCM_SUMAVERAGE", 22.02253645603182},
		{"3GLCM_SUMENTROPY", 4.5461042373680085},
	}},
	{5, {   // Nyxus shift (1,0,-1)
		{"3GLCM_ACOR", 119.10603980782429},
		{"3GLCM_ASM", 0.01189692484015583},
		{"3GLCM_CLUPROM", 1267.2565622498596},
		{"3GLCM_CLUSHADE", 11.54507375573305},
		{"3GLCM_CLUTEND", 19.87450367407197},
		{"3GLCM_CONTRAST", 11.4732326698696},
		{"3GLCM_CORRELATION", 0.2680024775002947},
		{"3GLCM_DIFAVE", 2.641386410432396},
		{"3GLCM_DIFENTRO", 2.9320375010138857},
		{"3GLCM_DIFVAR", 4.496310500652661},
		{"3GLCM_ID", 0.39809433588603105},
		{"3GLCM_IDM", 0.3134385203019809},
		{"3GLCM_IDMN", 0.9735822034505024},
		{"3GLCM_IDN", 0.8905868076720418},
		{"3GLCM_INFOMEAS1", -0.03128785862327127},
		{"3GLCM_INFOMEAS2", 0.44454724467597184},
		{"3GLCM_IV", 0.31019409436018264},
		{"3GLCM_JAVE", 10.81691832532601},
		{"3GLCM_JE", 6.927011650081292},
		{"3GLCM_JMAX", 0.03088538091969801},
		{"3GLCM_JVAR", 7.836934085985395},
		{"3GLCM_SUMAVERAGE", 21.63383665065203},
		{"3GLCM_SUMENTROPY", 4.191217325511744},
	}},
	{6, {   // Nyxus shift (1,-1,1)
		{"3GLCM_ACOR", 120.29666212534062},
		{"3GLCM_ASM", 0.012383736793650544},
		{"3GLCM_CLUPROM", 1121.8044475123918},
		{"3GLCM_CLUSHADE", 12.33640576632569},
		{"3GLCM_CLUTEND", 18.404854771919002},
		{"3GLCM_CONTRAST", 11.511239782016334},
		{"3GLCM_CORRELATION", 0.23043164867239763},
		{"3GLCM_DIFAVE", 2.6563351498637595},
		{"3GLCM_DIFENTRO", 2.937294937401924},
		{"3GLCM_DIFVAR", 4.455123353614623},
		{"3GLCM_ID", 0.39531874706431924},
		{"3GLCM_IDM", 0.30877967668591055},
		{"3GLCM_IDMN", 0.9734950888162255},
		{"3GLCM_IDN", 0.8899246548271258},
		{"3GLCM_INFOMEAS1", -0.025916567960102308},
		{"3GLCM_INFOMEAS2", 0.4064821828019596},
		{"3GLCM_IV", 0.30010139694815186},
		{"3GLCM_JAVE", 10.88913487738419},
		{"3GLCM_JE", 6.878076797274805},
		{"3GLCM_JMAX", 0.03201634877384196},
		{"3GLCM_JVAR", 7.479023638483842},
		{"3GLCM_SUMAVERAGE", 21.778269754768388},
		{"3GLCM_SUMENTROPY", 4.131444086915954},
	}},
	{7, {   // Nyxus shift (1,-1,0)
		{"3GLCM_ACOR", 128.0048683337021},
		{"3GLCM_ASM", 0.017119353602338256},
		{"3GLCM_CLUPROM", 3255.0254538040913},
		{"3GLCM_CLUSHADE", 3.602028371929358},
		{"3GLCM_CLUTEND", 31.26705609875856},
		{"3GLCM_CONTRAST", 3.5202478424430206},
		{"3GLCM_CORRELATION", 0.797613068929226},
		{"3GLCM_DIFAVE", 1.4330604115954864},
		{"3GLCM_DIFENTRO", 2.174072501219632},
		{"3GLCM_DIFVAR", 1.4665856991607957},
		{"3GLCM_ID", 0.5275581583227081},
		{"3GLCM_IDM", 0.4727813620655608},
		{"3GLCM_IDMN", 0.9914481443728197},
		{"3GLCM_IDN", 0.9359807113446338},
		{"3GLCM_INFOMEAS1", -0.208018057562316},
		{"3GLCM_INFOMEAS2", 0.8805432786108223},
		{"3GLCM_IV", 0.4519936138872456},
		{"3GLCM_JAVE", 11.003098030537734},
		{"3GLCM_JE", 6.431797093249814},
		{"3GLCM_JMAX", 0.044478867005974775},
		{"3GLCM_JVAR", 8.696825985300391},
		{"3GLCM_SUMAVERAGE", 22.00619606107546},
		{"3GLCM_SUMENTROPY", 4.504430519540905},
	}},
	{8, {   // Nyxus shift (1,-1,-1)
		{"3GLCM_ACOR", 118.25583791208791},
		{"3GLCM_ASM", 0.011943638864569475},
		{"3GLCM_CLUPROM", 1265.322254604925},
		{"3GLCM_CLUSHADE", 12.977028251800995},
		{"3GLCM_CLUTEND", 19.85394228882383},
		{"3GLCM_CONTRAST", 11.239010989010989},
		{"3GLCM_CORRELATION", 0.2770702166125256},
		{"3GLCM_DIFAVE", 2.6092032967032965},
		{"3GLCM_DIFENTRO", 2.9193407059481435},
		{"3GLCM_DIFVAR", 4.431069145483637},
		{"3GLCM_ID", 0.399216998470089},
		{"3GLCM_IDM", 0.31422037508803685},
		{"3GLCM_IDMN", 0.9741449485358443},
		{"3GLCM_IDN", 0.8917326930119162},
		{"3GLCM_INFOMEAS1", -0.03287868464291647},
		{"3GLCM_INFOMEAS2", 0.4541443706042446},
		{"3GLCM_IV", 0.3118106502479268},
		{"3GLCM_JAVE", 10.775068681318684},
		{"3GLCM_JE", 6.909822562831936},
		{"3GLCM_JMAX", 0.027472527472527472},
		{"3GLCM_JVAR", 7.773238319458705},
		{"3GLCM_SUMAVERAGE", 21.55013736263736},
		{"3GLCM_SUMENTROPY", 4.189279425622912},
	}},
	{9, {   // Nyxus shift (0,1,1)
		{"3GLCM_ACOR", 118.22840755735497},
		{"3GLCM_ASM", 0.012308407138473188},
		{"3GLCM_CLUPROM", 1191.3239192283063},
		{"3GLCM_CLUSHADE", 12.95736634370189},
		{"3GLCM_CLUTEND", 19.113462312482135},
		{"3GLCM_CONTRAST", 11.268556005398107},
		{"3GLCM_CORRELATION", 0.2582088597605505},
		{"3GLCM_DIFAVE", 2.607287449392712},
		{"3GLCM_DIFENTRO", 2.9151033437820915},
		{"3GLCM_DIFVAR", 4.4706081616373545},
		{"3GLCM_ID", 0.39920055109832436},
		{"3GLCM_IDM", 0.31571843881920364},
		{"3GLCM_IDMN", 0.9740680365556306},
		{"3GLCM_IDN", 0.8918773652150931},
		{"3GLCM_INFOMEAS1", -0.031794260469524764},
		{"3GLCM_INFOMEAS2", 0.4465142775575717},
		{"3GLCM_IV", 0.32318938597428376},
		{"3GLCM_JAVE", 10.782726045883939},
		{"3GLCM_JE", 6.882615788647874},
		{"3GLCM_JMAX", 0.029183535762483132},
		{"3GLCM_JVAR", 7.595504579470061},
		{"3GLCM_SUMAVERAGE", 21.56545209176788},
		{"3GLCM_SUMENTROPY", 4.160050102012882},
	}},
	{10, {   // Nyxus shift (0,1,0)
		{"3GLCM_ACOR", 128.84674493062977},
		{"3GLCM_ASM", 0.019856896997502163},
		{"3GLCM_CLUPROM", 3463.8066371424316},
		{"3GLCM_CLUSHADE", 1.1127236962313753},
		{"3GLCM_CLUTEND", 32.310208948151924},
		{"3GLCM_CONTRAST", 2.4552828175026695},
		{"3GLCM_CORRELATION", 0.8587517280610836},
		{"3GLCM_DIFAVE", 1.1848452508004272},
		{"3GLCM_DIFENTRO", 1.9545933925411618},
		{"3GLCM_DIFVAR", 1.0514245491583416},
		{"3GLCM_ID", 0.5702421608985111},
		{"3GLCM_IDM", 0.5272674452341727},
		{"3GLCM_IDMN", 0.99398784072338},
		{"3GLCM_IDN", 0.9461953369113131},
		{"3GLCM_INFOMEAS1", -0.27249533254181346},
		{"3GLCM_INFOMEAS2", 0.9266447847591984},
		{"3GLCM_IV", 0.4855162735318248},
		{"3GLCM_JAVE", 11.01739594450373},
		{"3GLCM_JE", 6.202201857024506},
		{"3GLCM_JMAX", 0.05208110992529349},
		{"3GLCM_JVAR", 8.691372941413647},
		{"3GLCM_SUMAVERAGE", 22.034791889007465},
		{"3GLCM_SUMENTROPY", 4.530197152507132},
	}},
	{11, {   // Nyxus shift (0,1,-1)
		{"3GLCM_ACOR", 119.95337135189536},
		{"3GLCM_ASM", 0.012060054728814192},
		{"3GLCM_CLUPROM", 1184.9709138201597},
		{"3GLCM_CLUSHADE", 11.169096901611988},
		{"3GLCM_CLUTEND", 19.092893030116606},
		{"3GLCM_CONTRAST", 11.587051325058695},
		{"3GLCM_CORRELATION", 0.24464978222138645},
		{"3GLCM_DIFAVE", 2.6712512579671253},
		{"3GLCM_DIFENTRO", 2.934302401701605},
		{"3GLCM_DIFVAR", 4.451468041867758},
		{"3GLCM_ID", 0.39073881595686355},
		{"3GLCM_IDM", 0.3042950881500287},
		{"3GLCM_IDMN", 0.9733212995379383},
		{"3GLCM_IDN", 0.8893204818623451},
		{"3GLCM_INFOMEAS1", -0.027639069043303904},
		{"3GLCM_INFOMEAS2", 0.4195159556697236},
		{"3GLCM_IV", 0.3090051265444866},
		{"3GLCM_JAVE", 10.866320026836627},
		{"3GLCM_JE", 6.906957769892934},
		{"3GLCM_JMAX", 0.030526668903052667},
		{"3GLCM_JVAR", 7.669986088793828},
		{"3GLCM_SUMAVERAGE", 21.73264005367327},
		{"3GLCM_SUMENTROPY", 4.161333070316757},
	}},
	{12, {   // Nyxus shift (0,0,1)
		{"3GLCM_ACOR", 118.87078464106844},
		{"3GLCM_ASM", 0.012020646542233717},
		{"3GLCM_CLUPROM", 1176.9869056254038},
		{"3GLCM_CLUSHADE", 11.121963385167216},
		{"3GLCM_CLUTEND", 19.030114854752362},
		{"3GLCM_CONTRAST", 11.564941569282128},
		{"3GLCM_CORRELATION", 0.2439993305456308},
		{"3GLCM_DIFAVE", 2.6487479131886476},
		{"3GLCM_DIFENTRO", 2.9369838994362514},
		{"3GLCM_DIFVAR", 4.549076061660921},
		{"3GLCM_ID", 0.3964443616113065},
		{"3GLCM_IDM", 0.31199652709871534},
		{"3GLCM_IDMN", 0.9734141487308566},
		{"3GLCM_IDN", 0.8903584441759088},
		{"3GLCM_INFOMEAS1", -0.02894236461435402},
		{"3GLCM_INFOMEAS2", 0.42832330927658696},
		{"3GLCM_IV", 0.31450045791240805},
		{"3GLCM_JAVE", 10.8168614357262},
		{"3GLCM_JE", 6.901557813262764},
		{"3GLCM_JMAX", 0.028380634390651086},
		{"3GLCM_JVAR", 7.648764106008619},
		{"3GLCM_SUMAVERAGE", 21.633722871452427},
		{"3GLCM_SUMENTROPY", 4.159643455411318},
	}},
};

static std::tuple<std::string, std::string, int> get_3d_segmented_phantom()
{
    // physical paths of the phantoms
    fs::path this_fpath(__FILE__);
    fs::path pp = this_fpath.parent_path();

    fs::path f1("/data/nifti/phantoms/ut_inten.nii");
    fs::path i_phys_path = (pp.string() + f1.make_preferred().string());

    fs::path f2("/data/nifti/phantoms/ut_mask57.nii");
    fs::path m_phys_path = (pp.string() + f2.make_preferred().string());

    std::string ipath = i_phys_path.string(),
        mpath = m_phys_path.string();

    // ROI sitting in the mask phantom
    return { ipath, mpath, 57 };
}

static std::tuple<std::string, std::string, int> get_3d_compat_phantom()
{
    // physical paths of the phantoms
    fs::path this_fpath(__FILE__);
    fs::path pp = this_fpath.parent_path();

    fs::path f1("/data/nifti/compat_int/compat_int_mri.nii");
    fs::path i_phys_path = (pp.string() + f1.make_preferred().string());

    fs::path f2("/data/nifti/compat_seg/compat_seg_liver.nii");
    fs::path m_phys_path = (pp.string() + f2.make_preferred().string());

    std::string ipath = i_phys_path.string(),
        mpath = m_phys_path.string();

    return { ipath, mpath, 1 };
}

void assert_3d_glcm_feature_pyradiomics (const Nyxus::Feature3D& expecting_fcode, const std::string& fname)
{
    // (1) prepare
    
    // check that requested feature exists
    auto iter = glcm_3d_pyradiomics_ref_vals.find(fname);
    ASSERT_TRUE(iter != glcm_3d_pyradiomics_ref_vals.end());

    // get segment info
    auto [ipath, mpath, label] = get_3d_compat_phantom();
    ASSERT_TRUE(fs::exists(ipath));
    ASSERT_TRUE(fs::exists(mpath));

    // (2) mock the 3D workflow

    Environment e;

    // slide -> dataset -> prescan 
    e.dataset.dataset_props.reserve(1);
    SlideProps& sp = e.dataset.dataset_props.emplace_back(ipath, mpath);
    ASSERT_TRUE(scan_slide_props(sp, 3, e.anisoOptions, e.resultOptions.need_annotation()));
    e.dataset.update_dataset_props_extrema();

    // properties of specific ROIs sitting in 'e.uniqueLabels'
    clear_slide_rois(e.uniqueLabels, e.roiData);
    ASSERT_TRUE(gatherRoisMetrics_3D(e, 0/*slide_index*/, ipath, mpath, 0/*t_index*/));

    // voxel clouds
    std::vector<int> batch = { label };   // expecting this roi label after metrics gathering
    ASSERT_TRUE(scanTrivialRois_3D(e, batch, ipath, mpath, 0/*t_index*/));

    // buffers
    ASSERT_NO_THROW(allocateTrivialRoisBuffers_3D(batch, e.roiData, e.hostCache));

    // (3) common feature extraction settings

    Fsettings s;
    s.resize((int)NyxSetting::__COUNT__);
    s[(int)NyxSetting::SOFTNAN].rval = 0.0;
    s[(int)NyxSetting::TINY].rval = 0.0;
    s[(int)NyxSetting::SINGLEROI].bval = false;
    s[(int)NyxSetting::GREYDEPTH].ival = 100;
    s[(int)NyxSetting::PIXELSIZEUM].rval = 100;
    s[(int)NyxSetting::PIXELDISTANCE].ival = 5;
    s[(int)NyxSetting::USEGPU].bval = false;
    s[(int)NyxSetting::VERBOSLVL].ival = 0;
    s[(int)NyxSetting::IBSI].bval = false;

    // (4) GLCM-specific feature settings mocking default pyRadiomics settings

    s[(int)NyxSetting::GLCM_GREYDEPTH].ival = -20;  // intentionally negative to activate radiomics binCount-based grey-binning
    s[(int)NyxSetting::GLCM_OFFSET].ival = 1;
    s[(int)NyxSetting::GLCM_SPARSEINTENS].bval = true;

    // (5) feature extraction

    // make it find the feature code by name
    int fcode = -1;
    ASSERT_TRUE(e.theFeatureSet.find_3D_FeatureByString(fname, fcode));
    // ... and that it's the feature we expect
    ASSERT_TRUE((int)expecting_fcode == fcode);

    // extract the feature
    LR& r = e.roiData[label];
    ASSERT_NO_THROW(r.initialize_fvals());
    D3_GLCM_feature f;
    ASSERT_NO_THROW(f.calculate(r, s));

    // (6) get values

    f.save_value(r.fvals);

    // (7) verdict, one direction at a time
    //
    // The base feature holds one value per 3D angle, so every angle is compared against its own
    // PyRadiomics value. The averaged quantity is asserted separately, on the stored *_AVE feature
    // (assert_3d_glcm_ave_feature_pyradiomics below) -- averaging here instead would let
    // per-direction errors cancel and pass.
    const std::vector<double>& per_angle = r.fvals[fcode];
    ASSERT_EQ(per_angle.size(), glcm_3d_pyradiomics_ref_vals_by_angle.size());

    const double tol = glcm_3d_pyradiomics_tol(fname);
    for (int k = 0; k < (int)per_angle.size(); k++)
    {
        SCOPED_TRACE(std::string("PYRADIOMICS_ORACLE__") + fname + "__angle" + std::to_string(k));
        auto ang = glcm_3d_pyradiomics_ref_vals_by_angle.find(k);
        ASSERT_TRUE(ang != glcm_3d_pyradiomics_ref_vals_by_angle.end());
        auto golden = ang->second.find(fname);
        ASSERT_TRUE(golden != ang->second.end());
        ASSERT_NEAR(per_angle[k], golden->second, std::abs(golden->second) * tol + 1e-12);
    }
}

// Deep-dive: verify the 7 config-sensitive 3D GLCM features equal their already-vetted twins
// (numerically, same fixture/config), so they can be promoted by equivalence. Dumps calc_ave pairs.
void test_3d_glcm_equivalence_dump_pyradiomics()
{
    auto [ipath, mpath, label] = get_3d_compat_phantom();
    ASSERT_TRUE(fs::exists(ipath));
    ASSERT_TRUE(fs::exists(mpath));

    Environment e;
    e.dataset.dataset_props.reserve(1);
    SlideProps& sp = e.dataset.dataset_props.emplace_back(ipath, mpath);
    ASSERT_TRUE(scan_slide_props(sp, 3, e.anisoOptions, e.resultOptions.need_annotation()));
    e.dataset.update_dataset_props_extrema();
    clear_slide_rois(e.uniqueLabels, e.roiData);
    ASSERT_TRUE(gatherRoisMetrics_3D(e, 0, ipath, mpath, 0));
    std::vector<int> batch = { label };
    ASSERT_TRUE(scanTrivialRois_3D(e, batch, ipath, mpath, 0));
    ASSERT_NO_THROW(allocateTrivialRoisBuffers_3D(batch, e.roiData, e.hostCache));

    Fsettings s;
    s.resize((int)NyxSetting::__COUNT__);
    s[(int)NyxSetting::SOFTNAN].rval = 0.0;
    s[(int)NyxSetting::TINY].rval = 0.0;
    s[(int)NyxSetting::SINGLEROI].bval = false;
    s[(int)NyxSetting::GREYDEPTH].ival = 100;
    s[(int)NyxSetting::PIXELSIZEUM].rval = 100;
    s[(int)NyxSetting::PIXELDISTANCE].ival = 5;
    s[(int)NyxSetting::USEGPU].bval = false;
    s[(int)NyxSetting::VERBOSLVL].ival = 0;
    s[(int)NyxSetting::IBSI].bval = false;
    s[(int)NyxSetting::GLCM_GREYDEPTH].ival = -20;
    s[(int)NyxSetting::GLCM_OFFSET].ival = 1;
    s[(int)NyxSetting::GLCM_SPARSEINTENS].bval = true;

    LR& r = e.roiData[label];
    ASSERT_NO_THROW(r.initialize_fvals());
    D3_GLCM_feature f;
    ASSERT_NO_THROW(f.calculate(r, s));
    f.save_value(r.fvals);

    using F = Nyxus::Feature3D;
    struct Pair { const char* a; F fa; const char* b; F fb; };
    std::vector<Pair> pairs = {
        {"3GLCM_DIS", F::GLCM_DIS, "3GLCM_DIFAVE", F::GLCM_DIFAVE},
        {"3GLCM_ENERGY", F::GLCM_ENERGY, "3GLCM_ASM", F::GLCM_ASM},
        {"3GLCM_ENTROPY", F::GLCM_ENTROPY, "3GLCM_JE", F::GLCM_JE},
        {"3GLCM_HOM1", F::GLCM_HOM1, "3GLCM_ID", F::GLCM_ID},
        {"3GLCM_HOM2", F::GLCM_HOM2, "3GLCM_IDM", F::GLCM_IDM},
        {"3GLCM_SUMVARIANCE", F::GLCM_SUMVARIANCE, "3GLCM_CLUTEND", F::GLCM_CLUTEND},
        {"3GLCM_VARIANCE", F::GLCM_VARIANCE, "3GLCM_JVAR", F::GLCM_JVAR},
    };
    for (auto& p : pairs)
    {
        double va = f.calc_ave(r.fvals[(int)p.fa]);
        double vb = f.calc_ave(r.fvals[(int)p.fb]);
        double m = std::max(std::abs(va), std::abs(vb)); if (m < 1e-12) m = 1e-12;
        double rel = std::abs(va - vb) / m;
        std::cout << "[3DGLCM-EQ] " << p.a << "=" << va << "  " << p.b << "=" << vb
                  << "  rel=" << rel << (rel < 1e-6 ? "  EQUAL" : "  DIFFER") << "\n";
        // Config-sensitive 3D GLCM features are numerically identical to their pyradiomics-vetted
        // twins (also guards the HOM2/ENTROPY /sum_p fix: pre-fix ENTROPY!=JE, HOM2!=IDM).
        EXPECT_NEAR(va, vb, std::abs(vb) * 1e-6 + 1e-9) << p.a << " != " << p.b;
    }
}


void test_3d_glcm_acor_pyradiomics()
{
    assert_3d_glcm_feature_pyradiomics (Nyxus::Feature3D::GLCM_ACOR, "3GLCM_ACOR");
}

void test_3d_glcm_asm_pyradiomics()
{
    assert_3d_glcm_feature_pyradiomics (Nyxus::Feature3D::GLCM_ASM, "3GLCM_ASM");
}

void test_3d_glcm_cluprom_pyradiomics()
{
    assert_3d_glcm_feature_pyradiomics (Nyxus::Feature3D::GLCM_CLUPROM, "3GLCM_CLUPROM");
}

void test_3d_glcm_clushade_pyradiomics()
{
    assert_3d_glcm_feature_pyradiomics (Nyxus::Feature3D::GLCM_CLUSHADE, "3GLCM_CLUSHADE");
}

void test_3d_glcm_clutend_pyradiomics()
{
    assert_3d_glcm_feature_pyradiomics (Nyxus::Feature3D::GLCM_CLUTEND, "3GLCM_CLUTEND");
}

void test_3d_glcm_contrast_pyradiomics()
{
    assert_3d_glcm_feature_pyradiomics (Nyxus::Feature3D::GLCM_CONTRAST, "3GLCM_CONTRAST");
}

void test_3d_glcm_correlation_pyradiomics()
{
    assert_3d_glcm_feature_pyradiomics (Nyxus::Feature3D::GLCM_CORRELATION, "3GLCM_CORRELATION");
}

void test_3d_glcm_difference_average_pyradiomics()
{
    assert_3d_glcm_feature_pyradiomics (Nyxus::Feature3D::GLCM_DIFAVE, "3GLCM_DIFAVE");
}

void test_3d_glcm_difference_entropy_pyradiomics()
{
    assert_3d_glcm_feature_pyradiomics (Nyxus::Feature3D::GLCM_DIFENTRO, "3GLCM_DIFENTRO");
}

void test_3d_glcm_difference_variance_pyradiomics()
{
    assert_3d_glcm_feature_pyradiomics (Nyxus::Feature3D::GLCM_DIFVAR, "3GLCM_DIFVAR");
}

void test_3d_glcm_id_pyradiomics()
{
    assert_3d_glcm_feature_pyradiomics (Nyxus::Feature3D::GLCM_ID, "3GLCM_ID");
}

void test_3d_glcm_idn_pyradiomics()
{
    assert_3d_glcm_feature_pyradiomics (Nyxus::Feature3D::GLCM_IDN, "3GLCM_IDN");
}

void test_3d_glcm_idm_pyradiomics()
{
    assert_3d_glcm_feature_pyradiomics (Nyxus::Feature3D::GLCM_IDM, "3GLCM_IDM");
}

void test_3d_glcm_idmn_pyradiomics()
{
    assert_3d_glcm_feature_pyradiomics (Nyxus::Feature3D::GLCM_IDMN, "3GLCM_IDMN");
}

void test_3d_glcm_infomeas1_pyradiomics()
{
    assert_3d_glcm_feature_pyradiomics (Nyxus::Feature3D::GLCM_INFOMEAS1, "3GLCM_INFOMEAS1");
}

void test_3d_glcm_infomeas2_pyradiomics()
{
    assert_3d_glcm_feature_pyradiomics (Nyxus::Feature3D::GLCM_INFOMEAS2, "3GLCM_INFOMEAS2");
}

void test_3d_glcm_iv_pyradiomics()
{
    assert_3d_glcm_feature_pyradiomics (Nyxus::Feature3D::GLCM_IV, "3GLCM_IV");
}

void test_3d_glcm_jave_pyradiomics()
{
    assert_3d_glcm_feature_pyradiomics (Nyxus::Feature3D::GLCM_JAVE, "3GLCM_JAVE");
}

void test_3d_glcm_je_pyradiomics()
{
    assert_3d_glcm_feature_pyradiomics (Nyxus::Feature3D::GLCM_JE, "3GLCM_JE");
}

void test_3d_glcm_jmax_pyradiomics()
{
    assert_3d_glcm_feature_pyradiomics (Nyxus::Feature3D::GLCM_JMAX, "3GLCM_JMAX");
}

void test_3d_glcm_jvar_pyradiomics()
{
    assert_3d_glcm_feature_pyradiomics (Nyxus::Feature3D::GLCM_JVAR, "3GLCM_JVAR");
}

void test_3d_glcm_sum_average_pyradiomics()
{
    assert_3d_glcm_feature_pyradiomics (Nyxus::Feature3D::GLCM_SUMAVERAGE, "3GLCM_SUMAVERAGE");
}

void test_3d_glcm_sum_entropy_pyradiomics()
{
    assert_3d_glcm_feature_pyradiomics (Nyxus::Feature3D::GLCM_SUMENTROPY, "3GLCM_SUMENTROPY");
}



// ---------------------------------------------------------------------------------------------
// The _AVE features. PyRadiomics reports one value per feature over its whole direction set, which
// is the Nyxus *_AVE aggregation -- assert_3d_glcm_feature_pyradiomics() above already compares that
// quantity, but it recomputes it with calc_ave() and books the result against the per-angle base
// feature. Nothing checked the *_AVE features that save_value() actually writes
// (fvals[..._AVE][0] = calc_ave(...) in 3d_glcm.cpp), which is what the registry's *_AVE rows name.
//
// These read the stored feature, so a defect in how save_value populates *_AVE fails here and
// nowhere else.
// ---------------------------------------------------------------------------------------------

void assert_3d_glcm_ave_feature_pyradiomics (const Nyxus::Feature3D& ave_fcode,
    const std::string& base_fname)
{
    auto iter = glcm_3d_pyradiomics_ref_vals.find(base_fname);
    ASSERT_TRUE(iter != glcm_3d_pyradiomics_ref_vals.end()) << base_fname;

    auto [ipath, mpath, label] = get_3d_compat_phantom();
    ASSERT_TRUE(fs::exists(ipath));
    ASSERT_TRUE(fs::exists(mpath));

    Environment e;
    e.dataset.dataset_props.reserve(1);
    SlideProps& sp = e.dataset.dataset_props.emplace_back(ipath, mpath);
    ASSERT_TRUE(scan_slide_props(sp, 3, e.anisoOptions, e.resultOptions.need_annotation()));
    e.dataset.update_dataset_props_extrema();

    clear_slide_rois(e.uniqueLabels, e.roiData);
    ASSERT_TRUE(gatherRoisMetrics_3D(e, 0, ipath, mpath, 0));
    std::vector<int> batch = { label };
    ASSERT_TRUE(scanTrivialRois_3D(e, batch, ipath, mpath, 0));
    ASSERT_NO_THROW(allocateTrivialRoisBuffers_3D(batch, e.roiData, e.hostCache));

    Fsettings s;
    s.resize((int)NyxSetting::__COUNT__);
    s[(int)NyxSetting::SOFTNAN].rval = 0.0;
    s[(int)NyxSetting::TINY].rval = 0.0;
    s[(int)NyxSetting::SINGLEROI].bval = false;
    s[(int)NyxSetting::GREYDEPTH].ival = 100;
    s[(int)NyxSetting::PIXELSIZEUM].rval = 100;
    s[(int)NyxSetting::PIXELDISTANCE].ival = 5;
    s[(int)NyxSetting::USEGPU].bval = false;
    s[(int)NyxSetting::VERBOSLVL].ival = 0;
    s[(int)NyxSetting::IBSI].bval = false;
    s[(int)NyxSetting::GLCM_GREYDEPTH].ival = -20;
    s[(int)NyxSetting::GLCM_OFFSET].ival = 1;
    s[(int)NyxSetting::GLCM_SPARSEINTENS].bval = true;

    LR& r = e.roiData[label];
    ASSERT_NO_THROW(r.initialize_fvals());
    D3_GLCM_feature f;
    ASSERT_NO_THROW(f.calculate(r, s));
    f.save_value(r.fvals);

    SCOPED_TRACE(std::string("PYRADIOMICS_ORACLE__") + base_fname + "_AVE");
    const double want = glcm_3d_pyradiomics_ref_vals[base_fname];
    const double tol = glcm_3d_pyradiomics_tol(base_fname);
    ASSERT_NEAR(r.fvals[(int)ave_fcode][0], want, std::abs(want) * tol + 1e-12);
}

void test_3d_glcm_acor_ave_pyradiomics()
{
    assert_3d_glcm_ave_feature_pyradiomics (Nyxus::Feature3D::GLCM_ACOR_AVE, "3GLCM_ACOR");
}

void test_3d_glcm_asm_ave_pyradiomics()
{
    assert_3d_glcm_ave_feature_pyradiomics (Nyxus::Feature3D::GLCM_ASM_AVE, "3GLCM_ASM");
}

void test_3d_glcm_cluprom_ave_pyradiomics()
{
    assert_3d_glcm_ave_feature_pyradiomics (Nyxus::Feature3D::GLCM_CLUPROM_AVE, "3GLCM_CLUPROM");
}

void test_3d_glcm_clushade_ave_pyradiomics()
{
    assert_3d_glcm_ave_feature_pyradiomics (Nyxus::Feature3D::GLCM_CLUSHADE_AVE, "3GLCM_CLUSHADE");
}

void test_3d_glcm_clutend_ave_pyradiomics()
{
    assert_3d_glcm_ave_feature_pyradiomics (Nyxus::Feature3D::GLCM_CLUTEND_AVE, "3GLCM_CLUTEND");
}

void test_3d_glcm_contrast_ave_pyradiomics()
{
    assert_3d_glcm_ave_feature_pyradiomics (Nyxus::Feature3D::GLCM_CONTRAST_AVE, "3GLCM_CONTRAST");
}

void test_3d_glcm_correlation_ave_pyradiomics()
{
    assert_3d_glcm_ave_feature_pyradiomics (Nyxus::Feature3D::GLCM_CORRELATION_AVE, "3GLCM_CORRELATION");
}

void test_3d_glcm_difave_ave_pyradiomics()
{
    assert_3d_glcm_ave_feature_pyradiomics (Nyxus::Feature3D::GLCM_DIFAVE_AVE, "3GLCM_DIFAVE");
}

void test_3d_glcm_difentro_ave_pyradiomics()
{
    assert_3d_glcm_ave_feature_pyradiomics (Nyxus::Feature3D::GLCM_DIFENTRO_AVE, "3GLCM_DIFENTRO");
}

void test_3d_glcm_difvar_ave_pyradiomics()
{
    assert_3d_glcm_ave_feature_pyradiomics (Nyxus::Feature3D::GLCM_DIFVAR_AVE, "3GLCM_DIFVAR");
}

void test_3d_glcm_id_ave_pyradiomics()
{
    assert_3d_glcm_ave_feature_pyradiomics (Nyxus::Feature3D::GLCM_ID_AVE, "3GLCM_ID");
}

void test_3d_glcm_idm_ave_pyradiomics()
{
    assert_3d_glcm_ave_feature_pyradiomics (Nyxus::Feature3D::GLCM_IDM_AVE, "3GLCM_IDM");
}

void test_3d_glcm_idmn_ave_pyradiomics()
{
    assert_3d_glcm_ave_feature_pyradiomics (Nyxus::Feature3D::GLCM_IDMN_AVE, "3GLCM_IDMN");
}

void test_3d_glcm_idn_ave_pyradiomics()
{
    assert_3d_glcm_ave_feature_pyradiomics (Nyxus::Feature3D::GLCM_IDN_AVE, "3GLCM_IDN");
}

void test_3d_glcm_infomeas1_ave_pyradiomics()
{
    assert_3d_glcm_ave_feature_pyradiomics (Nyxus::Feature3D::GLCM_INFOMEAS1_AVE, "3GLCM_INFOMEAS1");
}

void test_3d_glcm_infomeas2_ave_pyradiomics()
{
    assert_3d_glcm_ave_feature_pyradiomics (Nyxus::Feature3D::GLCM_INFOMEAS2_AVE, "3GLCM_INFOMEAS2");
}

void test_3d_glcm_iv_ave_pyradiomics()
{
    assert_3d_glcm_ave_feature_pyradiomics (Nyxus::Feature3D::GLCM_IV_AVE, "3GLCM_IV");
}

void test_3d_glcm_jave_ave_pyradiomics()
{
    assert_3d_glcm_ave_feature_pyradiomics (Nyxus::Feature3D::GLCM_JAVE_AVE, "3GLCM_JAVE");
}

void test_3d_glcm_je_ave_pyradiomics()
{
    assert_3d_glcm_ave_feature_pyradiomics (Nyxus::Feature3D::GLCM_JE_AVE, "3GLCM_JE");
}

void test_3d_glcm_jmax_ave_pyradiomics()
{
    assert_3d_glcm_ave_feature_pyradiomics (Nyxus::Feature3D::GLCM_JMAX_AVE, "3GLCM_JMAX");
}

void test_3d_glcm_jvar_ave_pyradiomics()
{
    assert_3d_glcm_ave_feature_pyradiomics (Nyxus::Feature3D::GLCM_JVAR_AVE, "3GLCM_JVAR");
}

void test_3d_glcm_sumaverage_ave_pyradiomics()
{
    assert_3d_glcm_ave_feature_pyradiomics (Nyxus::Feature3D::GLCM_SUMAVERAGE_AVE, "3GLCM_SUMAVERAGE");
}

void test_3d_glcm_sumentropy_ave_pyradiomics()
{
    assert_3d_glcm_ave_feature_pyradiomics (Nyxus::Feature3D::GLCM_SUMENTROPY_AVE, "3GLCM_SUMENTROPY");
}


// The six *_AVE features PyRadiomics does not report under their own name. Each is numerically
// identical to a twin that does, and the identity is not an assumption: it is asserted at 1e-6 on
// the per-angle values by test_3d_glcm_equivalence_dump_pyradiomics(). This pins the same identity
// on the stored *_AVE features, so the twin's PyRadiomics golden carries both.
void test_3d_glcm_ave_equivalence_pyradiomics()
{
    auto [ipath, mpath, label] = get_3d_compat_phantom();
    ASSERT_TRUE(fs::exists(ipath));
    ASSERT_TRUE(fs::exists(mpath));

    Environment e;
    e.dataset.dataset_props.reserve(1);
    SlideProps& sp = e.dataset.dataset_props.emplace_back(ipath, mpath);
    ASSERT_TRUE(scan_slide_props(sp, 3, e.anisoOptions, e.resultOptions.need_annotation()));
    e.dataset.update_dataset_props_extrema();

    clear_slide_rois(e.uniqueLabels, e.roiData);
    ASSERT_TRUE(gatherRoisMetrics_3D(e, 0, ipath, mpath, 0));
    std::vector<int> batch = { label };
    ASSERT_TRUE(scanTrivialRois_3D(e, batch, ipath, mpath, 0));
    ASSERT_NO_THROW(allocateTrivialRoisBuffers_3D(batch, e.roiData, e.hostCache));

    Fsettings s;
    s.resize((int)NyxSetting::__COUNT__);
    s[(int)NyxSetting::SOFTNAN].rval = 0.0;
    s[(int)NyxSetting::TINY].rval = 0.0;
    s[(int)NyxSetting::SINGLEROI].bval = false;
    s[(int)NyxSetting::GREYDEPTH].ival = 100;
    s[(int)NyxSetting::PIXELSIZEUM].rval = 100;
    s[(int)NyxSetting::PIXELDISTANCE].ival = 5;
    s[(int)NyxSetting::USEGPU].bval = false;
    s[(int)NyxSetting::VERBOSLVL].ival = 0;
    s[(int)NyxSetting::IBSI].bval = false;
    s[(int)NyxSetting::GLCM_GREYDEPTH].ival = -20;
    s[(int)NyxSetting::GLCM_OFFSET].ival = 1;
    s[(int)NyxSetting::GLCM_SPARSEINTENS].bval = true;

    LR& r = e.roiData[label];
    ASSERT_NO_THROW(r.initialize_fvals());
    D3_GLCM_feature f;
    ASSERT_NO_THROW(f.calculate(r, s));
    f.save_value(r.fvals);

    using F = Nyxus::Feature3D;
    struct Pair { F a; F b; const char* an; const char* bn; const char* golden; };
    std::vector<Pair> pairs = {
        { F::GLCM_DIS_AVE, F::GLCM_DIFAVE_AVE, "3GLCM_DIS_AVE", "3GLCM_DIFAVE_AVE", "3GLCM_DIFAVE" },
        { F::GLCM_ENERGY_AVE, F::GLCM_ASM_AVE, "3GLCM_ENERGY_AVE", "3GLCM_ASM_AVE", "3GLCM_ASM" },
        { F::GLCM_ENTROPY_AVE, F::GLCM_JE_AVE, "3GLCM_ENTROPY_AVE", "3GLCM_JE_AVE", "3GLCM_JE" },
        { F::GLCM_HOM1_AVE, F::GLCM_ID_AVE, "3GLCM_HOM1_AVE", "3GLCM_ID_AVE", "3GLCM_ID" },
        { F::GLCM_SUMVARIANCE_AVE, F::GLCM_CLUTEND_AVE, "3GLCM_SUMVARIANCE_AVE", "3GLCM_CLUTEND_AVE", "3GLCM_CLUTEND" },
        { F::GLCM_VARIANCE_AVE, F::GLCM_JVAR_AVE, "3GLCM_VARIANCE_AVE", "3GLCM_JVAR_AVE", "3GLCM_JVAR" }
    };

    for (auto& p : pairs)
    {
        SCOPED_TRACE(std::string("PYRADIOMICS_ORACLE__") + p.an + " via " + p.bn);
        double va = r.fvals[(int)p.a][0], vb = r.fvals[(int)p.b][0];
        // (1) the identity holds on the stored _AVE features
        EXPECT_NEAR(va, vb, std::abs(vb) * 1e-6 + 1e-9) << p.an << " != " << p.bn;
        // (2) and the twin still matches its PyRadiomics golden, so the identity carries the claim
        ASSERT_TRUE(agrees_gt(vb, glcm_3d_pyradiomics_ref_vals[p.golden], 10.));
    }
}
