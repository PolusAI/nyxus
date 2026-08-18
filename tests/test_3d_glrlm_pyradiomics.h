#pragma once

#include <gtest/gtest.h>
#include <string>
#include <tuple>
#include <unordered_set>
#include <vector>
#include "../src/nyx/environment.h"
#include "../src/nyx/featureset.h"
#include "../src/nyx/roi_cache.h"
#include "../src/nyx/features/3d_glrlm.h"
#include "../src/nyx/helpers/fsystem.h"
#include "test_ref_vals.h"

// PyRadiomics 3.0.1 on the compat phantom (compat_int/compat_int_mri.nii +
// compat_seg/compat_seg_liver.nii, label 1) at binCount 20, recipe glrlm3d.pyradiomics_bincount20.
// Nyxus side: GREYDEPTH=100, IBSI=false, GLRLM_GREYDEPTH=-20 (negative activates radiomics
// binCount-based binning, so the magnitude is the bin count).
//
// TWO REFERENCES, because one scalar cannot vet 13 directions:
//
//   glrlm_3d_pyradiomics_ref_vals            PyRadiomics' direction-set value -> the *_AVE features
//   glrlm_3d_pyradiomics_ref_vals_by_angle   one value per direction         -> the base features
//
// The base (unsuffixed) features hold one value per 3D angle. Comparing their average against the
// direction-set scalar validates the average and nothing more, and here that is not hypothetical:
// on this fixture the two agree to 1e-16 on the mean while individual directions differ by up to
// 554% under the wrong angle mapping. The per-direction goldens come from PyRadiomics' own feature
// formulas (RadiomicsGLRLM computes each feature per angle and averages last, so intercepting that
// average yields the per-angle vector).
//
// ANGLE ORDER: slot k here is slot k in PyRadiomics -- the identity, and the OPPOSITE of 3D GLCM.
// Both tools order angles (dz, dy, dx); Nyxus' shifts13 looks identical to the GLCM table but is
// typed AngleShift { int dz, dy, dx; } (texture_feature.h) where GLCM's is
// ShiftToNeighbor { int dx, dy, dz; } (3d_glcm.cpp), so the same brace initialiser lands in
// reversed fields. GLRLM slot 4 {1,0,0} is therefore dz=1 (z) where GLCM slot 4 is dx=1 (x). Both
// families still sweep a complete set of 13 directions -- which is why their averages are right and
// this went unnoticed -- but the same slot index means different geometry in each. That is filed as
// a defect; the table below pins what Nyxus emits, in Nyxus' own slot order.
//
// Regenerate with tests/vetting/oracles/gen_glrlm3d_pyradiomics.py, which also re-verifies every
// pin. See tests/vetting/audit/glrlm_3d_golden_regen.md.

static ref_vals_map<double> glrlm_3d_pyradiomics_ref_vals
{
    {"3GLRLM_GLN", 406.68709120394277},     // Case-1_original_glrlm_GrayLevelNonUniformity
    {"3GLRLM_GLNN", 0.09722976558135092},   // Case-1_original_glrlm_GrayLevelNonUniformityNormalized
    {"3GLRLM_GLV", 9.100102904831404},      // Case-1_original_glrlm_GrayLevelVariance
    {"3GLRLM_HGLRE", 130.25347348795043},   // Case-1_original_glrlm_HighGrayLevelRunEmphasis
    {"3GLRLM_LRE", 1.5538285862328314},     // Case-1_original_glrlm_LongRunEmphasis
    {"3GLRLM_LRHGLE", 200.98033929654184},  // Case-1_original_glrlm_LongRunHighGrayLevelEmphasis
    {"3GLRLM_LRLGLE", 0.01863138831176311}, // Case-1_original_glrlm_LongRunLowGrayLevelEmphasis
    {"3GLRLM_LGLRE", 0.012578735424633676}, // Case-1_original_glrlm_LowGrayLevelRunEmphasis
    {"3GLRLM_RE", 4.228290966541947},       // Case-1_original_glrlm_RunEntropy
    {"3GLRLM_RLN", 3309.7814564084974},     // Case-1_original_glrlm_RunLengthNonUniformity
    {"3GLRLM_RLNN", 0.7807974007564221},    // Case-1_original_glrlm_RunLengthNonUniformityNormalized
    {"3GLRLM_RP", 0.8714583333333334},      // Case-1_original_glrlm_RunPercentage
    {"3GLRLM_RV", 0.19950155996777244},     // Case-1_original_glrlm_RunVariance
    {"3GLRLM_SRE", 0.9003824440228139},     // Case-1_original_glrlm_ShortRunEmphasis
    {"3GLRLM_SRHGLE", 117.56903884692184},  // Case-1_original_glrlm_ShortRunHighGrayLevelEmphasis
    {"3GLRLM_SRLGLE", 0.011465297979291003} // Case-1_original_glrlm_ShortRunLowGrayLevelEmphasis
};

// Nyxus reproduces this tool to double precision on 15 of the 16 quantities. The exception is run
// entropy, the family's only sum over logarithms, which Nyxus evaluates through fast_log10 with an
// EPSILON guard: measured 3.9e-4 away, so it is held to 5e-3 and everything else to 1e-9. Same shape
// and same cause as the 2D family. See tests/vetting/audit/glrlm_3d_pyradiomics_vetting_report.md.
static double glrlm_3d_pyradiomics_frac_tolerance (const std::string& feature_name)
{
    static const std::unordered_set<std::string> log_based { "3GLRLM_RE" };
    return log_based.count (feature_name) ? 200. : 1.e9;
}

// One PyRadiomics value per 3D direction, in Nyxus angle-slot order. Regenerate with
// tests/vetting/oracles/gen_glrlm3d_pyradiomics.py, which re-verifies every entry here.
//
// Measured agreement, per direction: 15 of the 16 features reproduce PyRadiomics to <= 5.5e-16 in
// every direction. The exception is run entropy, the family's only sum over logarithms, which Nyxus
// evaluates through fast_log10 (helpers.h) -- a float-precision approximation of log2 -- against
// PyRadiomics' double numpy.log2: measured 6.2e-4 worst per direction, held to 2e-3 below.
static const ref_vals_map_by_angle<double> glrlm_3d_pyradiomics_ref_vals_by_angle
{
	{0, {   // AngleShift dz=1 dy=1 dx=1
		{"3GLRLM_GLN", 431.5570076611086},
		{"3GLRLM_GLNN", 0.09724132664738815},
		{"3GLRLM_GLV", 9.1269362206565},
		{"3GLRLM_HGLRE", 130.39477242000902},
		{"3GLRLM_LGLRE", 0.012620283050803344},
		{"3GLRLM_LRE", 1.2564218116268588},
		{"3GLRLM_LRHGLE", 162.20166741775574},
		{"3GLRLM_LRLGLE", 0.014992683667280104},
		{"3GLRLM_RE", 4.0256355782592665},
		{"3GLRLM_RLN", 3811.7845876520955},
		{"3GLRLM_RLNN", 0.8588969327742442},
		{"3GLRLM_RP", 0.9245833333333333},
		{"3GLRLM_RV", 0.08663188030936882},
		{"3GLRLM_SRE", 0.9422866281107606},
		{"3GLRLM_SRHGLE", 123.3033273246207},
		{"3GLRLM_SRLGLE", 0.012077123007699968},
	}},
	{1, {   // AngleShift dz=1 dy=1 dx=0
		{"3GLRLM_GLN", 430.52962460425147},
		{"3GLRLM_GLNN", 0.09736083776667831},
		{"3GLRLM_GLV", 9.141368319784245},
		{"3GLRLM_HGLRE", 130.4500226142017},
		{"3GLRLM_LGLRE", 0.012629846982743642},
		{"3GLRLM_LRE", 1.2731795567616464},
		{"3GLRLM_LRHGLE", 164.13048394391677},
		{"3GLRLM_LRLGLE", 0.015175302670140103},
		{"3GLRLM_RE", 4.042182310505922},
		{"3GLRLM_RLN", 3784.555857078245},
		{"3GLRLM_RLNN", 0.8558470956757678},
		{"3GLRLM_RP", 0.92125},
		{"3GLRLM_RV", 0.09490907372597969},
		{"3GLRLM_SRE", 0.9407664958038092},
		{"3GLRLM_SRHGLE", 123.16419794964573},
		{"3GLRLM_SRLGLE", 0.012073632720583882},
	}},
	{2, {   // AngleShift dz=1 dy=1 dx=-1
		{"3GLRLM_GLN", 430.8488003621548},
		{"3GLRLM_GLNN", 0.0975212314083646},
		{"3GLRLM_GLV", 9.122894765047143},
		{"3GLRLM_HGLRE", 130.0482118605704},
		{"3GLRLM_LGLRE", 0.012662192014970562},
		{"3GLRLM_LRE", 1.2765957446808511},
		{"3GLRLM_LRHGLE", 165.36645540968763},
		{"3GLRLM_LRLGLE", 0.015149140464254116},
		{"3GLRLM_RE", 4.043743880619867},
		{"3GLRLM_RLN", 3775.3635129017657},
		{"3GLRLM_RLNN", 0.8545413112045644},
		{"3GLRLM_RP", 0.9204166666666667},
		{"3GLRLM_RV", 0.09619071410610652},
		{"3GLRLM_SRE", 0.9401721492882652},
		{"3GLRLM_SRHGLE", 122.38485551531618},
		{"3GLRLM_SRLGLE", 0.012121189922986374},
	}},
	{3, {   // AngleShift dz=1 dy=0 dx=1
		{"3GLRLM_GLN", 431.413357400722},
		{"3GLRLM_GLNN", 0.09734055898030732},
		{"3GLRLM_GLV", 9.099020528010923},
		{"3GLRLM_HGLRE", 130.22314981949458},
		{"3GLRLM_LGLRE", 0.012636764789058404},
		{"3GLRLM_LRE", 1.259927797833935},
		{"3GLRLM_LRHGLE", 162.88064079422384},
		{"3GLRLM_LRLGLE", 0.015017013102271763},
		{"3GLRLM_RE", 4.031754615044065},
		{"3GLRLM_RLN", 3793.9350180505417},
		{"3GLRLM_RLNN", 0.8560322694157358},
		{"3GLRLM_RP", 0.9233333333333333},
		{"3GLRLM_RV", 0.08696842132700805},
		{"3GLRLM_SRE", 0.9410348977135982},
		{"3GLRLM_SRHGLE", 122.80035348977135},
		{"3GLRLM_SRLGLE", 0.01209445532657953},
	}},
	{4, {   // AngleShift dz=1 dy=0 dx=0
		{"3GLRLM_GLN", 429.7341485507246},
		{"3GLRLM_GLNN", 0.09731298653775468},
		{"3GLRLM_GLV", 9.135849053507666},
		{"3GLRLM_HGLRE", 130.2355072463768},
		{"3GLRLM_LGLRE", 0.01265259567017043},
		{"3GLRLM_LRE", 1.2753623188405796},
		{"3GLRLM_LRHGLE", 164.7271286231884},
		{"3GLRLM_LRLGLE", 0.015172397369338923},
		{"3GLRLM_RE", 4.048994319529059},
		{"3GLRLM_RLN", 3760.5978260869565},
		{"3GLRLM_RLNN", 0.8515846526465028},
		{"3GLRLM_RP", 0.92},
		{"3GLRLM_RV", 0.09388783868935097},
		{"3GLRLM_SRE", 0.9389750654186795},
		{"3GLRLM_SRHGLE", 122.51553693639292},
		{"3GLRLM_SRLGLE", 0.01209640486114401},
	}},
	{5, {   // AngleShift dz=1 dy=0 dx=-1
		{"3GLRLM_GLN", 428.3653454133635},
		{"3GLRLM_GLNN", 0.09702499329861008},
		{"3GLRLM_GLV", 9.141280484911292},
		{"3GLRLM_HGLRE", 130.3057757644394},
		{"3GLRLM_LGLRE", 0.012653544858021558},
		{"3GLRLM_LRE", 1.2770101925254813},
		{"3GLRLM_LRHGLE", 164.85050962627406},
		{"3GLRLM_LRLGLE", 0.015180937762880979},
		{"3GLRLM_RE", 4.050076915499827},
		{"3GLRLM_RLN", 3759.599320498301},
		{"3GLRLM_RLNN", 0.8515513749713026},
		{"3GLRLM_RP", 0.9197916666666667},
		{"3GLRLM_RV", 0.09500044248411867},
		{"3GLRLM_SRE", 0.9389502327922485},
		{"3GLRLM_SRHGLE", 122.65652919340629},
		{"3GLRLM_SRLGLE", 0.012094771438164618},
	}},
	{6, {   // AngleShift dz=1 dy=-1 dx=1
		{"3GLRLM_GLN", 427.14816492976894},
		{"3GLRLM_GLNN", 0.09677121996596487},
		{"3GLRLM_GLV", 9.15053376731654},
		{"3GLRLM_HGLRE", 130.3948799275034},
		{"3GLRLM_LGLRE", 0.01264771359185573},
		{"3GLRLM_LRE", 1.2782057091073855},
		{"3GLRLM_LRHGLE", 164.70752152242864},
		{"3GLRLM_LRLGLE", 0.015216153214793585},
		{"3GLRLM_RE", 4.048770298148489},
		{"3GLRLM_RLN", 3758.4753058450383},
		{"3GLRLM_RLNN", 0.8514896479032711},
		{"3GLRLM_RP", 0.9195833333333333},
		{"3GLRLM_RV", 0.09566032533548052},
		{"3GLRLM_SRE", 0.9389049363137492},
		{"3GLRLM_SRHGLE", 122.77461706439105},
		{"3GLRLM_SRLGLE", 0.01208620252636292},
	}},
	{7, {   // AngleShift dz=1 dy=-1 dx=0
		{"3GLRLM_GLN", 429.9393461104848},
		{"3GLRLM_GLNN", 0.09694235538004166},
		{"3GLRLM_GLV", 9.14293135596344},
		{"3GLRLM_HGLRE", 130.24712514092445},
		{"3GLRLM_LGLRE", 0.01264795317251898},
		{"3GLRLM_LRE", 1.2613303269447576},
		{"3GLRLM_LRHGLE", 162.9652762119504},
		{"3GLRLM_LRLGLE", 0.015002729498973875},
		{"3GLRLM_RE", 4.030642280703359},
		{"3GLRLM_RLN", 3810.245546786922},
		{"3GLRLM_RLNN", 0.8591309011920907},
		{"3GLRLM_RP", 0.9239583333333333},
		{"3GLRLM_RV", 0.08995728098082156},
		{"3GLRLM_SRE", 0.9423321433045221},
		{"3GLRLM_SRHGLE", 123.00077351872729},
		{"3GLRLM_SRLGLE", 0.012128377423873645},
	}},
	{8, {   // AngleShift dz=1 dy=-1 dx=-1
		{"3GLRLM_GLN", 432.1786918408631},
		{"3GLRLM_GLNN", 0.09714063651176964},
		{"3GLRLM_GLV", 9.138830776673732},
		{"3GLRLM_HGLRE", 130.33175994605529},
		{"3GLRLM_LGLRE", 0.01262564549321631},
		{"3GLRLM_LRE", 1.2501685772083615},
		{"3GLRLM_LRHGLE", 161.33445718138907},
		{"3GLRLM_LRLGLE", 0.014910859599112656},
		{"3GLRLM_RE", 4.01831476259772},
		{"3GLRLM_RLN", 3846.40705776579},
		{"3GLRLM_RLNN", 0.8645554186931423},
		{"3GLRLM_RP", 0.926875},
		{"3GLRLM_RV", 0.0861560258792728},
		{"3GLRLM_SRE", 0.9447160410579157},
		{"3GLRLM_SRHGLE", 123.44914119277736},
		{"3GLRLM_SRLGLE", 0.012119313335221924},
	}},
	{9, {   // AngleShift dz=0 dy=1 dx=1
		{"3GLRLM_GLN", 371.99552984485933},
		{"3GLRLM_GLNN", 0.09781633706149338},
		{"3GLRLM_GLV", 8.95187898837632},
		{"3GLRLM_HGLRE", 129.92874046805153},
		{"3GLRLM_LGLRE", 0.012332983807502508},
		{"3GLRLM_LRE", 1.9226926110965028},
		{"3GLRLM_LRHGLE", 251.3565606100447},
		{"3GLRLM_LRLGLE", 0.023430392722203888},
		{"3GLRLM_RE", 4.530506956610623},
		{"3GLRLM_RLN", 2493.760978175125},
		{"3GLRLM_RLNN", 0.6557352033066329},
		{"3GLRLM_RP", 0.7922916666666666},
		{"3GLRLM_RV", 0.3296410814800915},
		{"3GLRLM_SRE", 0.837562523738569},
		{"3GLRLM_SRHGLE", 109.05832851257777},
		{"3GLRLM_SRLGLE", 0.01019743894572805},
	}},
	{10, {   // AngleShift dz=0 dy=1 dx=0
		{"3GLRLM_GLN", 347.8561797752809},
		{"3GLRLM_GLNN", 0.09771241004923621},
		{"3GLRLM_GLV", 8.952221626057314},
		{"3GLRLM_HGLRE", 130.06348314606743},
		{"3GLRLM_LGLRE", 0.012374087102161691},
		{"3GLRLM_LRE", 2.296629213483146},
		{"3GLRLM_LRHGLE", 297.44185393258425},
		{"3GLRLM_LRLGLE", 0.028343084311604795},
		{"3GLRLM_RE", 4.705379501456226},
		{"3GLRLM_RLN", 2106.1022471910114},
		{"3GLRLM_RLNN", 0.5916017548289357},
		{"3GLRLM_RP", 0.7416666666666667},
		{"3GLRLM_RV", 0.47867693473046335},
		{"3GLRLM_SRE", 0.7975976856098246},
		{"3GLRLM_SRHGLE", 103.84269482184258},
		{"3GLRLM_SRLGLE", 0.009901624924371319},
	}},
	{11, {   // AngleShift dz=0 dy=1 dx=-1
		{"3GLRLM_GLN", 366.3831134564644},
		{"3GLRLM_GLNN", 0.09667100618904073},
		{"3GLRLM_GLV", 9.173514247324928},
		{"3GLRLM_HGLRE", 130.39419525065964},
		{"3GLRLM_LGLRE", 0.012660683587338769},
		{"3GLRLM_LRE", 1.949868073878628},
		{"3GLRLM_LRHGLE", 251.75646437994723},
		{"3GLRLM_LRLGLE", 0.023137644950870558},
		{"3GLRLM_RE", 4.554426919922261},
		{"3GLRLM_RLN", 2482.450659630607},
		{"3GLRLM_RLNN", 0.655000174045015},
		{"3GLRLM_RP", 0.7895833333333333},
		{"3GLRLM_RV", 0.34586921561392636},
		{"3GLRLM_SRE", 0.8371007036059807},
		{"3GLRLM_SRHGLE", 109.51369964819703},
		{"3GLRLM_SRLGLE", 0.010761326217803426},
	}},
	{12, {   // AngleShift dz=0 dy=0 dx=1
		{"3GLRLM_GLN", 328.9828757012105},
		{"3GLRLM_GLNN", 0.09713105276091247},
		{"3GLRLM_GLV", 9.024077629178203},
		{"3GLRLM_HGLRE", 130.27753173900206},
		{"3GLRLM_LGLRE", 0.012379266399875851},
		{"3GLRLM_LRE", 2.622379687038677},
		{"3GLRLM_LRHGLE", 339.0253912016534},
		{"3GLRLM_LRLGLE", 0.031479708719195094},
		{"3GLRLM_RE", 4.837354226148626},
		{"3GLRLM_RLN", 1843.881015648066},
		{"3GLRLM_RLNN", 0.5443994731762817},
		{"3GLRLM_RP", 0.705625},
		{"3GLRLM_RV", 0.6139710449190529},
		{"3GLRLM_SRE", 0.764572269538659},
		{"3GLRLM_SRHGLE", 99.9334498423176},
		{"3GLRLM_SRLGLE", 0.009297013080263376},
	}},
};

// Per-direction bands, measured (SPEC 7). Same cause and same shape as the *_AVE bands above.
static double glrlm_3d_pyradiomics_perangle_frac_tolerance (const std::string& feature_name)
{
    static const std::unordered_set<std::string> log_based { "3GLRLM_RE" };
    return log_based.count (feature_name) ? 500. : 1.e9;   // rel 2e-3 vs rel 1e-9
}

void test_3d_glrlm_matrix_correctness_pyradiomics()
{
    // data (data and gt source: pyradiomics web page)

    std::vector<PixIntens> rawVolume =
    {
        // z=0
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        // z=1
        5, 2, 5, 4, 4,
        3, 3, 3, 1, 3,
        2, 1, 1, 1, 3,
        4, 2, 2, 2, 3,
        3, 5, 3, 3, 2,
        // z=2
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0
    };

    SimpleCube <PixIntens> D(rawVolume, 5/*width*/, 5/*height*/, 3/*depth*/);
    PixIntens zeroI = 0;
    // --- unique intensities
    std::unordered_set<PixIntens> U (rawVolume.begin(), rawVolume.end());
    U.erase (0);
    // --- sorted non-zero (i.e. non-mask) intensities
    std::vector<PixIntens> I (U.begin(), U.end());
    std::sort (I.begin(), I.end());

    // zones

    std::vector <std::pair<PixIntens, int>> zones;
    AngleShift ash = {0, 0, 1}; // layout: dz,dy,dx
    D3_GLRLM_feature::gather_rl_zones (zones, ash, D, zeroI);

    // zone stats

    int maxZoneArea = 0;    // matrix width 
    for (const std::pair<PixIntens, int>& zo : zones)
        maxZoneArea = (std::max)(maxZoneArea, zo.second);

    // GLRLM
    SimpleMatrix <int> P;
    P.allocate (maxZoneArea /*width*/, I.size() /*height*/);
    P.fill (0);

    // --iterate zones and fill the matrix
    for (const auto& zone : zones)
    {
        // row of P-matrix
        auto itr = std::find (I.begin(), I.end(), zone.first);
        int row = (int) (itr - I.begin());

        // column of P-matrix
        int col = zone.second - 1;	// need a 0-based index
        auto& k = P.xy (col, row);
        k++;
    }

    //
    // Expecting the following GLRLM as the GT:
    // 
    //               rl=1   rl=2   rl=3
    //
    // [inten=1]     1      0      1
    // [inten=1]     3      0      1
    // [inten=1]     4      1      1
    // [inten=1]     1      1      0
    // [inten=1]     3      0      0
    // 
    //
    ASSERT_TRUE(P.yx(0, 0) == 1);     ASSERT_TRUE(P.yx(0, 1) == 0);   ASSERT_TRUE(P.yx(0, 2) == 1);
    ASSERT_TRUE(P.yx(1, 0) == 3);     ASSERT_TRUE(P.yx(1, 1) == 0);   ASSERT_TRUE(P.yx(1, 2) == 1);
    ASSERT_TRUE(P.yx(2, 0) == 4);     ASSERT_TRUE(P.yx(2, 1) == 1);   ASSERT_TRUE(P.yx(2, 2) == 1);
    ASSERT_TRUE(P.yx(3, 0) == 1);     ASSERT_TRUE(P.yx(3, 1) == 1);   ASSERT_TRUE(P.yx(3, 2) == 0);
    ASSERT_TRUE(P.yx(4, 0) == 3);     ASSERT_TRUE(P.yx(4, 1) == 0);   ASSERT_TRUE(P.yx(4, 2) == 0);
}

void assert_3d_glrlm_feature_pyradiomics(const Nyxus::Feature3D& expecting_fcode, const std::string& fname)
{
    // (1) prepare

    // check that requested feature exists
    auto iter = glrlm_3d_pyradiomics_ref_vals.find(fname);
    ASSERT_TRUE(iter != glrlm_3d_pyradiomics_ref_vals.end());

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

    // (4) GLRLM-specific feature settings mocking the pyRadiomics recipe above

    s[(int)NyxSetting::GLRLM_GREYDEPTH].ival = -20;  // intentionally negative to activate radiomics binCount-based grey-binning

    // (5) feature extraction

    // make it find the feature code by name
    int fcode = -1;
    ASSERT_TRUE(e.theFeatureSet.find_3D_FeatureByString(fname, fcode));
    // ... and that it's the feature we expect
    ASSERT_TRUE((int)expecting_fcode == fcode);

    // extract the feature
    LR& r = e.roiData[label];
    ASSERT_NO_THROW(r.initialize_fvals());
    D3_GLRLM_feature f;
    ASSERT_NO_THROW(f.calculate(r, s));

    // (6) get values

    f.save_value(r.fvals);

    // (7) verdict, one direction at a time
    //
    // The base feature holds one value per 3D angle, so each angle is compared against its own
    // PyRadiomics value. Averaging here instead -- which is what this assertion used to do -- lets
    // per-direction errors cancel: the mean of these 13 values matches PyRadiomics to 1e-16 even
    // when the directions are paired up wrongly. The averaged quantity is asserted separately, on
    // the stored *_AVE feature.
    const std::vector<double>& per_angle = r.fvals[fcode];
    ASSERT_EQ(per_angle.size(), glrlm_3d_pyradiomics_ref_vals_by_angle.size());

    const double frac = glrlm_3d_pyradiomics_perangle_frac_tolerance(fname);
    for (int k = 0; k < (int)per_angle.size(); k++)
    {
        SCOPED_TRACE(std::string("PYRADIOMICS_ORACLE__") + fname + "__angle" + std::to_string(k));
        auto ang = glrlm_3d_pyradiomics_ref_vals_by_angle.find(k);
        ASSERT_TRUE(ang != glrlm_3d_pyradiomics_ref_vals_by_angle.end());
        auto golden = ang->second.find(fname);
        ASSERT_TRUE(golden != ang->second.end());
        ASSERT_TRUE(agrees_gt(per_angle[k], golden->second, frac))
            << fname << " angle " << k << " actual=" << per_angle[k]
            << " pyradiomics=" << golden->second;
    }
}

// Vet the direction-averaged (_AVE) 3D GLRLM features vs PyRadiomics. save_value stores
// fvals[X_AVE][0] = calc_ave(angled_X) -- exactly the quantity the base test asserts == PyRadiomics
// (assert_3d_glrlm_feature_pyradiomics: atot = calc_ave(fvals[X])). So reading the _AVE slot directly and
// comparing to the same GT table is a direct PyRadiomics assertion on the _AVE feature. One workflow
// run covers all 16.
void test_3d_glrlm_ave_pyradiomics()
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
    s[(int)NyxSetting::GLRLM_GREYDEPTH].ival = -20;  // radiomics binCount-based grey binning

    LR& r = e.roiData[label];
    ASSERT_NO_THROW(r.initialize_fvals());
    D3_GLRLM_feature f;
    ASSERT_NO_THROW(f.calculate(r, s));
    f.save_value(r.fvals);

    using F = Nyxus::Feature3D;
    struct AvePair { F ave; const char* gt; };
    std::vector<AvePair> aves = {
        {F::GLRLM_GLN_AVE, "3GLRLM_GLN"},     {F::GLRLM_GLNN_AVE, "3GLRLM_GLNN"},
        {F::GLRLM_GLV_AVE, "3GLRLM_GLV"},     {F::GLRLM_HGLRE_AVE, "3GLRLM_HGLRE"},
        {F::GLRLM_LGLRE_AVE, "3GLRLM_LGLRE"}, {F::GLRLM_LRE_AVE, "3GLRLM_LRE"},
        {F::GLRLM_LRHGLE_AVE, "3GLRLM_LRHGLE"}, {F::GLRLM_LRLGLE_AVE, "3GLRLM_LRLGLE"},
        {F::GLRLM_RLN_AVE, "3GLRLM_RLN"},     {F::GLRLM_RLNN_AVE, "3GLRLM_RLNN"},
        {F::GLRLM_RV_AVE, "3GLRLM_RV"},       {F::GLRLM_SRE_AVE, "3GLRLM_SRE"},
        {F::GLRLM_SRHGLE_AVE, "3GLRLM_SRHGLE"}, {F::GLRLM_SRLGLE_AVE, "3GLRLM_SRLGLE"},
        // RP is only in its mathematical bound [0,1] at this recipe's binCount binning; it exceeds 1
        // at positive GLRLM_GREYDEPTH values (see the audit report), so this row vets it here and
        // says nothing about other configs.
        {F::GLRLM_RP_AVE, "3GLRLM_RP"},       {F::GLRLM_RE_AVE, "3GLRLM_RE"},
    };

    // Every 3D GLRLM feature the build exposes has to be asserted here -- the per-angle ones through
    // the golden table, the aggregated ones through the list above. Without this, a feature added to
    // the family later is vetted by nothing while this test still passes over the entries it has.
    std::unordered_set<int> covered_aves;
    for (const auto& a : aves)
        covered_aves.insert((int)a.ave);
    for (const auto& [name, code] : Nyxus::UserFacing_3D_featureNames)
    {
        if (name.rfind("3GLRLM_", 0) != 0)
            continue;
        const std::string suffix = "_AVE";
        bool is_ave = name.size() > suffix.size() &&
            name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0;
        if (is_ave)
            ASSERT_TRUE(covered_aves.count((int)code)) << name << " is not asserted by this test";
        else
            ASSERT_TRUE(glrlm_3d_pyradiomics_ref_vals.count(name)) << name << " has no pyradiomics golden";
    }

    for (auto& a : aves)
    {
        double v = r.fvals[(int)a.ave][0];
        ASSERT_TRUE(agrees_gt(v, glrlm_3d_pyradiomics_ref_vals[a.gt],
                              glrlm_3d_pyradiomics_frac_tolerance(a.gt))) << a.gt << "_AVE = " << v
            << " vs pyradiomics " << glrlm_3d_pyradiomics_ref_vals[a.gt];
    }
}

void test_3d_glrlm_gln_pyradiomics() {
    assert_3d_glrlm_feature_pyradiomics (Nyxus::Feature3D::GLRLM_GLN, "3GLRLM_GLN");
}

void test_3d_glrlm_glnn_pyradiomics() {
    assert_3d_glrlm_feature_pyradiomics (Nyxus::Feature3D::GLRLM_GLNN, "3GLRLM_GLNN");
}

void test_3d_glrlm_glv_pyradiomics() {
    assert_3d_glrlm_feature_pyradiomics (Nyxus::Feature3D::GLRLM_GLV, "3GLRLM_GLV");
}

void test_3d_glrlm_hglre_pyradiomics() {
    assert_3d_glrlm_feature_pyradiomics (Nyxus::Feature3D::GLRLM_HGLRE, "3GLRLM_HGLRE");
}

void test_3d_glrlm_lre_pyradiomics() {
    assert_3d_glrlm_feature_pyradiomics (Nyxus::Feature3D::GLRLM_LRE, "3GLRLM_LRE");
}

void test_3d_glrlm_lrhgle_pyradiomics() {
    assert_3d_glrlm_feature_pyradiomics (Nyxus::Feature3D::GLRLM_LRHGLE, "3GLRLM_LRHGLE");
}

void test_3d_glrlm_lrlgle_pyradiomics() {
    assert_3d_glrlm_feature_pyradiomics (Nyxus::Feature3D::GLRLM_LRLGLE, "3GLRLM_LRLGLE");
}

void test_3d_glrlm_lglre_pyradiomics() {
    assert_3d_glrlm_feature_pyradiomics (Nyxus::Feature3D::GLRLM_LGLRE, "3GLRLM_LGLRE");
}

void test_3d_glrlm_re_pyradiomics() {
    assert_3d_glrlm_feature_pyradiomics (Nyxus::Feature3D::GLRLM_RE, "3GLRLM_RE");
}

void test_3d_glrlm_rln_pyradiomics() {
    assert_3d_glrlm_feature_pyradiomics (Nyxus::Feature3D::GLRLM_RLN, "3GLRLM_RLN");
}

void test_3d_glrlm_rlnn_pyradiomics() {
    assert_3d_glrlm_feature_pyradiomics (Nyxus::Feature3D::GLRLM_RLNN, "3GLRLM_RLNN");
}

void test_3d_glrlm_rp_pyradiomics() {
    assert_3d_glrlm_feature_pyradiomics (Nyxus::Feature3D::GLRLM_RP, "3GLRLM_RP");
}

void test_3d_glrlm_rv_pyradiomics() {
    assert_3d_glrlm_feature_pyradiomics (Nyxus::Feature3D::GLRLM_RV, "3GLRLM_RV");
}

void test_3d_glrlm_sre_pyradiomics() {
    assert_3d_glrlm_feature_pyradiomics (Nyxus::Feature3D::GLRLM_SRE, "3GLRLM_SRE");
}

void test_3d_glrlm_srhgle_pyradiomics() {
    assert_3d_glrlm_feature_pyradiomics (Nyxus::Feature3D::GLRLM_SRHGLE, "3GLRLM_SRHGLE");
}

void test_3d_glrlm_srlgle_pyradiomics() {
    assert_3d_glrlm_feature_pyradiomics (Nyxus::Feature3D::GLRLM_SRLGLE, "3GLRLM_SRLGLE");
}








