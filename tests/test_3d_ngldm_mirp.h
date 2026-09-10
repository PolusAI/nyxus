#pragma once

#include "test_3d_ngldm_common.h"   // the fixture, and the <string> / <vector> / gtest it brings with it
#include "test_ref_vals.h"          // ref_vals_map

// ---------------------------------------------------------------------------------------------------
// MIRP-oracle'd 3D NGLDM: the sixteen features MIRP computes that can discriminate, asserted at the
// recipe where the two tools share a grey-level ladder (registry: mirp / vetted). The fixture lives
// in test_3d_ngldm_common.h and is the same Nyxus run the regression pins judge.
//
// WHAT THIS ORACLE COVERS, AND WHAT IT DOES NOT (SPEC 4 -- an oracle may cover one stage of a
// pipeline as long as the scope is stated): it covers the NGLD matrix -- which voxels are centres,
// which neighbours count, how a dependence count maps to a matrix column -- and the sixteen feature
// formulas evaluated over that matrix. It does NOT cover the discretisation. The grey levels are an
// input to the comparison rather than a result of it: the generator reproduces Nyxus' binning in
// numpy and hands the resulting levels to MIRP. Nyxus' own binning is judged nowhere here.
//
// Three of the family's nineteen features are deliberately absent and stay regression-only:
//   3NGLDM_DCP  -- Nyxus hard-codes it to 1 and MIRP returns 1 on any input where every voxel has a
//                  same-level neighbour, so an assertion here could not fail for the reasons an
//                  assertion should fail.
//   3NGLDM_GLM  -- MIRP's NGLDM emits no gl_mean column.
//   3NGLDM_DCM  -- MIRP's NGLDM emits no dc_mean column.
// ---------------------------------------------------------------------------------------------------

// ORACLE goldens -- MIRP NGLDM over the grey levels Nyxus bins to.
//
// Provenance (SPEC 6.4):
//   tool         = mirp 2.6.0 (numpy 2.4.6, pandas 3.0.3, Python 3.11)
//   config       = by_slice=false, base_feature_families="ngldm",
//                  base_discretisation_method="none", distance 1, difference level (alpha) 0,
//                  native 1x1x1 spacing; image = the grey levels Nyxus' GREYDEPTH=64 binning
//                  produces (21-64 on this fixture, 44 distinct)
//   fixture      = tests/data/nifti/phantoms/ut_inten.nii + ut_mask57.nii, label 57
//   recipe       = ngldm3d.mirp_samelevels
//   generator    = tests/vetting/oracles/gen_ngldm3d_mirp.py (re-verifies every pin below)
//
// MIRP suffixes each column with the neighbourhood and discretisation it was computed at; with the
// discretisation switched off that suffix is `_d1_a0.0_3d`, which is what the comment on each pin
// names. A column read at another config would carry `_fbn_n64` and is a different measurement --
// see ngldm3d.mirp_fbn64 in config_recipes.md, where the two tools do NOT share a level ladder.
static const ref_vals_map<double> ngldm_3d_mirp_ref_vals{
		{ "3NGLDM_LDE",	0.15365019670462557 },		// ngl_lde_d1_a0.0_3d
		{ "3NGLDM_HDE",	40.639400652985074 },		// ngl_hde_d1_a0.0_3d
		{ "3NGLDM_LGLCE",	0.0007839081761690363 },	// ngl_lgce_d1_a0.0_3d
		{ "3NGLDM_HGLCE",	1873.2488631063434 },		// ngl_hgce_d1_a0.0_3d
		{ "3NGLDM_LDLGLE",	7.802755638328767e-05 },	// ngl_ldlge_d1_a0.0_3d
		{ "3NGLDM_LDHGLE",	375.7076934248004 },		// ngl_ldhge_d1_a0.0_3d
		{ "3NGLDM_HDLGLE",	0.056243030790977366 },		// ngl_hdlge_d1_a0.0_3d
		{ "3NGLDM_HDHGLE",	44248.655200559704 },		// ngl_hdhge_d1_a0.0_3d
		{ "3NGLDM_GLNU",	6480.479944029851 },		// ngl_glnu_d1_a0.0_3d
		{ "3NGLDM_GLNUN",	0.023614155579633027 },		// ngl_glnu_norm_d1_a0.0_3d
		{ "3NGLDM_DCNU",	32085.42817164179 },		// ngl_dcnu_d1_a0.0_3d
		{ "3NGLDM_DCNUN",	0.11691576846592887 },		// ngl_dcnu_norm_d1_a0.0_3d
		{ "3NGLDM_GLV",	153.09596803358815 },		// ngl_gl_var_d1_a0.0_3d
		{ "3NGLDM_DCV",	14.616434951751636 },		// ngl_dc_var_d1_a0.0_3d
		{ "3NGLDM_DCENT",	8.405685600334046 },		// ngl_dc_entr_d1_a0.0_3d
		{ "3NGLDM_DCENE",	0.0034750116033420235 }		// ngl_dc_energy_d1_a0.0_3d
};

// frac_tolerance = 1e3, i.e. rel=1e-3: SPEC 7's same-definition-same-binning tier. Both sides
// evaluate the IBSI NGLDM formulas over one grey-level ladder, so only float and aggregation order
// separate them, and the band is not absorbing a definitional difference.
//
// The measured residual is far smaller -- worst 8.7e-16 over the sixteen on Windows/MSVC, recorded
// feature by feature in tests/vetting/audit/ngldm_3d_mirp_vetting_report.md. The band is left at the
// tier rather than tightened to match that measurement because it has to hold on every CI platform,
// and arm64 float divergence in aggregation-heavy sums is a thing this repo has been bitten by and
// has no local detector for.
//
// What these assertions discriminate, per failure mode, since not every one catches everything:
//   - NGLDM centres taken over the ROI's bounding box instead of the ROI: every one of the sixteen
//     fails. Ns changes from 274432 to 511360, and GLNU/GLNUN and DCNU/DCNUN each equal Ns exactly.
//   - a 24-shift neighbourhood missing the two pure-axial voxels: every dependence count changes, so
//     the four dependence-weighted features (LDE, HDE, DCNU, DCV) and the entropy/energy pair fail.
//   - the dependence count read as the matrix column j instead of j+1: LDE, HDE, LDLGLE and DCV
//     fail; the grey-level-only features (LGLCE, HGLCE) do not, which is why the set is asserted and
//     not a representative few.
//   - GLNU aggregated over anything but the grey-level row marginal: GLNU and GLNUN fail.
//   - the grey level taken as the row index i+1 instead of the LUT value U[i]: GLV fails by 2.7x,
//     and every grey-level-weighted feature moves.
//
// The band is not vacuous at that width: perturbing the 3NGLDM_GLNU golden by 0.5% -- a change small
// enough to read as a rounding difference -- fails TEST_3D_NGLDM_GLNU_MIRP.
static void assert_3d_ngldm_feature_mirp (const std::string& fname, const Nyxus::Feature3D& expecting_fcode)
{
	SCOPED_TRACE(std::string("MIRP_ORACLE__") + fname);
	ASSERT_TRUE(ngldm_3d_mirp_ref_vals.count(fname) > 0) << fname;

	double actual = 0.0;
	calculate_3d_ngldm_feature_value (fname, expecting_fcode, actual);

	ASSERT_TRUE(agrees_gt(actual, ngldm_3d_mirp_ref_vals.at(fname), 1e3))
		<< fname << " actual=" << std::setprecision(17) << actual
		<< " mirp=" << ngldm_3d_mirp_ref_vals.at(fname);
}

void test_3d_ngldm_lde_mirp() {
	assert_3d_ngldm_feature_mirp ("3NGLDM_LDE", Feature3D::NGLDM_LDE);
}

void test_3d_ngldm_hde_mirp() {
	assert_3d_ngldm_feature_mirp ("3NGLDM_HDE", Feature3D::NGLDM_HDE);
}

void test_3d_ngldm_lglce_mirp() {
	assert_3d_ngldm_feature_mirp ("3NGLDM_LGLCE", Feature3D::NGLDM_LGLCE);
}

void test_3d_ngldm_hglce_mirp() {
	assert_3d_ngldm_feature_mirp ("3NGLDM_HGLCE", Feature3D::NGLDM_HGLCE);
}

void test_3d_ngldm_ldlgle_mirp() {
	assert_3d_ngldm_feature_mirp ("3NGLDM_LDLGLE", Feature3D::NGLDM_LDLGLE);
}

void test_3d_ngldm_ldhgle_mirp() {
	assert_3d_ngldm_feature_mirp ("3NGLDM_LDHGLE", Feature3D::NGLDM_LDHGLE);
}

void test_3d_ngldm_hdlgle_mirp() {
	assert_3d_ngldm_feature_mirp ("3NGLDM_HDLGLE", Feature3D::NGLDM_HDLGLE);
}

void test_3d_ngldm_hdhgle_mirp() {
	assert_3d_ngldm_feature_mirp ("3NGLDM_HDHGLE", Feature3D::NGLDM_HDHGLE);
}

void test_3d_ngldm_glnu_mirp() {
	assert_3d_ngldm_feature_mirp ("3NGLDM_GLNU", Feature3D::NGLDM_GLNU);
}

void test_3d_ngldm_glnun_mirp() {
	assert_3d_ngldm_feature_mirp ("3NGLDM_GLNUN", Feature3D::NGLDM_GLNUN);
}

void test_3d_ngldm_dcnu_mirp() {
	assert_3d_ngldm_feature_mirp ("3NGLDM_DCNU", Feature3D::NGLDM_DCNU);
}

void test_3d_ngldm_dcnun_mirp() {
	assert_3d_ngldm_feature_mirp ("3NGLDM_DCNUN", Feature3D::NGLDM_DCNUN);
}

void test_3d_ngldm_glv_mirp() {
	assert_3d_ngldm_feature_mirp ("3NGLDM_GLV", Feature3D::NGLDM_GLV);
}

void test_3d_ngldm_dcv_mirp() {
	assert_3d_ngldm_feature_mirp ("3NGLDM_DCV", Feature3D::NGLDM_DCV);
}

void test_3d_ngldm_dcent_mirp() {
	assert_3d_ngldm_feature_mirp ("3NGLDM_DCENT", Feature3D::NGLDM_DCENT);
}

void test_3d_ngldm_dcene_mirp() {
	assert_3d_ngldm_feature_mirp ("3NGLDM_DCENE", Feature3D::NGLDM_DCENE);
}


// ---------------------------------------------------------------------------------------------------
// The family's SECOND config point, IBSI=true, at recipe ngldm3d.mirp_ibsi_rawlevels.
//
// IBSI reaches to_grayscale as disable_binning, so this point does not bin at all: the raw
// (loader-shifted) intensity IS the grey level, 2001 distinct of them on this fixture. That makes
// MIRP config-matched by construction at base_discretisation_method="none" over the same raw
// values -- there is no discretisation to disagree about, so unlike ngldm3d.mirp_samelevels above
// this recipe needs no numpy reproduction of a Nyxus binning step and its scope is correspondingly
// wider.
//
// It is a WEAK discriminator and that is measured, not assumed: 83.46% of ROI voxels have no
// matching neighbour at raw resolution and the maximum dependence reached is 17 of a possible 26,
// so the dependence distribution is concentrated in the first column. It is asserted anyway because
// SPEC 5.1 maps a VALID cell to an oracle assertion, and because a weak discriminator is still a
// second config point: the 26-neighbourhood and the ROI masking are exercised here on a level
// ladder 2001 wide rather than 44.
//
// Provenance (SPEC 6.4): same tool, fixture and generator as the table above;
//   config = by_slice=false, base_feature_families="ngldm", base_discretisation_method="none",
//            distance 1, alpha 0, native 1x1x1 spacing; image = the raw shifted intensities
//   recipe = ngldm3d.mirp_ibsi_rawlevels
//
// Measured agreement: worst rel 7.54e-15 over the sixteen -- looser than the samelevels run's
// 8.7e-16 because the sums run over 2001 grey rows rather than 44, and still four orders inside
// the rel=1e-3 band. 3NGLDM_DCP is excluded here for the same reason as above: MIRP returns
// ngl_dc_perc = 1.0 at this config too.
// ---------------------------------------------------------------------------------------------------
static const ref_vals_map<double> ngldm_3d_mirp_ibsi_ref_vals{
		{ "3NGLDM_LDE",	0.8699611118999319 },		// ngl_lde_d1_a0.0_3d
		{ "3NGLDM_HDE",	2.1725163246268657 },		// ngl_hde_d1_a0.0_3d
		{ "3NGLDM_LGLCE",	3.400101714136789e-07 },		// ngl_lgce_d1_a0.0_3d
		{ "3NGLDM_HGLCE",	4275550.791365438 },		// ngl_hgce_d1_a0.0_3d
		{ "3NGLDM_LDLGLE",	2.8075039742774145e-07 },		// ngl_ldlge_d1_a0.0_3d
		{ "3NGLDM_LDHGLE",	3831567.8645316567 },		// ngl_ldhge_d1_a0.0_3d
		{ "3NGLDM_HDLGLE",	9.362437020082865e-07 },		// ngl_hdlge_d1_a0.0_3d
		{ "3NGLDM_HDHGLE",	7987719.3384444965 },		// ngl_hdhge_d1_a0.0_3d
		{ "3NGLDM_GLNU",	157.9715485074627 },		// ngl_glnu_d1_a0.0_3d
		{ "3NGLDM_GLNUN",	0.0005756309341019367 },		// ngl_glnu_norm_d1_a0.0_3d
		{ "3NGLDM_DCNU",	196010.87360074627 },		// ngl_dcnu_d1_a0.0_3d
		{ "3NGLDM_DCNUN",	0.7142420475773462 },		// ngl_dcnu_norm_d1_a0.0_3d
		{ "3NGLDM_GLV",	341996.30156539247 },		// ngl_gl_var_d1_a0.0_3d
		{ "3NGLDM_DCV",	0.6119105729315323 },		// ngl_dc_var_d1_a0.0_3d
		{ "3NGLDM_DCENT",	11.490700886856692 },		// ngl_dc_entr_d1_a0.0_3d
		{ "3NGLDM_DCENE",	0.00040237620509422337 }		// ngl_dc_energy_d1_a0.0_3d
};

// Same band and the same reasoning as the samelevels assertions above: SPEC 7's
// same-definition tier, left at the tier rather than tightened to the measurement because it has
// to hold on every CI platform's float.
static void assert_3d_ngldm_feature_ibsi_mirp (const std::string& fname, const Nyxus::Feature3D& expecting_fcode)
{
	SCOPED_TRACE(std::string("MIRP_ORACLE_IBSI__") + fname);
	ASSERT_TRUE(ngldm_3d_mirp_ibsi_ref_vals.count(fname) > 0) << fname;

	double actual = 0.0;
	calculate_3d_ngldm_feature_value (fname, expecting_fcode, actual, true/*ibsi*/);

	ASSERT_TRUE(agrees_gt(actual, ngldm_3d_mirp_ibsi_ref_vals.at(fname), 1e3))
		<< fname << " actual=" << std::setprecision(17) << actual
		<< " mirp=" << ngldm_3d_mirp_ibsi_ref_vals.at(fname);
}

void test_3d_ngldm_lde_ibsi_mirp() {
	assert_3d_ngldm_feature_ibsi_mirp ("3NGLDM_LDE", Feature3D::NGLDM_LDE);
}

void test_3d_ngldm_hde_ibsi_mirp() {
	assert_3d_ngldm_feature_ibsi_mirp ("3NGLDM_HDE", Feature3D::NGLDM_HDE);
}

void test_3d_ngldm_lglce_ibsi_mirp() {
	assert_3d_ngldm_feature_ibsi_mirp ("3NGLDM_LGLCE", Feature3D::NGLDM_LGLCE);
}

void test_3d_ngldm_hglce_ibsi_mirp() {
	assert_3d_ngldm_feature_ibsi_mirp ("3NGLDM_HGLCE", Feature3D::NGLDM_HGLCE);
}

void test_3d_ngldm_ldlgle_ibsi_mirp() {
	assert_3d_ngldm_feature_ibsi_mirp ("3NGLDM_LDLGLE", Feature3D::NGLDM_LDLGLE);
}

void test_3d_ngldm_ldhgle_ibsi_mirp() {
	assert_3d_ngldm_feature_ibsi_mirp ("3NGLDM_LDHGLE", Feature3D::NGLDM_LDHGLE);
}

void test_3d_ngldm_hdlgle_ibsi_mirp() {
	assert_3d_ngldm_feature_ibsi_mirp ("3NGLDM_HDLGLE", Feature3D::NGLDM_HDLGLE);
}

void test_3d_ngldm_hdhgle_ibsi_mirp() {
	assert_3d_ngldm_feature_ibsi_mirp ("3NGLDM_HDHGLE", Feature3D::NGLDM_HDHGLE);
}

void test_3d_ngldm_glnu_ibsi_mirp() {
	assert_3d_ngldm_feature_ibsi_mirp ("3NGLDM_GLNU", Feature3D::NGLDM_GLNU);
}

void test_3d_ngldm_glnun_ibsi_mirp() {
	assert_3d_ngldm_feature_ibsi_mirp ("3NGLDM_GLNUN", Feature3D::NGLDM_GLNUN);
}

void test_3d_ngldm_dcnu_ibsi_mirp() {
	assert_3d_ngldm_feature_ibsi_mirp ("3NGLDM_DCNU", Feature3D::NGLDM_DCNU);
}

void test_3d_ngldm_dcnun_ibsi_mirp() {
	assert_3d_ngldm_feature_ibsi_mirp ("3NGLDM_DCNUN", Feature3D::NGLDM_DCNUN);
}

void test_3d_ngldm_glv_ibsi_mirp() {
	assert_3d_ngldm_feature_ibsi_mirp ("3NGLDM_GLV", Feature3D::NGLDM_GLV);
}

void test_3d_ngldm_dcv_ibsi_mirp() {
	assert_3d_ngldm_feature_ibsi_mirp ("3NGLDM_DCV", Feature3D::NGLDM_DCV);
}

void test_3d_ngldm_dcent_ibsi_mirp() {
	assert_3d_ngldm_feature_ibsi_mirp ("3NGLDM_DCENT", Feature3D::NGLDM_DCENT);
}

void test_3d_ngldm_dcene_ibsi_mirp() {
	assert_3d_ngldm_feature_ibsi_mirp ("3NGLDM_DCENE", Feature3D::NGLDM_DCENE);
}

// Regenerates the IBSI-mode goldens at full precision, in the shape ngldm_3d_mirp_ibsi_ref_vals
// wants. Run with
//     runAllTests --gtest_filter=*3D_NGLDM_DUMP_IBSI_MIRP*
void test_3d_ngldm_dump_ibsi_mirp()
{
	std::cout << "[3DNGLDM-IBSI-REGEN]\n";
	{ double v = 0.0; calculate_3d_ngldm_feature_value ("3NGLDM_LDE", Feature3D::NGLDM_LDE, v, true);
	  std::cout << "[3DNGLDM-IBSI-REGEN]\t\t{ \"3NGLDM_LDE\",\t" << std::setprecision(17) << v << " },\n"; }
	{ double v = 0.0; calculate_3d_ngldm_feature_value ("3NGLDM_HDE", Feature3D::NGLDM_HDE, v, true);
	  std::cout << "[3DNGLDM-IBSI-REGEN]\t\t{ \"3NGLDM_HDE\",\t" << std::setprecision(17) << v << " },\n"; }
	{ double v = 0.0; calculate_3d_ngldm_feature_value ("3NGLDM_LGLCE", Feature3D::NGLDM_LGLCE, v, true);
	  std::cout << "[3DNGLDM-IBSI-REGEN]\t\t{ \"3NGLDM_LGLCE\",\t" << std::setprecision(17) << v << " },\n"; }
	{ double v = 0.0; calculate_3d_ngldm_feature_value ("3NGLDM_HGLCE", Feature3D::NGLDM_HGLCE, v, true);
	  std::cout << "[3DNGLDM-IBSI-REGEN]\t\t{ \"3NGLDM_HGLCE\",\t" << std::setprecision(17) << v << " },\n"; }
	{ double v = 0.0; calculate_3d_ngldm_feature_value ("3NGLDM_LDLGLE", Feature3D::NGLDM_LDLGLE, v, true);
	  std::cout << "[3DNGLDM-IBSI-REGEN]\t\t{ \"3NGLDM_LDLGLE\",\t" << std::setprecision(17) << v << " },\n"; }
	{ double v = 0.0; calculate_3d_ngldm_feature_value ("3NGLDM_LDHGLE", Feature3D::NGLDM_LDHGLE, v, true);
	  std::cout << "[3DNGLDM-IBSI-REGEN]\t\t{ \"3NGLDM_LDHGLE\",\t" << std::setprecision(17) << v << " },\n"; }
	{ double v = 0.0; calculate_3d_ngldm_feature_value ("3NGLDM_HDLGLE", Feature3D::NGLDM_HDLGLE, v, true);
	  std::cout << "[3DNGLDM-IBSI-REGEN]\t\t{ \"3NGLDM_HDLGLE\",\t" << std::setprecision(17) << v << " },\n"; }
	{ double v = 0.0; calculate_3d_ngldm_feature_value ("3NGLDM_HDHGLE", Feature3D::NGLDM_HDHGLE, v, true);
	  std::cout << "[3DNGLDM-IBSI-REGEN]\t\t{ \"3NGLDM_HDHGLE\",\t" << std::setprecision(17) << v << " },\n"; }
	{ double v = 0.0; calculate_3d_ngldm_feature_value ("3NGLDM_GLNU", Feature3D::NGLDM_GLNU, v, true);
	  std::cout << "[3DNGLDM-IBSI-REGEN]\t\t{ \"3NGLDM_GLNU\",\t" << std::setprecision(17) << v << " },\n"; }
	{ double v = 0.0; calculate_3d_ngldm_feature_value ("3NGLDM_GLNUN", Feature3D::NGLDM_GLNUN, v, true);
	  std::cout << "[3DNGLDM-IBSI-REGEN]\t\t{ \"3NGLDM_GLNUN\",\t" << std::setprecision(17) << v << " },\n"; }
	{ double v = 0.0; calculate_3d_ngldm_feature_value ("3NGLDM_DCNU", Feature3D::NGLDM_DCNU, v, true);
	  std::cout << "[3DNGLDM-IBSI-REGEN]\t\t{ \"3NGLDM_DCNU\",\t" << std::setprecision(17) << v << " },\n"; }
	{ double v = 0.0; calculate_3d_ngldm_feature_value ("3NGLDM_DCNUN", Feature3D::NGLDM_DCNUN, v, true);
	  std::cout << "[3DNGLDM-IBSI-REGEN]\t\t{ \"3NGLDM_DCNUN\",\t" << std::setprecision(17) << v << " },\n"; }
	{ double v = 0.0; calculate_3d_ngldm_feature_value ("3NGLDM_GLV", Feature3D::NGLDM_GLV, v, true);
	  std::cout << "[3DNGLDM-IBSI-REGEN]\t\t{ \"3NGLDM_GLV\",\t" << std::setprecision(17) << v << " },\n"; }
	{ double v = 0.0; calculate_3d_ngldm_feature_value ("3NGLDM_DCV", Feature3D::NGLDM_DCV, v, true);
	  std::cout << "[3DNGLDM-IBSI-REGEN]\t\t{ \"3NGLDM_DCV\",\t" << std::setprecision(17) << v << " },\n"; }
	{ double v = 0.0; calculate_3d_ngldm_feature_value ("3NGLDM_DCENT", Feature3D::NGLDM_DCENT, v, true);
	  std::cout << "[3DNGLDM-IBSI-REGEN]\t\t{ \"3NGLDM_DCENT\",\t" << std::setprecision(17) << v << " },\n"; }
	{ double v = 0.0; calculate_3d_ngldm_feature_value ("3NGLDM_DCENE", Feature3D::NGLDM_DCENE, v, true);
	  std::cout << "[3DNGLDM-IBSI-REGEN]\t\t{ \"3NGLDM_DCENE\",\t" << std::setprecision(17) << v << " },\n"; }
}
