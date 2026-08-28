#pragma once

// 3D GLDM vs PyRadiomics 3.0.1 on the compat phantom.
//
// Recipe gldm3d.pyradiomics_bincount20: bench_compat_liver_3d (label 1) with binCount=20,
// no resampling, weightingNorm=None, imageType=Original; on the Nyxus side GREYDEPTH=100,
// IBSI=false and GLDM_GREYDEPTH=-20, whose negative sign selects the same binCount binning.
// PyRadiomics' distances=[1] and gldm_a=0 defaults are what Nyxus' fixed 26-offset shifts[] table
// and its pi == neig_pi test implement, so neither side has a knob the other lacks.
//
// Goldens and their reproduction: tests/vetting/oracles/gen_gldm3d_pyradiomics.py, which also
// re-verifies every pin below, rebuilds the dependence matrix from the definition, and recomputes
// all fourteen features from the pinned matrix.
// Measurements: tests/vetting/audit/gldm_3d_pyradiomics_vetting_report.md.

// Only what nothing this file already includes supplies: <algorithm> for the matrix assertion's max,
// <iomanip> for the dump helper's setprecision, <map> for its cell tally, and <unordered_set> for its
// level set. <iostream>, <string>, <vector>, <cmath> for std::abs, and gtest arrive through the
// common header and are not repeated. std::pair comes with <map>, whose value_type is one.
#include <algorithm>
#include <iomanip>
#include <map>
#include <unordered_set>

#include "test_3d_gldm_common.h"   // gtest, <iostream>, <cmath>, the phantoms, the settings recipe, extract_3d_gldm
#include "test_ref_vals.h"         // ref_vals_map, ref_vals_list

static const ref_vals_map<double> gldm_3d_pyradiomics_ref_vals
{
	{"3GLDM_DE", 6.60487318745419},            // original_gldm_DependenceEntropy
	{"3GLDM_DN", 620.2816666666666},           // original_gldm_DependenceNonUniformity
	{"3GLDM_DNN", 0.12922534722222223},        // original_gldm_DependenceNonUniformityNormalized
	{"3GLDM_DV", 5.425478993055556},           // original_gldm_DependenceVariance
	{"3GLDM_GLN", 481.78125},                  // original_gldm_GrayLevelNonUniformity
	{"3GLDM_GLV", 8.728494401041667},          // original_gldm_GrayLevelVariance
	{"3GLDM_HGLE", 129.87979166666668},        // original_gldm_HighGrayLevelEmphasis
	{"3GLDM_LDE", 24.279166666666665},         // original_gldm_LargeDependenceEmphasis
	{"3GLDM_LDHGLE", 3061.1764583333334},      // original_gldm_LargeDependenceHighGrayLevelEmphasis
	{"3GLDM_LDLGLE", 0.252649584876794},       // original_gldm_LargeDependenceLowGrayLevelEmphasis
	{"3GLDM_LGLE", 0.012371308742463947},      // original_gldm_LowGrayLevelEmphasis
	{"3GLDM_SDE", 0.1635035514256671},         // original_gldm_SmallDependenceEmphasis
	{"3GLDM_SDHGLE", 21.9586484612667},        // original_gldm_SmallDependenceHighGrayLevelEmphasis
	{"3GLDM_SDLGLE", 0.0024445083605478196}    // original_gldm_SmallDependenceLowGrayLevelEmphasis
};

// SPEC 7 row 1, the exact tier, which is an ABSOLUTE 1e-9 and is asserted with ASSERT_NEAR. The tier
// applies for the reason the SPEC gives it -- no estimator disagreement. Both sides bin this phantom
// into the same twenty levels, walk the same 26 neighbours at the same alpha = 0 cutoff, and start
// the dependence count at the same 1, so nothing separates them but float summation order.
static const double gldm_3d_pyradiomics_abs_tolerance = 1e-9;

// 3GLDM_DE is the family's only sum over logarithms, and Nyxus takes it through fast_log10() -- a
// float-precision log10 approximation divided by a ten-digit LOG10_2 -- where PyRadiomics uses
// numpy.log2. That is a deliberate fast path and its error is a convention of this codebase, so it
// belongs in the band rather than in a defect report.
//
// Measured on this fixture at this recipe: Nyxus 6.6031219389041276 against PyRadiomics
// 6.6048731874541904, an absolute residual of 1.7512e-3. The band is that doubled and rounded up to
// one significant figure, which is the derivation the series settled on -- so it is 2.3x the
// measurement and states which measurement. It is this family's own number: the 2D twin's identical
// calc_DE() lands at 1.3e-3 RELATIVE where this one is 2.65e-4 relative, so carrying 2D GLDM's
// figure across would have been five times wider than anything measured here.
static const double gldm_3d_pyradiomics_de_abs_tolerance = 4e-3;

// One non-empty cell of a dependence matrix: the grey level, the dependence counted, and how many
// voxels of that level carry that dependence.
struct Gldm3dMatrixCell
{
	unsigned int level;
	int dep;
	int count;
};

// Dimensions of the phantom's dependence matrix. 'ng' is the number of grey levels the ROI actually
// holds, which is what Nyxus' sorted unique-intensity list I holds -- not the bin count. 'nd' is the
// largest dependence observed, which is what calculate() trims the matrix to whenever the binning is
// not IBSI. 'nz' is the number of dependence zones and 'np' the ROI's voxel count; PyRadiomics allows
// incomplete zones, so every ROI voxel owns exactly one zone and the two are equal by construction.
static const int gldm_3d_pyradiomics_matrix_ng = 20;
static const int gldm_3d_pyradiomics_matrix_nd = 15;
static const int gldm_3d_pyradiomics_matrix_nz = 4800;
static const int gldm_3d_pyradiomics_matrix_np = 4800;

// Every non-empty cell, keyed by {grey level, dependence}. PyRadiomics deletes the rows of absent
// grey levels and the columns of absent dependences and carries the survivors in ivector / jvector;
// these cells are keyed by the values themselves, so the two representations line up entry for entry
// however either side chooses to index.
static const ref_vals_list<Gldm3dMatrixCell> gldm_3d_pyradiomics_matrix_ref_vals
{
	{ 1, 3, 2 },
	{ 1, 4, 2 },
	{ 1, 5, 1 },
	{ 2, 1, 6 },
	{ 3, 1, 4 },
	{ 3, 2, 6 },
	{ 3, 3, 7 },
	{ 3, 4, 2 },
	{ 3, 5, 2 },
	{ 4, 1, 10 },
	{ 4, 2, 21 },
	{ 4, 3, 12 },
	{ 4, 4, 10 },
	{ 4, 6, 1 },
	{ 5, 1, 14 },
	{ 5, 2, 28 },
	{ 5, 3, 19 },
	{ 5, 4, 9 },
	{ 5, 5, 9 },
	{ 5, 6, 1 },
	{ 6, 1, 15 },
	{ 6, 2, 43 },
	{ 6, 3, 26 },
	{ 6, 4, 22 },
	{ 6, 5, 15 },
	{ 6, 6, 6 },
	{ 6, 7, 8 },
	{ 6, 8, 1 },
	{ 7, 1, 30 },
	{ 7, 2, 49 },
	{ 7, 3, 48 },
	{ 7, 4, 41 },
	{ 7, 5, 17 },
	{ 7, 6, 13 },
	{ 7, 7, 14 },
	{ 7, 8, 5 },
	{ 7, 9, 4 },
	{ 7, 10, 2 },
	{ 8, 1, 35 },
	{ 8, 2, 74 },
	{ 8, 3, 63 },
	{ 8, 4, 63 },
	{ 8, 5, 45 },
	{ 8, 6, 35 },
	{ 8, 7, 8 },
	{ 8, 8, 5 },
	{ 8, 9, 1 },
	{ 8, 10, 1 },
	{ 9, 1, 34 },
	{ 9, 2, 73 },
	{ 9, 3, 73 },
	{ 9, 4, 89 },
	{ 9, 5, 74 },
	{ 9, 6, 74 },
	{ 9, 7, 34 },
	{ 9, 8, 30 },
	{ 9, 9, 16 },
	{ 9, 10, 10 },
	{ 9, 11, 4 },
	{ 9, 12, 3 },
	{ 9, 13, 3 },
	{ 9, 14, 1 },
	{ 10, 1, 26 },
	{ 10, 2, 71 },
	{ 10, 3, 91 },
	{ 10, 4, 116 },
	{ 10, 5, 124 },
	{ 10, 6, 83 },
	{ 10, 7, 70 },
	{ 10, 8, 47 },
	{ 10, 9, 19 },
	{ 10, 10, 10 },
	{ 10, 11, 9 },
	{ 10, 12, 2 },
	{ 10, 13, 2 },
	{ 10, 14, 1 },
	{ 10, 15, 2 },
	{ 11, 1, 31 },
	{ 11, 2, 65 },
	{ 11, 3, 104 },
	{ 11, 4, 125 },
	{ 11, 5, 125 },
	{ 11, 6, 95 },
	{ 11, 7, 86 },
	{ 11, 8, 57 },
	{ 11, 9, 48 },
	{ 11, 10, 33 },
	{ 11, 11, 9 },
	{ 11, 12, 5 },
	{ 11, 13, 4 },
	{ 12, 1, 38 },
	{ 12, 2, 72 },
	{ 12, 3, 83 },
	{ 12, 4, 99 },
	{ 12, 5, 111 },
	{ 12, 6, 82 },
	{ 12, 7, 48 },
	{ 12, 8, 39 },
	{ 12, 9, 17 },
	{ 12, 10, 13 },
	{ 12, 11, 7 },
	{ 12, 12, 8 },
	{ 12, 13, 2 },
	{ 12, 14, 1 },
	{ 13, 1, 40 },
	{ 13, 2, 72 },
	{ 13, 3, 76 },
	{ 13, 4, 84 },
	{ 13, 5, 71 },
	{ 13, 6, 32 },
	{ 13, 7, 33 },
	{ 13, 8, 22 },
	{ 13, 9, 8 },
	{ 13, 10, 4 },
	{ 14, 1, 40 },
	{ 14, 2, 59 },
	{ 14, 3, 83 },
	{ 14, 4, 44 },
	{ 14, 5, 45 },
	{ 14, 6, 35 },
	{ 14, 7, 22 },
	{ 14, 8, 7 },
	{ 14, 9, 8 },
	{ 14, 10, 1 },
	{ 14, 11, 3 },
	{ 15, 1, 25 },
	{ 15, 2, 47 },
	{ 15, 3, 46 },
	{ 15, 4, 45 },
	{ 15, 5, 30 },
	{ 15, 6, 12 },
	{ 15, 7, 8 },
	{ 15, 8, 6 },
	{ 15, 9, 2 },
	{ 16, 1, 16 },
	{ 16, 2, 35 },
	{ 16, 3, 42 },
	{ 16, 4, 33 },
	{ 16, 5, 16 },
	{ 16, 6, 10 },
	{ 16, 8, 2 },
	{ 17, 1, 17 },
	{ 17, 2, 18 },
	{ 17, 3, 19 },
	{ 17, 4, 17 },
	{ 17, 5, 12 },
	{ 17, 6, 1 },
	{ 17, 7, 1 },
	{ 18, 1, 8 },
	{ 18, 2, 12 },
	{ 18, 3, 15 },
	{ 18, 4, 11 },
	{ 18, 5, 7 },
	{ 18, 6, 4 },
	{ 18, 7, 4 },
	{ 18, 8, 5 },
	{ 19, 1, 8 },
	{ 19, 2, 5 },
	{ 19, 3, 5 },
	{ 19, 4, 5 },
	{ 19, 5, 2 },
	{ 20, 1, 3 },
	{ 20, 2, 4 }
};

// A 4x4x3 volume small enough to check by hand, and the direct check on the vertical half of the
// neighbourhood: its two populated slices are identical and the third is empty, so every voxel sees
// its own in-slice matches twice -- once in its own slice, once among the eight diagonal neighbours
// of the identical slice -- plus itself and its vertical partner. Every dependence is therefore
// 2 * (in-slice matches) + 2, and every value below is even. A neighbourhood missing the z offsets
// would halve them; one missing the two strictly-vertical offsets would make them odd.
static const std::vector<PixIntens> gldm_3d_pyradiomics_small_volume
{
	0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,
	1, 2, 3, 4,  1, 3, 4, 4,  3, 2, 2, 2,  4, 1, 4, 1,
	1, 2, 3, 4,  1, 3, 4, 4,  3, 2, 2, 2,  4, 1, 4, 1
};

static const int gldm_3d_pyradiomics_smallmatrix_ng = 4;
static const int gldm_3d_pyradiomics_smallmatrix_nd = 6;
static const int gldm_3d_pyradiomics_smallmatrix_nz = 32;

static const ref_vals_list<Gldm3dMatrixCell> gldm_3d_pyradiomics_smallmatrix_ref_vals
{
	{ 1, 2, 4 },
	{ 1, 4, 4 },
	{ 2, 2, 2 },
	{ 2, 4, 4 },
	{ 2, 6, 2 },
	{ 3, 4, 4 },
	{ 3, 6, 2 },
	{ 4, 2, 4 },
	{ 4, 6, 6 }
};

// The 26 offsets calculate() walks, in {dz, dy, dx}. Spelled out here rather than reached for in the
// feature class, because the definition arm below exists to check that the neighbourhood is the one
// PyRadiomics uses -- borrowing the table under test would make that arm circular.
static const int gldm_3d_neighbourhood_size = 26;

// What calculate() leaves in Nd on the small volume. Unlike the phantom's, that run is at no binning,
// so the 'if (greyInfo) Nd = max_Nd' trim is skipped and Nd stays at the width the matrix was
// allocated to -- one column per neighbour, plus the centre voxel's own count. Pinned separately from
// the 6 above because those two numbers being different is the trim being exercised.
static const int gldm_3d_pyradiomics_smallmatrix_nd_reported = gldm_3d_neighbourhood_size + 1;

static const int gldm_3d_neighbour_shifts[gldm_3d_neighbourhood_size][3] =
{
	{ 0,-1,-1}, { 0,-1, 0}, { 0,-1,+1}, { 0, 0,-1}, { 0, 0,+1}, { 0,+1,-1}, { 0,+1, 0}, { 0,+1,+1},
	{-1,-1,-1}, {-1,-1, 0}, {-1,-1,+1}, {-1, 0,-1}, {-1, 0, 0}, {-1, 0,+1}, {-1,+1,-1}, {-1,+1, 0}, {-1,+1,+1},
	{+1,-1,-1}, {+1,-1, 0}, {+1,-1,+1}, {+1, 0,-1}, {+1, 0, 0}, {+1, 0,+1}, {+1,+1,-1}, {+1,+1, 0}, {+1,+1,+1}
};

// The tally of one dependence matrix: cells keyed by {grey level, dependence}, plus the three
// dimensions the pinned table carries alongside them.
struct Gldm3dMatrixTally
{
	std::map<std::pair<unsigned int, int>, int> cells;
	int ng = 0;        // distinct grey levels that own at least one cell
	int nd = 0;        // largest dependence observed
	int nz = 0;        // dependence zones, one per ROI voxel
};

// The same matrix from the definition alone, walking the shift table spelled out above. It shares no
// code with the feature class, so it fails when production and the pins drift together.
static Gldm3dMatrixTally tally_3d_gldm_matrix_definition (const SimpleCube<PixIntens>& D)
{
	int w = D.width(), h = D.height(), d = D.depth();

	Gldm3dMatrixTally t;
	std::unordered_set<PixIntens> levels;

	for (int z = 0; z < d; z++)
		for (int y = 0; y < h; y++)
			for (int x = 0; x < w; x++)
			{
				PixIntens pi = D.zyx(z, y, x);
				if (pi == 0)          // background at radiomics binning; never a dependence centre
					continue;
				t.nz++;
				levels.insert(pi);

				int dep = 1;          // the voxel counts toward its own dependence
				for (const auto& sh : gldm_3d_neighbour_shifts)
				{
					int az = z + sh[0], ay = y + sh[1], ax = x + sh[2];
					if (!D.safe(az, ay, ax))
						continue;
					if (D.zyx(az, ay, ax) == pi)   // alpha = 0: dependent iff equal
						dep++;
				}
				t.cells[{ pi, dep }]++;
				t.nd = (std::max)(t.nd, dep);
			}

	t.ng = (int)levels.size();
	return t;
}

// Compares one tally against the pinned table, cell for cell and in both directions -- a populated
// cell the table does not carry fails as loudly as a pinned cell the cube does not produce.
static void assert_3d_gldm_tally_matches (
	const Gldm3dMatrixTally& got,
	const ref_vals_list<Gldm3dMatrixCell>& expect,
	int expect_ng,
	int expect_nd,
	int expect_nz)
{
	ASSERT_EQ (got.ng, expect_ng) << "grey levels present in the ROI";
	ASSERT_EQ (got.nd, expect_nd) << "largest dependence observed";
	ASSERT_EQ (got.nz, expect_nz) << "dependence zones, one per ROI voxel";

	ASSERT_EQ ((int)expect.size(), (int)got.cells.size()) << "non-empty cells";
	for (const auto& c : expect)
	{
		SCOPED_TRACE ("level " + std::to_string(c.level) + ", dependence " + std::to_string(c.dep));
		auto it = got.cells.find({ c.level, c.dep });
		ASSERT_TRUE (it != got.cells.end()) << "pinned cell is empty in this run";
		ASSERT_EQ (it->second, c.count);
	}
}

// Holds the pinned table to the dependence matrix a run left behind -- 'P', the grey-level list 'I'
// its rows are indexed through, and the three dimensions the fourteen features are contracted from.
// Reading the feature object rather than rebuilding the table beside it is what puts calculate()'s
// row mapping, its Ng and Nd, its allocation and its fill loop under the assertion: a defect in any
// of those lives in this object and nowhere the definition arm below can see.
//
// 'expect_nd' is the largest dependence carrying a zone, read off P. 'expect_nd_reported' is what
// calculate() leaves in Nd, which is that same number only when the binning is not IBSI -- at
// GLDM_GREYDEPTH=0 the trim is skipped and Nd stays at the full neighbourhood width. The two are
// separate parameters because a run that stopped trimming would otherwise be invisible.
static void assert_3d_gldm_matrix_production (
	const D3_GLDM_feature& f,
	const ref_vals_list<Gldm3dMatrixCell>& expect,
	int expect_ng,
	int expect_nd,
	int expect_nd_reported,
	int expect_nz)
{
	const SimpleMatrix<int>& P = f.get_P();
	const std::vector<PixIntens>& I = f.get_I();

	ASSERT_EQ (f.get_Ng(), expect_ng) << "grey levels the matrix is indexed over";
	ASSERT_EQ (f.get_Nd(), expect_nd_reported) << "dependence count calculate() reports";
	ASSERT_EQ (f.get_Nz(), expect_nz) << "dependence zones, summed off the matrix";

	// The allocation, so a mis-sized table is not read as a matrix whose missing cells are merely
	// empty. calculate() allocates at the full neighbourhood width, before it trims Nd, and one row
	// per grey level plus one -- the +1s are what a zone at the largest dependence needs.
	ASSERT_EQ (P.width(), gldm_3d_neighbourhood_size + 2) << "allocated dependence columns";
	ASSERT_EQ (P.height(), expect_ng + 1) << "allocated grey-level rows";

	int pinned_zones = 0;
	for (const Gldm3dMatrixCell& c : expect)
	{
		SCOPED_TRACE ("level " + std::to_string(c.level) + ", dependence " + std::to_string(c.dep));
		auto it = std::find (I.begin(), I.end(), (PixIntens)c.level);
		ASSERT_TRUE (it != I.end()) << "pinned grey level is absent from the run's level list";
		ASSERT_EQ (P.yx (int(it - I.begin()), c.dep - 1), c.count);
		pinned_zones += c.count;
	}

	// Nothing outside the pinned cells, and the deepest occupied column read off P rather than off
	// the pins: without these a matrix carrying an extra populated cell, or one dependence deeper
	// than anything pinned, would still pass every line above.
	int nonempty = 0, total = 0, deepest = 0;
	for (int row = 0; row < P.height(); row++)
		for (int col = 0; col < P.width(); col++)
		{
			int v = P.yx (row, col);
			if (!v)
				continue;
			nonempty++;
			total += v;
			deepest = (std::max)(deepest, col + 1);
		}
	ASSERT_EQ (nonempty, (int)expect.size()) << "non-empty cells";
	ASSERT_EQ (total, expect_nz) << "zones the matrix accounts for";
	ASSERT_EQ (pinned_zones, expect_nz) << "zones the pinned cells account for";
	ASSERT_EQ (deepest, expect_nd) << "largest dependence carrying a zone";
}

// Holds the same pinned table to the matrix the definition produces, walking the shift table spelled
// out above and sharing no code with the feature class. This is what keeps the pins from being
// whatever production happens to say; the assertion above is what says production produced them.
//
// EXPECT rather than ASSERT between the two arms at the call sites, so a failing arm does not hide
// the other's verdict: the pair of them is what says whether production moved, the pins moved, or
// both moved together.
static void assert_3d_gldm_matrix_definition (
	const SimpleCube<PixIntens>& D,
	const ref_vals_list<Gldm3dMatrixCell>& expect,
	int expect_ng,
	int expect_nd,
	int expect_nz)
{
	assert_3d_gldm_tally_matches (tally_3d_gldm_matrix_definition (D), expect, expect_ng, expect_nd, expect_nz);
}

// Asserts one feature against its PyRadiomics golden at the tier SPEC 7 gives it.
static void assert_3d_gldm_feature_pyradiomics (const Nyxus::Feature3D& expecting_fcode, const std::string& fname)
{
	auto iter = gldm_3d_pyradiomics_ref_vals.find (fname);
	ASSERT_TRUE (iter != gldm_3d_pyradiomics_ref_vals.end()) << fname;

	auto [ipath, mpath, label] = get_3d_compat_phantom();
	Fsettings s = make_gldm3d_settings (100/*greydepth*/, -20/*radiomics binCount binning*/);

	std::vector<std::vector<double>> fvals;
	SimpleCube<PixIntens> cube;
	PixIntens lo = 0, hi = 0;
	ASSERT_NO_FATAL_FAILURE(extract_3d_gldm (fvals, cube, lo, hi, ipath, mpath, label, s));

	int fcode = -1;
	ASSERT_NO_FATAL_FAILURE(resolve_3d_gldm_fcode (fcode, expecting_fcode, fname));

	double band = fname == "3GLDM_DE"
		? gldm_3d_pyradiomics_de_abs_tolerance
		: gldm_3d_pyradiomics_abs_tolerance;
	ASSERT_NEAR (fvals[fcode][0], iter->second, band) << fname;
}

// The matrix under the fourteen features, read off the feature object the fourteen oracle assertions
// run through: this assertion and those describe one run of one phantom rather than two, and a
// hand-written copy of the phantom would keep passing after the loader started producing something
// else. The definition arm beside it re-bins the cube that run read, with the same
// bin_intensities_3d() calculate() calls, and walks the neighbourhood from the definition.
void test_3d_gldm_matrix_pyradiomics()
{
	auto [ipath, mpath, label] = get_3d_compat_phantom();
	Fsettings s = make_gldm3d_settings (100/*greydepth*/, -20/*radiomics binCount binning*/);
	std::vector<std::vector<double>> fvals;
	SimpleCube<PixIntens> cube;
	PixIntens lo = 0, hi = 0;
	D3_GLDM_feature f;
	ASSERT_NO_FATAL_FAILURE(extract_3d_gldm (fvals, cube, lo, hi, ipath, mpath, label, s, f));

	{
		SCOPED_TRACE ("the dependence matrix D3_GLDM_feature::calculate() built");
		EXPECT_NO_FATAL_FAILURE (assert_3d_gldm_matrix_production (
			f, gldm_3d_pyradiomics_matrix_ref_vals,
			gldm_3d_pyradiomics_matrix_ng, gldm_3d_pyradiomics_matrix_nd,
			gldm_3d_pyradiomics_matrix_nd/*trimmed: the binning is not IBSI*/,
			gldm_3d_pyradiomics_matrix_nz));
	}

	SimpleCube<PixIntens> binned;
	binned.allocate (cube.width(), cube.height(), cube.depth());
	f.bin_intensities_3d (binned, cube, lo, hi, -20);

	{
		SCOPED_TRACE ("the matrix the definition builds, sharing no code with the feature class");
		EXPECT_NO_FATAL_FAILURE (assert_3d_gldm_matrix_definition (
			binned, gldm_3d_pyradiomics_matrix_ref_vals,
			gldm_3d_pyradiomics_matrix_ng, gldm_3d_pyradiomics_matrix_nd,
			gldm_3d_pyradiomics_matrix_nz));
	}

	// Nz == Np: every ROI voxel owns exactly one dependence zone, so the zone count the matrix
	// carries has to be the voxel count, reached here by counting the cube instead of the matrix.
	int occupied = 0;
	for (auto v : binned)
		if (v)
			occupied++;
	ASSERT_EQ (occupied, gldm_3d_pyradiomics_matrix_np);
}

// The neighbourhood check: a volume small enough to count by hand, where the vertical offsets have a
// visible signature. It runs the same calculate() the phantom assertion does, at the family's
// no-binning setting, so the matrix is indexed by the literal's own levels 1..4 -- which is what
// PyRadiomics reads at binWidth=1, and which leaves Nd untrimmed at the full neighbourhood width.
void test_3d_gldm_smallmatrix_pyradiomics()
{
	Fsettings s = make_gldm3d_settings (64/*greydepth, inert here*/, 0/*no binning: the raw levels*/);
	std::vector<std::vector<double>> fvals;
	D3_GLDM_feature f;
	ASSERT_NO_FATAL_FAILURE(run_3d_gldm_on_volume (
		fvals, gldm_3d_pyradiomics_small_volume, 4/*width*/, 4/*height*/, 3/*depth*/, s, f));

	{
		SCOPED_TRACE ("the dependence matrix D3_GLDM_feature::calculate() built");
		EXPECT_NO_FATAL_FAILURE (assert_3d_gldm_matrix_production (
			f, gldm_3d_pyradiomics_smallmatrix_ref_vals,
			gldm_3d_pyradiomics_smallmatrix_ng, gldm_3d_pyradiomics_smallmatrix_nd,
			gldm_3d_pyradiomics_smallmatrix_nd_reported,
			gldm_3d_pyradiomics_smallmatrix_nz));
	}

	{
		SCOPED_TRACE ("the matrix the definition builds, sharing no code with the feature class");
		SimpleCube<PixIntens> D (gldm_3d_pyradiomics_small_volume, 4/*width*/, 4/*height*/, 3/*depth*/);
		EXPECT_NO_FATAL_FAILURE (assert_3d_gldm_matrix_definition (
			D, gldm_3d_pyradiomics_smallmatrix_ref_vals,
			gldm_3d_pyradiomics_smallmatrix_ng, gldm_3d_pyradiomics_smallmatrix_nd,
			gldm_3d_pyradiomics_smallmatrix_nz));
	}
}

// Regenerates every golden in gldm_3d_pyradiomics_ref_vals at full precision, in the shape the table
// wants. Run it with
//     runAllTests --gtest_filter=*3D_GLDM_DUMP_PYRADIOMICS*
// It prints Nyxus' own values at the recipe the assertions use, which is what makes the residual
// against the pinned PyRadiomics goldens readable without a debugger.
void test_3d_gldm_dump_pyradiomics()
{
	auto [ipath, mpath, label] = get_3d_compat_phantom();
	Fsettings s = make_gldm3d_settings (100/*greydepth*/, -20/*radiomics binCount binning*/);
	std::vector<std::vector<double>> fvals;
	SimpleCube<PixIntens> cube;
	PixIntens lo = 0, hi = 0;
	ASSERT_NO_FATAL_FAILURE(extract_3d_gldm (fvals, cube, lo, hi, ipath, mpath, label, s));

	Environment e;
	std::cout << "[3DGLDM-PYRAD] cube " << cube.width() << "x" << cube.height() << "x" << cube.depth()
	          << ", intensity range [" << lo << ", " << hi << "]\n";
	for (const auto& nv : gldm_3d_pyradiomics_ref_vals)
	{
		int fcode = -1;
		ASSERT_TRUE(e.theFeatureSet.find_3D_FeatureByString(nv.first, fcode));
		double got = fvals[fcode][0];
		std::cout << "[3DGLDM-PYRAD]\t{\"" << nv.first << "\", "
		          << std::setprecision(17) << got << "},\tpinned "
		          << std::setprecision(17) << nv.second
		          << "\tabs " << std::setprecision(3) << std::abs(got - nv.second)
		          << "\trel " << std::setprecision(3) << std::abs(got - nv.second) / std::abs(nv.second)
		          << "\n";
	}
}

void test_3d_gldm_de_pyradiomics() { assert_3d_gldm_feature_pyradiomics(Nyxus::Feature3D::GLDM_DE, "3GLDM_DE"); }
void test_3d_gldm_dn_pyradiomics() { assert_3d_gldm_feature_pyradiomics(Nyxus::Feature3D::GLDM_DN, "3GLDM_DN"); }
void test_3d_gldm_dnn_pyradiomics() { assert_3d_gldm_feature_pyradiomics(Nyxus::Feature3D::GLDM_DNN, "3GLDM_DNN"); }
void test_3d_gldm_dv_pyradiomics() { assert_3d_gldm_feature_pyradiomics(Nyxus::Feature3D::GLDM_DV, "3GLDM_DV"); }
void test_3d_gldm_gln_pyradiomics() { assert_3d_gldm_feature_pyradiomics(Nyxus::Feature3D::GLDM_GLN, "3GLDM_GLN"); }
void test_3d_gldm_glv_pyradiomics() { assert_3d_gldm_feature_pyradiomics(Nyxus::Feature3D::GLDM_GLV, "3GLDM_GLV"); }
void test_3d_gldm_hgle_pyradiomics() { assert_3d_gldm_feature_pyradiomics(Nyxus::Feature3D::GLDM_HGLE, "3GLDM_HGLE"); }
void test_3d_gldm_lde_pyradiomics() { assert_3d_gldm_feature_pyradiomics(Nyxus::Feature3D::GLDM_LDE, "3GLDM_LDE"); }
void test_3d_gldm_ldhgle_pyradiomics() { assert_3d_gldm_feature_pyradiomics(Nyxus::Feature3D::GLDM_LDHGLE, "3GLDM_LDHGLE"); }
void test_3d_gldm_ldlgle_pyradiomics() { assert_3d_gldm_feature_pyradiomics(Nyxus::Feature3D::GLDM_LDLGLE, "3GLDM_LDLGLE"); }
void test_3d_gldm_lgle_pyradiomics() { assert_3d_gldm_feature_pyradiomics(Nyxus::Feature3D::GLDM_LGLE, "3GLDM_LGLE"); }
void test_3d_gldm_sde_pyradiomics() { assert_3d_gldm_feature_pyradiomics(Nyxus::Feature3D::GLDM_SDE, "3GLDM_SDE"); }
void test_3d_gldm_sdhgle_pyradiomics() { assert_3d_gldm_feature_pyradiomics(Nyxus::Feature3D::GLDM_SDHGLE, "3GLDM_SDHGLE"); }
void test_3d_gldm_sdlgle_pyradiomics() { assert_3d_gldm_feature_pyradiomics(Nyxus::Feature3D::GLDM_SDLGLE, "3GLDM_SDLGLE"); }
