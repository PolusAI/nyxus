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
// feature class, because one of the two matrices below exists to check that the neighbourhood is the
// one PyRadiomics uses -- borrowing the table under test would make that arm circular.
static const int gldm_3d_neighbour_shifts[26][3] =
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

// The matrix production builds: D3_GLDM_feature::gather_dependence_zones() is the same traversal
// calculate() fills P from, so its shifts[] table, its alpha = 0 test and its background rule are
// all under this tally rather than beside it.
static Gldm3dMatrixTally tally_3d_gldm_matrix_production (const SimpleCube<PixIntens>& D)
{
	std::vector<std::pair<PixIntens, int>> Z;
	D3_GLDM_feature::gather_dependence_zones (Z, D, 0/*background at radiomics binning*/);

	Gldm3dMatrixTally t;
	std::unordered_set<PixIntens> levels;
	for (const auto& z : Z)
	{
		t.cells[{ z.first, z.second }]++;
		levels.insert (z.first);
		t.nd = (std::max)(t.nd, z.second);
		t.nz++;
	}
	t.ng = (int)levels.size();
	return t;
}

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

// Holds the pinned table to both matrices of a binned cube: the one production's own traversal
// produces, and the one the definition produces independently of it. Production carries the claim --
// a change to shifts[], to the alpha cutoff or to the background rule fails the first arm -- and the
// definition arm keeps the pins from being whatever production happens to say.
//
// EXPECT rather than ASSERT between the two, so a failing arm does not hide the other's verdict: the
// pair of them is what says whether production moved, the pins moved, or both moved together.
static void assert_3d_gldm_matrix_pyradiomics (
	const SimpleCube<PixIntens>& D,
	const ref_vals_list<Gldm3dMatrixCell>& expect,
	int expect_ng,
	int expect_nd,
	int expect_nz)
{
	{
		SCOPED_TRACE ("matrix built by D3_GLDM_feature::gather_dependence_zones()");
		EXPECT_NO_FATAL_FAILURE (assert_3d_gldm_tally_matches (
			tally_3d_gldm_matrix_production (D), expect, expect_ng, expect_nd, expect_nz));
	}

	{
		SCOPED_TRACE ("matrix built from the definition, sharing no code with the feature class");
		EXPECT_NO_FATAL_FAILURE (assert_3d_gldm_tally_matches (
			tally_3d_gldm_matrix_definition (D), expect, expect_ng, expect_nd, expect_nz));
	}
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

// The matrix under the fourteen features, on the voxels the featurisation actually read and binned by
// the same bin_intensities_3d() calculate() calls. It takes the cube and the intensity extrema back
// out of the same extract_3d_gldm() the fourteen oracle assertions run, so this assertion and those
// describe one run rather than two -- a hand-written copy of the phantom would keep passing after the
// loader started producing something else.
void test_3d_gldm_matrix_pyradiomics()
{
	auto [ipath, mpath, label] = get_3d_compat_phantom();
	Fsettings s = make_gldm3d_settings (100/*greydepth*/, -20/*radiomics binCount binning*/);
	std::vector<std::vector<double>> fvals;
	SimpleCube<PixIntens> cube;
	PixIntens lo = 0, hi = 0;
	ASSERT_NO_FATAL_FAILURE(extract_3d_gldm (fvals, cube, lo, hi, ipath, mpath, label, s));

	D3_GLDM_feature f;
	SimpleCube<PixIntens> binned;
	binned.allocate (cube.width(), cube.height(), cube.depth());
	f.bin_intensities_3d (binned, cube, lo, hi, -20);

	ASSERT_NO_FATAL_FAILURE(assert_3d_gldm_matrix_pyradiomics (
		binned, gldm_3d_pyradiomics_matrix_ref_vals,
		gldm_3d_pyradiomics_matrix_ng, gldm_3d_pyradiomics_matrix_nd, gldm_3d_pyradiomics_matrix_nz));

	// Nz == Np: every ROI voxel owns exactly one dependence zone, so the zone count the matrix
	// carries has to be the voxel count, reached here by counting the cube instead of the matrix.
	int occupied = 0;
	for (auto v : binned)
		if (v)
			occupied++;
	ASSERT_EQ (occupied, gldm_3d_pyradiomics_matrix_np);
}

// The neighbourhood check: a volume small enough to count by hand, where the vertical offsets have a
// visible signature.
void test_3d_gldm_smallmatrix_pyradiomics()
{
	SimpleCube<PixIntens> D (gldm_3d_pyradiomics_small_volume, 4/*width*/, 4/*height*/, 3/*depth*/);
	ASSERT_NO_FATAL_FAILURE(assert_3d_gldm_matrix_pyradiomics (
		D, gldm_3d_pyradiomics_smallmatrix_ref_vals,
		gldm_3d_pyradiomics_smallmatrix_ng, gldm_3d_pyradiomics_smallmatrix_nd,
		gldm_3d_pyradiomics_smallmatrix_nz));
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
