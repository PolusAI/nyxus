#pragma once

#include <gtest/gtest.h>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <vector>
#include "../src/nyx/environment.h"
#include "../src/nyx/globals.h"
#include "../src/nyx/nested_roi.h"
#include "../src/nyx/helpers/fsystem.h"
#include "../src/nyx/raw_tiff.h"          // libtiff, for writing the fixtures

// The nested-ROI table (aggregate_features2) reads each parent's and child's record back from
// the run's feature CSV and writes one row per parent: the parent's non-feature columns, its
// features, then its children's features. These tests write that feature CSV the way the CSV
// sink does -- quoted header from get_header(), unquoted values -- and read the nested table.

namespace nested_roi_test
{
	namespace fs = std::filesystem;

	inline std::vector<std::string> split_csv (const std::string& line)
	{
		std::vector<std::string> cells;
		std::istringstream ss (line);
		Nyxus::parse_csv_line (cells, ss);
		return cells;
	}

	inline std::string unquote (const std::string& s)
	{
		return (s.size() >= 2 && s.front() == '"' && s.back() == '"') ? s.substr (1, s.size() - 2) : s;
	}

	/// @brief An environment whose single-CSV output lands in a fresh directory, with MEAN and MAX enabled.
	struct NestedFixture
	{
		Environment env;
		fs::path dir;
		Nyxus::NestableRois P, C;

		explicit NestedFixture (const std::string& tag)
		{
			dir = fs::temp_directory_path() / ("nyxus_nested_" + tag);
			fs::remove_all (dir);
			fs::create_directories (dir);
			env.output_dir = dir.string();
			env.separateCsv = false;
			env.nyxus_result_fname = "features";
			env.theFeatureSet.enableAll (false);
			env.theFeatureSet.enableFeatures ({ Nyxus::Feature2D::MEAN, Nyxus::Feature2D::MAX });
			env.dataset.dataset_props.push_back (SlideProps ("int_c1.tif", "seg_c1.tif"));

			// parent 1 holds child 2; both come from slide 0
			NestedLR par, chi;
			par.label = 1;
			par.slide_idx = 0;
			par.children = { 2 };
			chi.label = 2;
			chi.slide_idx = 0;
			P[1] = par;
			C[2] = chi;
		}

		~NestedFixture() { std::error_code ec; fs::remove_all (dir, ec); }

		/// @brief Writes the run's feature CSV: 'header' quoted, then 'rows' as they are.
		void write_features (const std::vector<std::string>& header, const std::vector<std::string>& rows)
		{
			std::ofstream f (Nyxus::get_feature_output_fname (env, "int_c1.tif", "seg_c1.tif"));
			for (size_t i = 0; i < header.size(); i++)
				f << (i ? "," : "") << '"' << header[i] << '"';
			f << "\n";
			for (const auto& r : rows)
				f << r << "\n";
		}

		/// @brief Runs the nested aggregation and returns the nested table's lines.
		std::vector<std::string> nested_table (NestedRoiOptions::Aggregations aggr)
		{
			EXPECT_TRUE (Nyxus::aggregate_features2 (env, env.theFeatureSet, P, C, dir.string(), "par", aggr, 0));
			std::ifstream f (dir / "par_nested_features.csv");
			std::vector<std::string> lines;
			for (std::string line; std::getline (f, line); )
				lines.push_back (line);
			return lines;
		}
	};

	/// @brief The cell of 'row' under column 'name' of 'header', or "<absent>".
	inline std::string cell (const std::vector<std::string>& header, const std::vector<std::string>& row, const std::string& name)
	{
		for (size_t i = 0; i < header.size(); i++)
			if (unquote (header[i]) == name)
				return i < row.size() ? row[i] : "<short row>";
		return "<absent>";
	}
}

// The parent's non-feature columns -- unit, channel and spacing among them -- are carried into
// the nested table under their own names, and the features stay under theirs.
void test_2d_nested_roi_table_carries_leading_columns_mechanics()
{
	using namespace nested_roi_test;
	NestedFixture fx ("leading");

	const std::vector<std::string> head = Nyxus::get_header (fx.env);
	ASSERT_EQ (Nyxus::leading_columns (head), (std::vector<std::string>{ "intensity_image", "mask_image", "phys_unit",
		"ROI_label", "t_index", "c_index", "phys_x", "phys_y", "phys_z" }));
	ASSERT_EQ (head.size(), 11u);	// + MAX, MEAN (feature-code order)

	// distinct values in every column, so a shifted cell cannot pass for the right one
	fx.write_features (head, {
		"int_c1.tif,seg_c1.tif,um,1,3,2,0.5,0.25,2,11,10",
		"int_c1.tif,seg_c1.tif,um,2,3,2,0.5,0.25,2,21,20" });

	for (auto aggr : { NestedRoiOptions::Aggregations::aNONE, NestedRoiOptions::Aggregations::aSUM })
	{
		SCOPED_TRACE (aggr);
		auto lines = fx.nested_table (aggr);
		ASSERT_EQ (lines.size(), 2u);
		auto h = split_csv (lines[0]), row = split_csv (lines[1]);
		ASSERT_EQ (h.size(), row.size());

		EXPECT_EQ (cell (h, row, "intensity_image"), "int_c1.tif");
		EXPECT_EQ (cell (h, row, "mask_image"), "seg_c1.tif");
		EXPECT_EQ (cell (h, row, "phys_unit"), "um");
		EXPECT_EQ (cell (h, row, "ROI_label"), "1");
		EXPECT_EQ (cell (h, row, "t_index"), "3");
		EXPECT_EQ (cell (h, row, "c_index"), "2");
		EXPECT_EQ (cell (h, row, "phys_x"), "0.5");
		EXPECT_EQ (cell (h, row, "phys_y"), "0.25");
		EXPECT_EQ (cell (h, row, "phys_z"), "2");
		EXPECT_EQ (cell (h, row, "MEAN"), "10");
		EXPECT_EQ (cell (h, row, "MAX"), "11");
		if (aggr == NestedRoiOptions::Aggregations::aNONE)
		{
			EXPECT_EQ (cell (h, row, "child1_MEAN"), "20");
			EXPECT_EQ (cell (h, row, "child1_MAX"), "21");
		}
		else
		{
			EXPECT_EQ (cell (h, row, "aggr_MEAN"), "20");
			EXPECT_EQ (cell (h, row, "aggr_MAX"), "21");
		}
	}
}

// A parent missing from the feature CSV gets a row as wide as the header: its non-feature cells
// blank, its own and its children's features zero-filled.
void test_2d_nested_roi_table_missing_parent_fills_whole_row_mechanics()
{
	using namespace nested_roi_test;
	NestedFixture fx ("missing");

	const std::vector<std::string> head = Nyxus::get_header (fx.env);
	fx.write_features (head, { "int_c1.tif,seg_c1.tif,um,2,0,0,1,1,1,20,21" });	// the child only

	auto lines = fx.nested_table (NestedRoiOptions::Aggregations::aNONE);
	ASSERT_EQ (lines.size(), 2u);
	auto h = split_csv (lines[0]), row = split_csv (lines[1]);
	const size_t n_leading = Nyxus::leading_columns (head).size();
	ASSERT_EQ (row.size(), h.size());
	for (size_t i = 0; i < n_leading; i++)
		EXPECT_EQ (row[i], "") << unquote (h[i]);
	for (size_t i = n_leading; i < row.size(); i++)
		EXPECT_EQ (row[i], "0.0") << unquote (h[i]);
}

// A feature CSV lacking one of the non-feature columns leaves that cell blank; the others stay
// under their own names rather than shifting into its place.
void test_2d_nested_roi_table_column_absent_from_source_is_blank_mechanics()
{
	using namespace nested_roi_test;
	NestedFixture fx ("absent");

	std::vector<std::string> head = Nyxus::get_header (fx.env);
	head.erase (std::find (head.begin(), head.end(), std::string (Nyxus::colname_c_index)));
	fx.write_features (head, {
		"int_c1.tif,seg_c1.tif,um,1,3,0.5,0.25,2,11,10",
		"int_c1.tif,seg_c1.tif,um,2,3,0.5,0.25,2,21,20" });

	auto lines = fx.nested_table (NestedRoiOptions::Aggregations::aNONE);
	ASSERT_EQ (lines.size(), 2u);
	auto h = split_csv (lines[0]), row = split_csv (lines[1]);
	ASSERT_EQ (row.size(), h.size());
	EXPECT_EQ (cell (h, row, "c_index"), "");
	EXPECT_EQ (cell (h, row, "t_index"), "3");
	EXPECT_EQ (cell (h, row, "phys_x"), "0.5");
	EXPECT_EQ (cell (h, row, "MEAN"), "10");
	EXPECT_EQ (cell (h, row, "child1_MAX"), "21");
}

// A header without phys_z has no recognisable non-feature prefix.
void test_2d_nested_roi_leading_columns_need_phys_z_mechanics()
{
	EXPECT_TRUE (Nyxus::leading_columns ({ "\"intensity_image\"", "\"mask_image\"", "\"ROI_label\"", "\"MEAN\"" }).empty());
	EXPECT_TRUE (Nyxus::leading_columns ({}).empty());
	EXPECT_EQ (Nyxus::leading_columns ({ "\"ROI_label\"", "phys_z", "MEAN" }), (std::vector<std::string>{ "ROI_label", "phys_z" }));
}

// Nested ROI through the CLI flow on plain (non-OME) TIFF: featurize a parent and a child mask
// channel to separate CSVs, then have mine_segment_relations2 pair each parent with its children
// and write the aligned nested feature table from those CSVs.

// A W x H uint16 strip TIFF whose pixel (x,y) is value(x,y).
template <class F>
static void write_2d_nested_roi_tiff (const fs::path& path, uint32_t W, uint32_t H, F value)
{
    TIFF* t = TIFFOpen(path.string().c_str(), "w");
    ASSERT_NE(t, nullptr) << path.string();
    TIFFSetField(t, TIFFTAG_IMAGEWIDTH, W);
    TIFFSetField(t, TIFFTAG_IMAGELENGTH, H);
    TIFFSetField(t, TIFFTAG_SAMPLESPERPIXEL, 1);
    TIFFSetField(t, TIFFTAG_BITSPERSAMPLE, 16);
    TIFFSetField(t, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_UINT);
    TIFFSetField(t, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
    TIFFSetField(t, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
    TIFFSetField(t, TIFFTAG_ROWSPERSTRIP, 1u);
    std::vector<uint16_t> row(W);
    for (uint32_t y = 0; y < H; ++y)
    {
        for (uint32_t x = 0; x < W; ++x)
            row[x] = (uint16_t) value(x, y);
        ASSERT_EQ(TIFFWriteScanline(t, row.data(), y, 0), 1) << path.string() << " row " << y;
    }
    TIFFClose(t);
}

static std::vector<std::string> split_2d_nested_roi_csv (const std::string& line)
{
    std::vector<std::string> cells;
    std::stringstream ss(line);
    std::string cell;
    while (std::getline(ss, cell, ','))
        cells.push_back(cell);
    return cells;
}

// Channel 1 masks one parent (label 1, the square 2..29); channel 2 masks two children inside
// it (label 5 at 4..9, label 7 at 15..20). Each channel's intensity is constant over each ROI,
// so the parent's MEAN is 10 and the children's are 50 and 70. What this discriminates: a
// lookup that takes the ROI label from a fixed column finds no record in a feature CSV whose
// leading columns moved (phys_unit, c_index, the spacing columns), and the nested table is
// zero-filled; one that skips a fixed number of columns reads non-feature cells as features.
void test_2d_nested_roi_csv_mechanics()
{
    const uint32_t W = 32, H = 32;
    fs::path root = fs::temp_directory_path() / "nyxus_2d_nested_roi",
        intDir = root / "int", segDir = root / "seg", outDir = root / "out";
    std::error_code ec;
    fs::remove_all(root, ec);
    fs::create_directories(intDir);
    fs::create_directories(segDir);
    fs::create_directories(outDir);

    auto inside = [](uint32_t x, uint32_t y, uint32_t a, uint32_t b) { return x >= a && x <= b && y >= a && y <= b; };
    ASSERT_NO_FATAL_FAILURE(write_2d_nested_roi_tiff(intDir / "a_c1.tif", W, H, [](uint32_t, uint32_t) { return 10; }));
    ASSERT_NO_FATAL_FAILURE(write_2d_nested_roi_tiff(segDir / "a_c1.tif", W, H, [&](uint32_t x, uint32_t y) { return inside(x, y, 2, 29) ? 1 : 0; }));
    ASSERT_NO_FATAL_FAILURE(write_2d_nested_roi_tiff(intDir / "a_c2.tif", W, H, [&](uint32_t x, uint32_t y) { return inside(x, y, 4, 9) ? 50 : inside(x, y, 15, 20) ? 70 : 1; }));
    ASSERT_NO_FATAL_FAILURE(write_2d_nested_roi_tiff(segDir / "a_c2.tif", W, H, [&](uint32_t x, uint32_t y) { return inside(x, y, 4, 9) ? 5 : inside(x, y, 15, 20) ? 7 : 0; }));

    std::vector<std::string> args = { "nyxus",
        "--intDir=" + intDir.generic_string() + "/", "--segDir=" + segDir.generic_string() + "/",
        "--outDir=" + outDir.generic_string() + "/", "--features=MEAN", "--filePattern=.*\\.tif",
        "--outputType=separatecsv", "--hsig=_c", "--hpar=1", "--hchi=2", "--hag=NONE" };
    std::vector<char*> argv;
    for (auto& a : args)
        argv.push_back(a.data());

    Nyxus::nestedRoiData.clear();
    Environment env;
    ASSERT_TRUE(env.parse_cmdline((int) argv.size(), argv.data()));
    ASSERT_TRUE(env.theFeatureMgr.compile());
    env.theFeatureMgr.apply_user_selection(env.theFeatureSet);
    ASSERT_TRUE(env.theFeatureMgr.init_feature_classes());
    env.compile_feature_settings();

    std::vector<std::string> intensFiles, labelFiles;
    auto err = Nyxus::read_2D_dataset(env.intensity_dir, env.labels_dir, env.get_file_pattern(), env.output_dir,
        env.intSegMapDir, env.intSegMapFile, true, intensFiles, labelFiles);
    ASSERT_FALSE(err.has_value()) << *err;
    ASSERT_EQ(Nyxus::processDataset_2D_segmented(env, intensFiles, labelFiles, env.n_reduce_threads, env.saveOption, env.output_dir), 0);
    ASSERT_TRUE(Nyxus::mine_segment_relations2(env, labelFiles));

    std::ifstream f(outDir / "a_c1_nested_features.csv");
    ASSERT_TRUE(f.good());
    std::string headerLine, rowLine;
    ASSERT_TRUE((bool) std::getline(f, headerLine));
    ASSERT_TRUE((bool) std::getline(f, rowLine));
    std::vector<std::string> header = split_2d_nested_roi_csv(headerLine),
        row = split_2d_nested_roi_csv(rowLine);

    auto column = [&header](const std::string& name) -> size_t
    {
        auto it = std::find(header.begin(), header.end(), name);
        return it == header.end() ? header.size() : (size_t) (it - header.begin());
    };
    const size_t cLabel = column(Nyxus::colname_roi_label), cMean = column("MEAN"),
        cChild1 = column("child1_MEAN"), cChild2 = column("child2_MEAN");
    ASSERT_LT(cChild2, header.size()) << headerLine;
    ASSERT_EQ(row.size(), header.size()) << rowLine;

    EXPECT_EQ(row[cLabel], "1") << rowLine;
    EXPECT_DOUBLE_EQ(std::stod(row[cMean]), 10.0) << rowLine;
    std::set<double> childMeans = { std::stod(row[cChild1]), std::stod(row[cChild2]) };
    EXPECT_EQ(childMeans, (std::set<double>{ 50.0, 70.0 })) << rowLine;

    f.close();
    Nyxus::nestedRoiData.clear();
    fs::remove_all(root, ec);
}
