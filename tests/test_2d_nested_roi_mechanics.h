#pragma once

#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include "../src/nyx/globals.h"
#include "../src/nyx/nested_roi.h"

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
