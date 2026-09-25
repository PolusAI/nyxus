#pragma once

// Nested ROI through the CLI flow on plain (non-OME) TIFF: featurize a parent and a child mask
// channel to separate CSVs, then have mine_segment_relations2 pair each parent with its children
// and write the aligned nested feature table from those CSVs.

#include <gtest/gtest.h>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <vector>
#include "../src/nyx/environment.h"
#include "../src/nyx/globals.h"
#include "../src/nyx/helpers/fsystem.h"
#include "../src/nyx/raw_tiff.h"          // libtiff, for writing the fixtures

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
