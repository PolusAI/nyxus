#pragma once

// Guards and structural invariants of the readers and accumulators shared by both dimensions:
// a guard only fires on input the value fixtures never produce, and a structural property (what
// a container holds one entry per) is invisible to a test that only compares numbers.

#include <gtest/gtest.h>
#include <cstdio>
#include "../src/nyx/environment.h"
#include "../src/nyx/globals.h"
#include "../src/nyx/features/histogram.h"
#include "../src/nyx/features/3d_ooc_volume.h"
#include "../src/nyx/features/voxel_cloud_nontriv.h"
#include "../src/nyx/tiff_sample.h"
#include "../src/nyx/helpers/fsystem.h"

// TrivialHistogram::initialize_uniques describes the data of THAT call. What this discriminates:
// an implementation that accumulates instead of resetting reports statistics over the union of
// every set it has been given -- which is what chords does, summarizing its max chords and then
// all chords through one instance, so the second answer silently describes both.
void test_histogram_uniques_reset_mechanics()
{
    TrivialHistogram h;

    // a first set whose median and mode are unambiguous
    const std::vector<HistoItem> first { 10, 10, 10, 90, 90 };
    h.initialize_uniques (first);
    auto [med1, mode1, p11, p101, p251, p751, p901, p991, iqr1, rmad1, ent1, unif1] = h.get_stats();
    EXPECT_DOUBLE_EQ(med1, 10.0);
    EXPECT_EQ(mode1, (HistoItem) 10);

    // a second set, disjoint from the first and with its own clear median and mode
    const std::vector<HistoItem> second { 1000, 1000, 1000, 1000, 2000 };
    h.initialize_uniques (second);
    auto [med2, mode2, p12, p102, p252, p752, p902, p992, iqr2, rmad2, ent2, unif2] = h.get_stats();
    EXPECT_DOUBLE_EQ(med2, 1000.0) << "the second call describes the second set alone";
    EXPECT_EQ(mode2, (HistoItem) 1000);

    // an accumulating instance holds 10 voxels spanning both sets: its median falls between the
    // two groups and its mode is whichever value happens to win across the union
    EXPECT_NE(med2, 545.0) << "the two sets were merged";
}

// The frequency map holds one entry per distinct grey level, not one per item: that is what lets
// an out-of-core ROI of a billion voxels be summarized in bounded memory. What this discriminates:
// a per-item container holds the whole cloud, which the value-parity tests cannot see because the
// numbers it reports are identical either way.
//
// initialize<Src>() is the one the out-of-core passes take -- it is the template every intensity
// source goes through, in-RAM vector and disk-backed cloud alike -- so that is the entry point
// asserted here. initialize_uniques(), the chords path, is asserted beside it because it keeps its
// own copy of the same loop.
void test_histogram_is_bounded_by_levels_mechanics()
{
    const int N_BINS = 8;   // what a caller passes: the grey-bin count the features ask for
    // many items, few values, through the source template the out-of-core passes use
    std::vector<Pixel2> cloud;
    for (int k = 0; k < 20000; k++)
        cloud.push_back (Pixel2 ((StatsInt) (k % 100), (StatsInt) (k / 100), (PixIntens) (100 + (k % 7))));
    {
        TrivialHistogram h;
        h.initialize (N_BINS, (HistoItem) 100, (HistoItem) 106, cloud);
        EXPECT_EQ(h.n_distinct(), (size_t) 7)
            << "one entry per grey level the cloud carries, not one per voxel";
        EXPECT_LT(h.n_distinct(), cloud.size() / 1000) << "and nothing that grows with the voxel count";
    }

    // and a second, wider cloud through the same instance replaces the first rather than adding
    {
        TrivialHistogram h;
        h.initialize (N_BINS, (HistoItem) 100, (HistoItem) 106, cloud);
        // disjoint from the first cloud's 100..106, so a map that accumulated would hold 507
        std::vector<Pixel2> wider;
        for (int k = 0; k < 500; k++)
            wider.push_back (Pixel2 ((StatsInt) k, (StatsInt) 0, (PixIntens) (1000 + k)));
        h.initialize (N_BINS, (HistoItem) 1000, (HistoItem) 1499, wider);
        EXPECT_EQ(h.n_distinct(), (size_t) 500) << "the second cloud alone, not both";
    }

    // the chords path keeps its own loop over the same map
    {
        TrivialHistogram h;
        std::vector<HistoItem> data;
        for (int k = 0; k < 20000; k++)
            data.push_back ((HistoItem) (100 + (k % 7)));
        h.initialize_uniques (data);
        EXPECT_EQ(h.n_distinct(), (size_t) 7);
    }
}

// with_tiff_sample_type dispatches on the (SampleFormat, BitsPerSample) pair a TIFF declares, and
// refuses a pair it has no C++ type for. What this discriminates: falling through to a default
// type reads the file's bytes as something they are not, silently, for every pixel.
void test_tiff_sample_unsupported_throws_mechanics()
{
    auto nothing = [](auto) {};

    // supported pairs go through
    EXPECT_NO_THROW(Nyxus::with_tiff_sample_type (SAMPLEFORMAT_UINT, 8, "test", nothing));
    EXPECT_NO_THROW(Nyxus::with_tiff_sample_type (SAMPLEFORMAT_UINT, 16, "test", nothing));
    EXPECT_NO_THROW(Nyxus::with_tiff_sample_type (SAMPLEFORMAT_INT, 32, "test", nothing));
    EXPECT_NO_THROW(Nyxus::with_tiff_sample_type (SAMPLEFORMAT_IEEEFP, 32, "test", nothing));
    EXPECT_NO_THROW(Nyxus::with_tiff_sample_type (SAMPLEFORMAT_IEEEFP, 16, "test", nothing))
        << "a 16-bit IEEE sample is half-float, which TiffHalf covers";

    // and the ones with no type behind them are refused, by name
    EXPECT_THROW(Nyxus::with_tiff_sample_type (SAMPLEFORMAT_UINT, 4, "test", nothing), std::runtime_error);
    EXPECT_THROW(Nyxus::with_tiff_sample_type (SAMPLEFORMAT_IEEEFP, 8, "test", nothing), std::runtime_error);
    EXPECT_THROW(Nyxus::with_tiff_sample_type (SAMPLEFORMAT_COMPLEXINT, 64, "test", nothing), std::runtime_error);
    try
    {
        Nyxus::with_tiff_sample_type (SAMPLEFORMAT_UINT, 4, "who-asked", nothing);
        FAIL() << "expected a throw";
    }
    catch (const std::runtime_error& e)
    {
        EXPECT_NE(std::string (e.what()).find ("who-asked"), std::string::npos)
            << "the message names the reader that asked: " << e.what();
    }
}
