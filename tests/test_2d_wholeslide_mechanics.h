#pragma once

// The 2D whole-slide workflow's refusal points: the places where a slide is declined rather than
// featurized. Each fires only on input the value fixtures never produce, so none is reachable
// from a test that compares numbers.

#include <gtest/gtest.h>
#include <fstream>
#include <string>
#include <vector>
#include "../src/nyx/environment.h"
#include "../src/nyx/globals.h"
#include "../src/nyx/features/intensity.h"	// PixelIntensityFeatures::featureset
#include "../src/nyx/helpers/fsystem.h"
#include "test_main_nyxus.h"

// The 2D whole-slide worker refuses a slide whose loader will not open, and reports that refusal
// through its status. What this discriminates: falling through reads the slide through loaders
// open() never allocated, and the status it would otherwise carry is the one the run's exit code
// is built from. The CLI test covers the rest of that chain but cannot reach this link -- a run
// whose prescan already rejected the file never gets here.
void test_2d_wsi_thread_unopenable_slide_mechanics()
{
    fs::path missing = fs::temp_directory_path() / "nyxus_no_such_slide.tif";
    std::error_code ec;
    fs::remove (missing, ec);
    ASSERT_FALSE(fs::exists (missing));

    fs::path outdir = fs::temp_directory_path() / "nyxus_wsi_unopenable_out";
    fs::remove_all (outdir, ec);
    fs::create_directories (outdir);

    Environment e;
    e.set_dim (2);
    e.theFeatureSet.enableAll (false);
    e.theFeatureSet.enableFeatures (PixelIntensityFeatures::featureset);
    ASSERT_TRUE(e.theFeatureMgr.compile());
    e.theFeatureMgr.apply_user_selection (e.theFeatureSet);
    ASSERT_TRUE(e.theFeatureMgr.init_feature_classes());
    e.compile_feature_settings();
    e.output_dir = outdir.string();

    // the slide entry a prescan would have left, pointing at a file that is not there
    e.dataset.dataset_props.emplace_back (missing.string(), "");

    std::vector<std::string> ifiles { missing.string() }, mfiles { std::string() };
    int rv = 0;
    ASSERT_NO_THROW(Nyxus::featurize_wsi_thread (e, ifiles, mfiles, 0, 1,
        outdir.string(), false, Nyxus::SaveOption::saveCSV, rv))
        << "an unopenable slide is refused, not read through";
    EXPECT_NE(rv, 0) << "and the refusal is the slide's status";

    size_t datarows = 0;
    for (auto& de : fs::directory_iterator (outdir))
        if (de.path().extension() == ".csv")
        {
            std::ifstream f (de.path()); std::string ln; size_t n = 0;
            while (std::getline (f, ln)) if (!ln.empty()) ++n;
            if (n) datarows += n - 1;
        }
    EXPECT_EQ(datarows, (size_t) 0) << "no row is written for a slide that was never read";

    fs::remove_all (outdir, ec);
}
