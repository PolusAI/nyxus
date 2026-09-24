#pragma once

// Guards and failure exits of the 3D out-of-core and whole-volume paths: each fires only on
// input the value-parity fixtures never produce, so none of them is reachable from a test that
// compares numbers.

#include <gtest/gtest.h>
#include <cstdio>
#include <fstream>
#include <map>
#include <string>
#include <vector>
#include "test_main_nyxus.h"	// write_tiled_plane_u16, ometiff_data_path
#include "../src/nyx/environment.h"
#include "../src/nyx/globals.h"
#include "../src/nyx/features/histogram.h"
#include "../src/nyx/features/3d_ooc_volume.h"
#include "../src/nyx/features/voxel_cloud_nontriv.h"
#include "../src/nyx/tiff_sample.h"
#include "../src/nyx/helpers/fsystem.h"

// OocBinnedVolume's two guards. Neither fires on any parity fixture: every family that builds the
// volume with a mask asks for the mask, and every family walks planes inside the ROI's own depth.
void test_3d_ooc_volume_guards_mechanics()
{
    // a minimal ROI with a disk-backed cloud of one plane
    LR r (1);
    r.aabb.init_x (0); r.aabb.update_x (3);
    r.aabb.init_y (0); r.aabb.update_y (3);
    r.aabb.init_z (0); r.aabb.update_z (0);
    r.raw_voxels_NT.init (r.label, "raw_voxels_NT_guardtest");
    r.raw_voxels_NT.begin_slab (0);
    for (int y = 0; y < 4; y++)
        for (int x = 0; x < 4; x++)
            r.raw_voxels_NT.add_voxel (Pixel3 (x, y, 0, (PixIntens)(1 + x + y)));

    auto identity = [](PixIntens v) { return v; };

    // built WITHOUT the mask: asking for it is a programming error, not a data one
    {
        Nyxus::OocBinnedVolume vol (r, identity, 0, 1, /*with_mask=*/ false);
        EXPECT_THROW(vol.mask (0), std::logic_error)
            << "a volume built without the ROI mask has no mask to hand out";
        EXPECT_NO_THROW(vol.plane (0));
    }

    // a plane outside the ROI's depth is rejected rather than indexed: slot() wraps modulo the
    // window, so an unchecked lz would quietly return some other plane's buffer
    {
        Nyxus::OocBinnedVolume vol (r, identity, 0, 1, /*with_mask=*/ true);
        EXPECT_NO_THROW(vol.plane (0));
        EXPECT_THROW(vol.plane (r.aabb.get_z_depth()), std::out_of_range);
        EXPECT_THROW(vol.plane (-1), std::out_of_range);
        EXPECT_THROW(vol.mask (r.aabb.get_z_depth()), std::out_of_range);
    }

    r.raw_voxels_NT.clear();
}

// The offset a record sits at passes 2 GB at about 134 million voxels, which an oversized ROI
// reaches. What this discriminates: a seek taken through the C `long` overload truncates that
// offset on Windows, where long is 32 bits, and the read lands somewhere else in the file. This
// asserts the arithmetic and the seek without writing a 2 GB file: seeking past the end is legal
// and allocates nothing.
void test_3d_voxel_cloud_seek_beyond_2gb_mechanics()
{
    fs::path f = fs::temp_directory_path() / "nyxus_seek2gb.bin";
    std::error_code ec;
    fs::remove (f, ec);
    {
        FILE* w = fopen (f.string().c_str(), "wb");
        ASSERT_NE(w, nullptr);
        const char probe[16] = { 0 };
        ASSERT_EQ(fwrite (probe, 1, sizeof(probe), w), sizeof(probe));
        fclose (w);
    }

    FILE* fp = fopen (f.string().c_str(), "rb");
    ASSERT_NE(fp, nullptr);

    // the record index an oversized cloud reaches, and the byte offset it lands on. The record
    // on disk is the four fields, not sizeof(Pixel3): the struct carries padding and a base the
    // cloud does not write, so sizeof would compute an offset the reader never seeks to.
    const size_t item_size = sizeof (Pixel3::x) + sizeof (Pixel3::y)
        + sizeof (Pixel3::z) + sizeof (Pixel3::inten);
    const size_t index = (size_t) 3 * 1024 * 1024 * 1024 / item_size + 7;
    const size_t offset = index * item_size;
    ASSERT_GT(offset, (size_t) 0x7fffffff) << "the fixture must exceed what a 32-bit offset holds";

    ASSERT_NO_THROW(Nyxus::seek_record (fp, offset));
#ifdef _WIN32
    EXPECT_EQ((size_t) _ftelli64 (fp), offset) << "the file position is the offset asked for";
#else
    EXPECT_EQ((size_t) ftello (fp), offset) << "the file position is the offset asked for";
#endif

    fclose (fp);
    fs::remove (f, ec);
}

// A pair that streams has no refusal to report, and the callers read that off the empty string
// rather than asking a second question. What this discriminates: a reason that returns prose for
// a streamable pair turns the callers' "is there a reason" test into a tautology, and the refusal
// message then appears on a failure that had nothing to do with streaming.
void test_3d_ooc_reason_empty_when_bounded_mechanics()
{
    fs::path ds = fs::temp_directory_path() / "nyxus_reason_bounded.tif";
    std::error_code ec;
    fs::remove (ds, ec);
    write_tiled_plane_u16 (ds, 32, 32, 16, [](uint32_t x, uint32_t y) { return (uint16_t)(1 + x + y); });

    SlideProps p;
    p.fname_int = ds.string();
    p.fname_seg = "";
    FpImageOptions fp;
    ImageLoader il;
    ASSERT_TRUE(il.open (p, fp)) << ds.string();
    ASSERT_TRUE(il.streams_bounded()) << "a strip/tile TIFF plane streams";

    EXPECT_TRUE(Nyxus::ooc_unstreamable_reason (il).empty())
        << "a streamable pair has no refusal to explain";

    il.close();
    fs::remove (ds, ec);
}

// A slide whose loader will not open is refused before anything reads through that loader. What
// this discriminates: carrying on dereferences loaders open() never allocated -- streams_bounded()
// asks the intensity loader for its extents first thing -- so this is a crash, not a wrong value.
void test_3d_wv_thread_unopenable_slide_mechanics()
{
    fs::path missing = fs::temp_directory_path() / "nyxus_no_such_volume.tif";
    std::error_code ec;
    fs::remove (missing, ec);
    ASSERT_FALSE(fs::exists (missing));

    fs::path outdir = fs::temp_directory_path() / "nyxus_wv_unopenable_out";
    fs::remove_all (outdir, ec);
    fs::create_directories (outdir);

    Environment e;
    e.set_dim (3);
    e.theFeatureSet.enableAll (false);
    e.theFeatureSet.enableFeatures (D3_VoxelIntensityFeatures::featureset);
    ASSERT_TRUE(e.theFeatureMgr.compile());
    e.theFeatureMgr.apply_user_selection (e.theFeatureSet);
    ASSERT_TRUE(e.theFeatureMgr.init_feature_classes());
    e.compile_feature_settings();
    e.output_dir = outdir.string();

    // the slide entry a prescan would have left, pointing at a file that is not there
    e.dataset.dataset_props.emplace_back (missing.string(), "");
    e.dataset.dataset_props[0].inten_channels = 1;
    e.dataset.dataset_props[0].inten_time = 1;

    std::vector<std::string> ifiles { missing.string() }, mfiles;
    int rv = 1;

    // rv alone does not discriminate: the open failure sets rv = 1 whether or not the function
    // then carries on. What does is that nothing downstream runs -- featurize_wholevolume announces
    // itself here, and reverting the early return faults inside it rather than reaching this line.
    e.set_verbosity_level (2);
    std::string out;
    bool threw = false;
    testing::internal::CaptureStdout();
    try
    {
        rv = Nyxus::featurize_3d_wv_thread (e, ifiles, mfiles, 0, 1,
            outdir.string(), false, Nyxus::SaveOption::saveCSV);
    }
    catch (...)
    {
        threw = true;
    }
    // ends the capture on every path: leaving it open would swallow the rest of the process's
    // stdout, including other tests' failure output
    out = testing::internal::GetCapturedStdout();
    ASSERT_FALSE(threw) << "an unopenable slide is refused, not read through";

    EXPECT_NE(rv, 0) << "the refusal is the slide's status";
    EXPECT_EQ(out.find ("Gathering vROI metrics"), std::string::npos)
        << "no pass over the slide may begin once its loader has refused to open:\n" << out;

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

// A batch scan that fails carries its failure out of phase 2, so the pair fails rather than
// reducing ROIs over voxels that were never cached. What this discriminates: a phase 2 that
// discards the scan's status goes on to allocate buffers and reduce, and writes rows of zeros for
// ROIs it never read -- with nyxus reporting success.
void test_3d_trivial_rois_scan_failure_propagates_mechanics()
{
    fs::path src_i = ometiff_data_path ("dim3_zyx.ome.tif"),
        src_m = ometiff_data_path ("dim3_mask.ome.tif");
    ASSERT_TRUE(fs::exists (src_i)) << src_i.string();
    ASSERT_TRUE(fs::exists (src_m)) << src_m.string();

    fs::path dir = fs::temp_directory_path() / "nyxus_3d_scanfail";
    std::error_code ec;
    fs::remove_all (dir, ec);
    fs::create_directories (dir);
    fs::path i_copy = dir / "i.ome.tif", m_copy = dir / "m.ome.tif";
    fs::copy_file (src_i, i_copy, fs::copy_options::overwrite_existing);
    fs::copy_file (src_m, m_copy, fs::copy_options::overwrite_existing);

    Environment e;
    e.set_dim (3);
    e.theFeatureSet.enableAll (false);
    e.theFeatureSet.enableFeatures (D3_VoxelIntensityFeatures::featureset);
    ASSERT_TRUE(e.theFeatureMgr.compile());
    e.theFeatureMgr.apply_user_selection (e.theFeatureSet);
    ASSERT_TRUE(e.theFeatureMgr.init_feature_classes());
    e.compile_feature_settings();
    ASSERT_TRUE(e.set_ram_limit (64));	// the fixture is kilobytes; a bigger limit is one a busy box can refuse

    SlideProps& sp = e.dataset.dataset_props.emplace_back (i_copy.string(), m_copy.string());
    ASSERT_TRUE(Nyxus::scan_slide_props (sp, 3, e.anisoOptions, e.use_physical_spacing(),
        e.fpimageOptions, e.resultOptions.need_annotation()));
    e.dataset.update_dataset_props_extrema();

    ASSERT_TRUE(Nyxus::gatherRoisMetrics_3D (e, 0, i_copy.string(), m_copy.string(), 0, 0));
    ASSERT_GT(e.uniqueLabels.size(), (size_t) 0);
    for (auto lab : e.uniqueLabels)
        e.roiData[lab].initialize_fvals();
    std::vector<int> triv (e.uniqueLabels.begin(), e.uniqueLabels.end());

    // the volume goes away between the two passes, which is what a scan failing mid-run looks
    // like from phase 2's side
    fs::remove (i_copy, ec);
    ASSERT_FALSE(fs::exists (i_copy));

    EXPECT_FALSE(Nyxus::processTrivialRois_3D (e, 0, 0, 0, triv, i_copy.string(), m_copy.string(),
        e.get_ram_limit())) << "a batch that could not be scanned must fail the pass";

    fs::remove_all (dir, ec);
}
