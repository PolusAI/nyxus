#pragma once

// The 2.5D (layoutA) path: a volume given as a collection of per-Z slice files, addressed through
// a '*' placeholder in the file name and a list of z-index strings. Its two passes are
// gatherRoisMetrics_25D (phase 1) and scanTrivialRois_25D / processTrivialRois_25D (phase 2);
// both walk each plane's tile grid themselves rather than through the volumetric streamer, which
// is why the grid walk is asserted here and not only on the 2D whole-slide path.

#include <gtest/gtest.h>
#include <functional>
#include <string>
#include <vector>
#include "../src/nyx/environment.h"
#include "../src/nyx/globals.h"
#include "../src/nyx/helpers/fsystem.h"
#include "test_main_nyxus.h"	// write_tiled_plane_u16

// A layoutA stack: planes named <stem>_z<N>.tif beside masks named <stem>_mask_z<N>.tif, with one
// ROI (label 1) filling the same rectangle on every plane. Returns the two '*'-placeholder paths.
struct Z25Stack
{
    fs::path dir;
    std::string int_pattern, seg_pattern;
    std::vector<std::string> z_indices;
    uint32_t W = 0, H = 0;
    uint32_t rx0 = 0, rx1 = 0, ry0 = 0, ry1 = 0;   // the ROI's half-open box

    size_t roi_voxels() const { return (size_t)(rx1 - rx0) * (ry1 - ry0) * z_indices.size(); }
    static uint16_t enc (uint32_t x, uint32_t y, uint32_t z) { return (uint16_t)(1 + ((z * 1000 + y * 53 + x) % 4000)); }
};

static void make_25d_stack (Z25Stack& s, const std::string& tag, uint32_t W, uint32_t H,
    uint32_t tile, size_t nz)
{
    s.dir = fs::temp_directory_path() / ("nyxus_25d_" + tag);
    fs::remove_all (s.dir);
    fs::create_directories (s.dir);
    s.W = W; s.H = H;
    s.rx0 = 2; s.rx1 = (std::min)(W, 10u);
    s.ry0 = 3; s.ry1 = (std::min)(H, 11u);
    s.z_indices.clear();

    for (size_t k = 0; k < nz; k++)
    {
        const uint32_t z = (uint32_t)(k + 1);
        s.z_indices.push_back (std::to_string (z));
        write_tiled_plane_u16 (s.dir / ("i_z" + std::to_string(z) + ".tif"), W, H, tile,
            [z](uint32_t x, uint32_t y) { return Z25Stack::enc (x, y, z); });
        write_tiled_plane_u16 (s.dir / ("m_z" + std::to_string(z) + ".tif"), W, H, tile,
            [&s](uint32_t x, uint32_t y)
            { return (uint16_t)((x >= s.rx0 && x < s.rx1 && y >= s.ry0 && y < s.ry1) ? 1 : 0); });
    }
    s.int_pattern = (s.dir / "i_z*.tif").string();
    s.seg_pattern = (s.dir / "m_z*.tif").string();
}

// The stack's slide entry, as a real run's prescan leaves it: every 3D feature reads the
// intensity domain back through env.dataset.dataset_props[r.slide_idx].
static void prescan_25d_stack (Environment& e, const Z25Stack& s)
{
    SlideProps& sp = e.dataset.dataset_props.emplace_back (
        (s.dir / "i_z1.tif").string(), (s.dir / "m_z1.tif").string());
    ASSERT_TRUE(Nyxus::scan_slide_props (sp, 2, e.anisoOptions, e.use_physical_spacing(),
        e.fpimageOptions, e.resultOptions.need_annotation()));
    e.dataset.update_dataset_props_extrema();
}

static void enable_3d_intensity (Environment& e)
{
    e.set_dim (3);      // layoutA is a volume given plane by plane, so the reducer takes the 3D branch
    e.theFeatureSet.enableAll (false);
    e.theFeatureSet.enableFeatures (D3_VoxelIntensityFeatures::featureset);
    ASSERT_TRUE(e.theFeatureMgr.compile());
    e.theFeatureMgr.apply_user_selection (e.theFeatureSet);
    ASSERT_TRUE(e.theFeatureMgr.init_feature_classes());
    e.compile_feature_settings();
}

// Both 2.5D passes read every plane of the stack and every voxel of each plane: phase 1 finds the
// ROI and sizes it, phase 2 caches its voxels and reduces them. This is the path's happy case,
// which nothing covered before.
void test_3d_layouta_segmented_passes_mechanics()
{
    Z25Stack s;
    make_25d_stack (s, "happy", 24, 16, 16, 3);

    Environment e;
    enable_3d_intensity (e);
    prescan_25d_stack (e, s);
    ASSERT_TRUE(e.set_ram_limit (64));	// these ROIs are kilobytes; a bigger limit is one a busy box can refuse

    ASSERT_TRUE(Nyxus::gatherRoisMetrics_25D (e, 0, s.int_pattern, s.seg_pattern, s.z_indices))
        << "phase 1 must read every plane of the stack";
    ASSERT_EQ(e.uniqueLabels.size(), (size_t) 1) << "the stack holds exactly one ROI";
    ASSERT_EQ(*e.uniqueLabels.begin(), 1);
    LR& r = e.roiData[1];
    EXPECT_EQ((size_t) r.aux_area, s.roi_voxels()) << "phase 1 sizes the ROI over all planes";

    for (auto lab : e.uniqueLabels)
        e.roiData[lab].initialize_fvals();
    std::vector<int> triv (e.uniqueLabels.begin(), e.uniqueLabels.end());
    ASSERT_TRUE(Nyxus::processTrivialRois_25D (e, triv, s.int_pattern, s.seg_pattern,
        e.get_ram_limit(), s.z_indices)) << "phase 2 must read the same voxels back";

    // the reduced values describe the ROI's voxels, and only those
    const double vmin = r.get_fvals ((int) Nyxus::Feature3D::MIN)[0],
        vmax = r.get_fvals ((int) Nyxus::Feature3D::MAX)[0];
    uint16_t want_min = 0xffff, want_max = 0;
    for (size_t k = 0; k < s.z_indices.size(); k++)
        for (uint32_t y = s.ry0; y < s.ry1; y++)
            for (uint32_t x = s.rx0; x < s.rx1; x++)
            {
                uint16_t v = Z25Stack::enc (x, y, (uint32_t)(k + 1));
                want_min = (std::min)(want_min, v);
                want_max = (std::max)(want_max, v);
            }
    EXPECT_DOUBLE_EQ(vmin, (double) want_min);
    EXPECT_DOUBLE_EQ(vmax, (double) want_max);

    fs::remove_all (s.dir);
}

// Each 2.5D pass walks its plane's tile grid directly, bounding the row by the number of tile ROWS
// and the column by the number of tile COLUMNS. What this discriminates: a walk that takes the two
// counts the other way round asks for tiles that do not exist on a grid that is not square -- here
// 3 columns and 5 rows -- and load_tile refuses them, so phase 1 never finds the ROI and phase 2
// caches nothing. The 2D whole-slide test cannot see this: it drives a different function.
void test_3d_layouta_nonsquare_tile_grid_mechanics()
{
    Z25Stack s;
    make_25d_stack (s, "oblong", 48, 80, 16, 2);   // 3 tiles across, 5 down

    Environment e;
    enable_3d_intensity (e);
    prescan_25d_stack (e, s);
    ASSERT_TRUE(e.set_ram_limit (64));	// these ROIs are kilobytes; a bigger limit is one a busy box can refuse

    ASSERT_TRUE(Nyxus::gatherRoisMetrics_25D (e, 0, s.int_pattern, s.seg_pattern, s.z_indices))
        << "phase 1 must reach every tile of an oblong grid";
    ASSERT_EQ(e.uniqueLabels.size(), (size_t) 1);
    LR& r = e.roiData[1];
    EXPECT_EQ((size_t) r.aux_area, s.roi_voxels());

    for (auto lab : e.uniqueLabels)
        e.roiData[lab].initialize_fvals();
    std::vector<int> triv (e.uniqueLabels.begin(), e.uniqueLabels.end());
    ASSERT_TRUE(Nyxus::processTrivialRois_25D (e, triv, s.int_pattern, s.seg_pattern,
        e.get_ram_limit(), s.z_indices)) << "phase 2 must reach every tile of an oblong grid";

    fs::remove_all (s.dir);
}

// A plane the pass cannot read fails the pass. What this discriminates: a phase 2 that discards
// its scan's status returns true having cached nothing for the missing plane, and the ROI is then
// reduced over a partial cloud and written out as if it had been measured.
void test_3d_layouta_unreadable_plane_fails_the_pass_mechanics()
{
    Z25Stack s;
    make_25d_stack (s, "broken", 24, 16, 16, 3);

    Environment e;
    enable_3d_intensity (e);
    ASSERT_TRUE(e.set_ram_limit (64));	// these ROIs are kilobytes; a bigger limit is one a busy box can refuse

    // phase 1 over the intact stack, so phase 2 has an ROI to work on
    ASSERT_TRUE(Nyxus::gatherRoisMetrics_25D (e, 0, s.int_pattern, s.seg_pattern, s.z_indices));
    ASSERT_EQ(e.uniqueLabels.size(), (size_t) 1);
    for (auto lab : e.uniqueLabels)
        e.roiData[lab].initialize_fvals();

    // now the middle plane goes missing
    std::error_code ec;
    fs::remove (s.dir / "i_z2.tif", ec);
    ASSERT_FALSE(fs::exists (s.dir / "i_z2.tif"));

    std::vector<int> triv (e.uniqueLabels.begin(), e.uniqueLabels.end());
    EXPECT_FALSE(Nyxus::processTrivialRois_25D (e, triv, s.int_pattern, s.seg_pattern,
        e.get_ram_limit(), s.z_indices)) << "a plane that cannot be read must fail the pass";

    // and phase 1 reports it the same way
    Environment e2;
    enable_3d_intensity (e2);
    EXPECT_FALSE(Nyxus::gatherRoisMetrics_25D (e2, 0, s.int_pattern, s.seg_pattern, s.z_indices));

    fs::remove_all (s.dir);
}

// An anisotropic 2.5D pass over a multi-tile slide reads the stack as acquired, every tile of every
// plane once, and reports the values the isotropic pass reports: the factors reach the shape family
// alone. What this discriminates: a pass that resamples by the factors walks a virtual grid whose
// tile arithmetic can ask for a tile past the end of the grid (32 px of 16-px tiles at 0.8), and
// whose duplicated and dropped voxels move every first-order value.
void test_3d_layouta_anisotropic_tile_index_mechanics()
{
    Z25Stack s;
    make_25d_stack (s, "aniso", 32, 32, 16, 2);   // a 2x2 tile grid

    auto run = [&](bool anisotropic) -> std::vector<std::vector<double>>
    {
        Environment e;
        enable_3d_intensity (e);
        prescan_25d_stack (e, s);
        EXPECT_TRUE(e.set_ram_limit (64));
        if (anisotropic)
        {
            e.anisoOptions.set_aniso_x (0.8);
            e.anisoOptions.set_aniso_y (0.8);
            EXPECT_TRUE(e.anisoOptions.customized());
        }

        EXPECT_TRUE(Nyxus::gatherRoisMetrics_25D (e, 0, s.int_pattern, s.seg_pattern, s.z_indices));
        EXPECT_EQ(e.uniqueLabels.size(), (size_t) 1);
        for (auto lab : e.uniqueLabels)
            e.roiData[lab].initialize_fvals();
        std::vector<int> triv (e.uniqueLabels.begin(), e.uniqueLabels.end());

        EXPECT_TRUE(Nyxus::processTrivialRois_25D (e, triv, s.int_pattern, s.seg_pattern,
            e.get_ram_limit(), s.z_indices));
        return e.roiData[1].fvals;
    };
    auto iso = run (false), aniso = run (true);

    for (auto f : { Nyxus::Feature3D::MIN, Nyxus::Feature3D::MAX, Nyxus::Feature3D::MEAN,
        Nyxus::Feature3D::ENERGY, Nyxus::Feature3D::STANDARD_DEVIATION })
        EXPECT_EQ(aniso[(int) f][0], iso[(int) f][0]) << "feature " << (int) f << " moved with the factors";
    EXPECT_GT(iso[(int) Nyxus::Feature3D::MAX][0], 0.0) << "the walk reached the ROI";

    fs::remove_all (s.dir);
}

// What a full-plane fixture stores at (x,y). Distinct for every cell of the planes below, so a
// resampled voxel can be checked against the physical pixel it has to have come from; the tests
// that verify a mapping call this on the physical coordinate rather than restating the formula.
static uint16_t plane_enc (uint32_t W, uint32_t x, uint32_t y)
{
    return (uint16_t) (1 + (y * W + x) % 4000);
}

// One plane whose ROI is the whole plane: the shape each anisotropic test below starts from.
static void make_25d_full_plane (Z25Stack& s, const std::string& tag, uint32_t W, uint32_t H, uint32_t tile)
{
    s.dir = fs::temp_directory_path() / ("nyxus_25d_" + tag);
    fs::remove_all (s.dir);
    fs::create_directories (s.dir);
    s.W = W; s.H = H;
    s.rx0 = 0; s.rx1 = W; s.ry0 = 0; s.ry1 = H;
    s.z_indices = { "1" };
    write_tiled_plane_u16 (s.dir / "i_z1.tif", W, H, tile,
        [W](uint32_t x, uint32_t y) { return plane_enc (W, x, y); });
    write_tiled_plane_u16 (s.dir / "m_z1.tif", W, H, tile,
        [](uint32_t, uint32_t) { return (uint16_t) 1; });
    s.int_pattern = (s.dir / "i_z*.tif").string();
    s.seg_pattern = (s.dir / "m_z*.tif").string();
}

// The ROI's extent and voxel count describe the cloud the scan caches, which is the stack as
// acquired whatever the factors, and the ROI carries the factors for the shape family. The extent
// sizes aux_image_cube, which calculate_from_pixelcloud fills by coordinate, and the voxel count
// divides every feature that averages.
//
// What this discriminates: x is 2.5 and y is 1.4, so a box scaled by the factors reaches column
// 118 and row 43 where the cloud ends at 47 and 31, and a resampled cloud holds 120 x 44 voxels
// where the stack has 48 x 32.
void test_3d_layouta_anisotropic_cloud_fits_its_aabb_mechanics()
{
    const uint32_t W = 48, H = 32, TILE = 16;
    const double AX = 2.5, AY = 1.4;

    Z25Stack s;
    make_25d_full_plane (s, "aniso_aabb", W, H, TILE);

    Environment e;
    enable_3d_intensity (e);
    prescan_25d_stack (e, s);
    ASSERT_TRUE(e.set_ram_limit (64));
    e.anisoOptions.set_aniso_x (AX);
    e.anisoOptions.set_aniso_y (AY);
    ASSERT_TRUE(e.anisoOptions.customized());

    ASSERT_TRUE(Nyxus::gatherRoisMetrics_25D (e, 0, s.int_pattern, s.seg_pattern, s.z_indices));
    ASSERT_EQ(e.uniqueLabels.size(), (size_t) 1);
    std::vector<int> triv (e.uniqueLabels.begin(), e.uniqueLabels.end());
    for (auto lab : triv)
        e.roiData[lab].initialize_fvals();

    // the whole pass, so the cube is allocated from the extent and filled from the cloud
    ASSERT_TRUE(Nyxus::processTrivialRois_25D (e, triv, s.int_pattern, s.seg_pattern,
        e.get_ram_limit(), s.z_indices));

    LR& r = e.roiData[1];
    EXPECT_EQ((size_t) r.aabb.get_xmin(), (size_t) 0);
    EXPECT_EQ((size_t) r.aabb.get_ymin(), (size_t) 0);
    EXPECT_EQ((size_t) r.aabb.get_xmax(), (size_t) W - 1) << "the extent's last column is the stack's";
    EXPECT_EQ((size_t) r.aabb.get_ymax(), (size_t) H - 1) << "the extent's last row is the stack's";
    EXPECT_EQ((size_t) r.aux_area, (size_t) W * H) << "the voxel count is the stack's";
    EXPECT_DOUBLE_EQ(r.spacing_x, AX);
    EXPECT_DOUBLE_EQ(r.spacing_y, AY);
    EXPECT_DOUBLE_EQ(r.spacing_z, 1.0);

    fs::remove_all (s.dir);
}

// A one-pixel ROI at factors 0.3 is featurized like any other: the stack is read as acquired, so
// its pixel is cached and reduced. What this discriminates: a pass that resampled by 0.3 would
// read physical columns 0, 3, 6, 10, ... only, never reach column 5, and have no voxel of this ROI
// to featurize.
void test_3d_layouta_anisotropic_thin_roi_is_featurized_mechanics()
{
    const uint32_t W = 48, H = 32, TILE = 16, RX = 5, RY = 5;
    const double AX = 0.3, AY = 0.3;

    Z25Stack s;
    make_25d_full_plane (s, "aniso_vanished", W, H, TILE);
    // one pixel, at a position a 0.3 resampling never reads
    write_tiled_plane_u16 (s.dir / "m_z1.tif", W, H, TILE,
        [RX, RY](uint32_t x, uint32_t y) { return (uint16_t)((x == RX && y == RY) ? 1 : 0); });
    s.rx0 = RX; s.rx1 = RX + 1; s.ry0 = RY; s.ry1 = RY + 1;

    Environment e;
    enable_3d_intensity (e);
    prescan_25d_stack (e, s);
    ASSERT_TRUE(e.set_ram_limit (64));
    e.anisoOptions.set_aniso_x (AX);
    e.anisoOptions.set_aniso_y (AY);
    ASSERT_TRUE(e.anisoOptions.customized());

    ASSERT_TRUE(Nyxus::gatherRoisMetrics_25D (e, 0, s.int_pattern, s.seg_pattern, s.z_indices));
    ASSERT_EQ(e.uniqueLabels.size(), (size_t) 1);
    std::vector<int> triv (e.uniqueLabels.begin(), e.uniqueLabels.end());
    for (auto lab : triv)
        e.roiData[lab].initialize_fvals();

    ASSERT_TRUE(Nyxus::processTrivialRois_25D (e, triv, s.int_pattern, s.seg_pattern,
        e.get_ram_limit(), s.z_indices)) << "a one-pixel ROI is featurized whatever the factors";
    const double want = plane_enc (W, RX, RY);
    EXPECT_EQ(e.roiData[1].get_fvals ((int) Nyxus::Feature3D::MIN)[0], want);
    EXPECT_EQ(e.roiData[1].get_fvals ((int) Nyxus::Feature3D::MAX)[0], want);

    fs::remove_all (s.dir);
}
