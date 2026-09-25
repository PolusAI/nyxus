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

// An anisotropic 2.5D pass over a multi-tile slide. The virtual grid is walked by mapping each
// virtual pixel back to its physical one and taking the tile from that, which cannot ask for a
// tile the slide does not have. What this discriminates: deriving the tile from a VIRTUAL tile
// width instead -- (tile * factor), truncated -- lets the virtual extent exceed tile_count *
// virtual_tile_width, so the last column asks for one tile past the end. Here 32 px of 16-px tiles
// at 0.8 gives a virtual width of 25 over a virtual tile of 12, and virtual column 24 asks for
// tile 2 of 2. load_tile refuses it, and since the pass now carries its scan's status, an
// ordinary anisotropy factor fails the whole run on legitimate input.
void test_3d_layouta_anisotropic_tile_index_mechanics()
{
    Z25Stack s;
    make_25d_stack (s, "aniso", 32, 32, 16, 2);   // a 2x2 tile grid

    Environment e;
    enable_3d_intensity (e);
    prescan_25d_stack (e, s);
    ASSERT_TRUE(e.set_ram_limit (64));
    e.anisoOptions.set_aniso_x (0.8);
    e.anisoOptions.set_aniso_y (0.8);
    ASSERT_TRUE(e.anisoOptions.customized()) << "the anisotropic branch is the one under test";

    ASSERT_TRUE(Nyxus::gatherRoisMetrics_25D (e, 0, s.int_pattern, s.seg_pattern, s.z_indices));
    ASSERT_EQ(e.uniqueLabels.size(), (size_t) 1);
    for (auto lab : e.uniqueLabels)
        e.roiData[lab].initialize_fvals();
    std::vector<int> triv (e.uniqueLabels.begin(), e.uniqueLabels.end());

    ASSERT_TRUE(Nyxus::processTrivialRois_25D (e, triv, s.int_pattern, s.seg_pattern,
        e.get_ram_limit(), s.z_indices))
        << "an ordinary anisotropy factor must not ask for a tile past the end of the grid";

    // and the walk reached the ROI rather than stopping short: the reduce, which frees the cloud
    // behind it, produced values inside the range the ROI's voxels carry
    const double vmin = e.roiData[1].get_fvals ((int) Nyxus::Feature3D::MIN)[0],
        vmax = e.roiData[1].get_fvals ((int) Nyxus::Feature3D::MAX)[0];
    EXPECT_GT(vmax, 0.0);
    EXPECT_GE(vmax, vmin);

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

// The anisotropic scan's MAPPING, not just its grid walk: every virtual voxel must carry the
// intensity of the physical pixel it maps back to, at the virtual coordinate it was asked for.
//
// What this discriminates, and why the other anisotropic fixtures cannot: the mapping is
// verifiably identical to any plausible wrong one when the factors are 1, when they divide
// exactly (0.5, 2.0), when they are equal on both axes, or when the slide is a single tile --
// which is every anisotropic fixture in the tree. This slide is oblong (48x32 over 16-px tiles,
// so 3 tile columns by 2 rows), its factors differ per axis and divide nothing exactly (0.6 and
// 1.4), and one of them is above 1, so the virtual extent exceeds the physical one. A transposed
// map, an off-by-one reverse map, or a bound that compares virtual coordinates against physical
// extents each change the values or the count here.
void test_3d_layouta_anisotropic_resampling_mechanics()
{
    const uint32_t W = 48, H = 32, TILE = 16;
    const double AX = 0.6, AY = 1.4;

    Z25Stack s;
    make_25d_full_plane (s, "resample", W, H, TILE);

    Environment e;
    enable_3d_intensity (e);
    prescan_25d_stack (e, s);
    ASSERT_TRUE(e.set_ram_limit (64));

    // the same factors the scan below is given: phase 1 reads them to build the ROI's extent, so
    // leaving them unset would size the ROI from the physical grid while the scan caches a virtual
    // cloud, and the two would be describing different geometry
    e.anisoOptions.set_aniso_x (AX);
    e.anisoOptions.set_aniso_y (AY);
    ASSERT_TRUE(e.anisoOptions.customized());

    ASSERT_TRUE(Nyxus::gatherRoisMetrics_25D (e, 0, s.int_pattern, s.seg_pattern, s.z_indices));
    ASSERT_EQ(e.uniqueLabels.size(), (size_t) 1);
    std::vector<int> batch (e.uniqueLabels.begin(), e.uniqueLabels.end());
    for (auto lab : batch)
        e.roiData[lab].initialize_fvals();

    ASSERT_TRUE(Nyxus::scanTrivialRois_25D_anisotropic (e, batch, s.int_pattern, s.seg_pattern,
        s.z_indices, AX, AY, 1.0));

    LR& r = e.roiData[1];
    const size_t vw = (size_t) (double(W) * AX), vh = (size_t) (double(H) * AY);
    EXPECT_EQ(r.raw_pixels_3D.size(), vw * vh)
        << "one virtual voxel per cell of the virtual grid: " << vw << " x " << vh;

    // every voxel, at its own coordinate, carries its physical pixel's value
    size_t checked = 0, wrong = 0;
    std::string first_wrong;
    for (auto& px : r.raw_pixels_3D)
    {
        const size_t vc = (size_t) px.x, vr = (size_t) px.y;
        ASSERT_LT(vc, vw);
        ASSERT_LT(vr, vh);
        const uint32_t ph_col = (uint32_t) (double(vc) / AX), ph_row = (uint32_t) (double(vr) / AY);
        ASSERT_LT(ph_col, W);
        ASSERT_LT(ph_row, H);
        const uint32_t want = plane_enc (W, ph_col, ph_row);
        if ((uint32_t) px.inten != want)
        {
            if (wrong == 0)
                first_wrong = "virtual (" + std::to_string(vc) + "," + std::to_string(vr) + ") -> physical ("
                    + std::to_string(ph_col) + "," + std::to_string(ph_row) + "): got "
                    + std::to_string((uint32_t) px.inten) + ", want " + std::to_string(want);
            wrong++;
        }
        checked++;
    }
    EXPECT_EQ(wrong, (size_t) 0) << wrong << " of " << checked << " voxels carry the wrong pixel; first: " << first_wrong;
    EXPECT_EQ(checked, vw * vh);

    // and the far corner is present: a virtual row past the physical height must not be dropped
    const size_t last_vr = vh - 1;
    EXPECT_GT(last_vr, (size_t) H - 1) << "the fixture must have more virtual rows than physical ones";
    bool saw_last_row = false;
    for (auto& px : r.raw_pixels_3D)
        if ((size_t) px.y == last_vr) { saw_last_row = true; break; }
    EXPECT_TRUE(saw_last_row) << "the last virtual row is part of the cloud";

    fs::remove_all (s.dir);
}


// The ROI's extent and voxel count must describe the cloud the anisotropic scan cached. The extent
// sizes aux_image_cube and calculate_from_pixelcloud writes each voxel at its own coordinate, so a
// voxel outside the extent lands in the cube's next row, or past its end from the last row; the
// voxel count divides every feature that averages.
//
// What this discriminates, and why the other anisotropic fixtures cannot: an extent taken from the
// ROI's PHYSICAL box and scaled agrees with the scan at 1, at factors that divide exactly, and at
// every factor below 2 -- which is all of them elsewhere in this tree. The two part company once a
// factor passes 2, because the scaled box ends at (last physical column * factor) while the scan
// runs to the virtual width, (width * factor) - 1. Here x is 2.5, so they differ by a column, and
// y is 1.4, so the axes cannot agree by being equal.
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

    // the virtual grid, derived the way the scan derives it
    const size_t vw = (size_t)(double(W) * AX), vh = (size_t)(double(H) * AY);
    ASSERT_GT(vw, (size_t)(double(W - 1) * AX) + 1)
        << "the fixture's x factor must be one where a scaled physical box falls short of the virtual grid";

    LR& r = e.roiData[1];
    EXPECT_EQ((size_t) r.aabb.get_xmin(), (size_t) 0);
    EXPECT_EQ((size_t) r.aabb.get_ymin(), (size_t) 0);
    EXPECT_EQ((size_t) r.aabb.get_xmax(), vw - 1) << "the extent's last column is the scan's last virtual column";
    EXPECT_EQ((size_t) r.aabb.get_ymax(), vh - 1) << "the extent's last row is the scan's last virtual row";
    EXPECT_EQ((size_t) r.aux_area, vw * vh) << "the voxel count is the virtual grid's, not the physical one's";

    fs::remove_all (s.dir);
}

// A ROI thinner than the factor's step maps to no virtual voxel at all: at 0.3 the scan's virtual
// columns land on physical 0, 3, 6, 10, ... so a ROI one pixel wide at column 5 is never sampled,
// and the same holds for its row. There is no cloud to take an extent or a voxel count from, so
// the pass names the ROI and refuses instead of reducing a cube of zeros.
void test_3d_layouta_anisotropic_vanished_roi_is_refused_mechanics()
{
    const uint32_t W = 48, H = 32, TILE = 16, RX = 5, RY = 5;
    const double AX = 0.3, AY = 0.3;

    Z25Stack s;
    make_25d_full_plane (s, "aniso_vanished", W, H, TILE);
    // one pixel, at a position no virtual coordinate maps back to
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
    ASSERT_EQ(e.uniqueLabels.size(), (size_t) 1) << "phase 1 finds the ROI on the physical grid";
    std::vector<int> triv (e.uniqueLabels.begin(), e.uniqueLabels.end());
    for (auto lab : triv)
        e.roiData[lab].initialize_fvals();

    EXPECT_FALSE(Nyxus::processTrivialRois_25D (e, triv, s.int_pattern, s.seg_pattern,
        e.get_ram_limit(), s.z_indices))
        << "a ROI the resampling drops cannot be featurized, and the pass says so";
    EXPECT_TRUE(e.roiData[1].raw_pixels_3D.empty()) << "and it is empty because the scan reached no voxel of it";

    fs::remove_all (s.dir);
}
