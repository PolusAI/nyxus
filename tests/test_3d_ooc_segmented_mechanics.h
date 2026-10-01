#pragma once

// The segmented 3D out-of-core pass (processNontrivialRois_3D): which voxels it streams into an
// ROI's cloud, and the status it returns when a feature fails.
//
// The fixture is a small Z-stack pair written here: a 16x16x4 tiled volume whose mask carries two
// labels side by side inside a background border. In single-ROI mode phase 1 and phase 2 collapse
// both labels into ROI 1, so the out-of-core cloud has to collapse them too; in multi-ROI mode the
// two labels stay apart. The two regions carry intensities far apart, so a cloud holding either
// half alone is visible in the maximum, the mean and every moment.

#include <algorithm>
#include <string>
#include <vector>
#include "test_main_nyxus.h"		// gtest, libtiff, fs, globals.h, roi_cache.h
#include "../src/nyx/environment.h"
#include "../src/nyx/features/3d_intensity.h"
#include "../src/nyx/features/3d_gldzm.h"

namespace
{
	const uint32_t ooc_seg_W = 16, ooc_seg_H = 16, ooc_seg_D = 4, ooc_seg_tile = 16;

	// mask: 0 on the outer border, label 1 on the left half, label 2 on the right half
	uint16_t ooc_seg_label (uint32_t x, uint32_t y, uint32_t)
	{
		if (x == 0 || y == 0 || x == ooc_seg_W - 1 || y == ooc_seg_H - 1)
			return 0;
		return x < ooc_seg_W / 2 ? 1 : 2;
	}

	uint16_t ooc_seg_inten (uint32_t x, uint32_t y, uint32_t z)
	{
		return ooc_seg_label (x, y, z) == 2
			? (uint16_t) (500 + (x * 7 + y * 3 + z * 11) % 40)
			: (uint16_t) (10 + (x + 2 * y + 5 * z) % 9);
	}

	// One tiled 16-bit page per Z plane: a plain TIFF whose pages share a shape reads as a Z-stack
	void write_ooc_seg_stack (const fs::path& f, uint16_t (*val) (uint32_t, uint32_t, uint32_t))
	{
		TIFF* t = TIFFOpen (f.string().c_str(), "w");
		ASSERT_NE(t, nullptr) << f.string();
		std::vector<uint16_t> buf (ooc_seg_tile * ooc_seg_tile);
		for (uint32_t z = 0; z < ooc_seg_D; z++)
		{
			TIFFSetField (t, TIFFTAG_IMAGEWIDTH, ooc_seg_W);
			TIFFSetField (t, TIFFTAG_IMAGELENGTH, ooc_seg_H);
			TIFFSetField (t, TIFFTAG_SAMPLESPERPIXEL, 1);
			TIFFSetField (t, TIFFTAG_BITSPERSAMPLE, 16);
			TIFFSetField (t, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_UINT);
			TIFFSetField (t, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
			TIFFSetField (t, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
			TIFFSetField (t, TIFFTAG_TILEWIDTH, ooc_seg_tile);
			TIFFSetField (t, TIFFTAG_TILELENGTH, ooc_seg_tile);
			for (uint32_t y0 = 0; y0 < ooc_seg_H; y0 += ooc_seg_tile)
				for (uint32_t x0 = 0; x0 < ooc_seg_W; x0 += ooc_seg_tile)
				{
					for (uint32_t r = 0; r < ooc_seg_tile; r++)
						for (uint32_t c = 0; c < ooc_seg_tile; c++)
							buf[r * ooc_seg_tile + c] = val (x0 + c, y0 + r, z);
					ASSERT_GE(TIFFWriteTile (t, buf.data(), x0, y0, 0, 0), 0) << f.string();
				}
			ASSERT_EQ(TIFFWriteDirectory (t), 1) << f.string();
		}
		TIFFClose (t);
	}

	struct OocSegPair
	{
		fs::path dir, inten, mask;

		OocSegPair (const char* name)
		{
			dir = fs::temp_directory_path() / name;
			std::error_code ec;
			fs::remove_all (dir, ec);
			fs::create_directories (dir);
			inten = dir / "i.tif";
			mask = dir / "m.tif";
		}
		~OocSegPair()
		{
			std::error_code ec;
			fs::remove_all (dir, ec);
		}
	};

	// An environment through phase 1 over the pair, with 'featureset' requested and every ROI's
	// values initialized: the state phase 2 and phase 3 both start from
	template <class Featureset>
	void prepare_ooc_seg_env (Environment& e, const OocSegPair& p, bool singleroi, const Featureset& featureset)
	{
		e.set_dim (3);
		e.singleROI = singleroi;
		e.theFeatureSet.enableAll (false);
		e.theFeatureSet.enableFeatures (featureset);
		ASSERT_TRUE(e.theFeatureMgr.compile());
		e.theFeatureMgr.apply_user_selection (e.theFeatureSet);
		ASSERT_TRUE(e.theFeatureMgr.init_feature_classes());
		e.compile_feature_settings();
		e.refresh_feature_settings_singleroi();
		ASSERT_TRUE(e.set_ram_limit (64));	// the fixture is kilobytes; a bigger limit is one a busy box can refuse

		SlideProps& sp = e.dataset.dataset_props.emplace_back (p.inten.string(), p.mask.string());
		ASSERT_TRUE(Nyxus::scan_slide_props (sp, 3, e.anisoOptions, e.use_physical_spacing(),
			e.fpimageOptions, e.resultOptions.need_annotation()));
		e.dataset.update_dataset_props_extrema();

		ASSERT_TRUE(Nyxus::gatherRoisMetrics_3D (e, 0, p.inten.string(), p.mask.string(), 0, 0));
		for (auto lab : e.uniqueLabels)
			e.roiData[lab].initialize_fvals();
	}

	// Every 3D intensity value of ROI 'lab', in-RAM and out-of-core, for the same pair and mode
	void expect_ooc_seg_matches_in_ram (const OocSegPair& p, bool singleroi, const std::vector<int>& labels)
	{
		Environment in_ram, ooc;
		prepare_ooc_seg_env (in_ram, p, singleroi, D3_VoxelIntensityFeatures::featureset);
		prepare_ooc_seg_env (ooc, p, singleroi, D3_VoxelIntensityFeatures::featureset);
		std::vector<int> found (in_ram.uniqueLabels.begin(), in_ram.uniqueLabels.end());
		std::sort (found.begin(), found.end());
		ASSERT_EQ(found, labels);

		ASSERT_TRUE(Nyxus::processTrivialRois_3D (in_ram, 0, 0, 0, labels, p.inten.string(), p.mask.string(),
			in_ram.get_ram_limit()));
		ASSERT_TRUE(Nyxus::processNontrivialRois_3D (ooc, labels, p.inten.string(), p.mask.string(), 0, 0));

		for (int lab : labels)
			for (auto fcode : D3_VoxelIntensityFeatures::featureset)
			{
				const double want = in_ram.roiData[lab].fvals[(int) fcode][0],
					got = ooc.roiData[lab].fvals[(int) fcode][0];
				EXPECT_NEAR(got, want, 1e-9 * (std::max) (1.0, std::abs (want)))
					<< "ROI " << lab << " feature " << (int) fcode << " differs out-of-core";
			}
	}
}

// Single-ROI mode on a two-label mask: the out-of-core ROI is the whole foreground, as the in-RAM
// one is. What this discriminates: a cloud that keeps only voxels whose mask value is literally 1
// holds the left half alone, so its maximum sits near 18 where the in-RAM one is past 500, and the
// mean and every moment move with it.
void test_3d_ooc_singleroi_collapses_labels_mechanics()
{
	OocSegPair p ("nyxus_3d_ooc_singleroi");
	write_ooc_seg_stack (p.inten, ooc_seg_inten);
	write_ooc_seg_stack (p.mask, ooc_seg_label);

	expect_ooc_seg_matches_in_ram (p, /*singleroi=*/ true, { 1 });

	// the fixture discriminates: ROI 1 in this mode reaches the label-2 intensities
	Environment e;
	prepare_ooc_seg_env (e, p, true, D3_VoxelIntensityFeatures::featureset);
	ASSERT_TRUE(Nyxus::processNontrivialRois_3D (e, { 1 }, p.inten.string(), p.mask.string(), 0, 0));
	int fmax = -1;
	ASSERT_TRUE(e.theFeatureSet.find_3D_FeatureByString ("3MAX", fmax));
	EXPECT_GE(e.roiData[1].fvals[fmax][0], 500.0) << "the single ROI's cloud lacks the label-2 half";
}

// Multi-ROI mode on the same pair: each label streams only its own voxels. What this
// discriminates: a cloud that keeps every nonzero mask voxel whatever the mode gives both ROIs the
// whole foreground, and neither matches its in-RAM values.
void test_3d_ooc_multiroi_keeps_labels_apart_mechanics()
{
	OocSegPair p ("nyxus_3d_ooc_multiroi");
	write_ooc_seg_stack (p.inten, ooc_seg_inten);
	write_ooc_seg_stack (p.mask, ooc_seg_label);

	expect_ooc_seg_matches_in_ram (p, /*singleroi=*/ false, { 1, 2 });
}

// A feature that fails out-of-core fails the pair on the CLI build. GLDZM refuses a box whose
// border distances outgrow its 16-bit buffer; the ROI's box is widened to that size after phase 1,
// which is the only state the refusal reads. What this discriminates: a pass that logs the failure
// and carries on returns true, and the ROI's row is then written with GLDZM's initialized zeros
// as if they were measured.
void test_3d_ooc_failed_feature_fails_pair_mechanics()
{
	OocSegPair p ("nyxus_3d_ooc_failed_feature");
	write_ooc_seg_stack (p.inten, ooc_seg_inten);
	write_ooc_seg_stack (p.mask, ooc_seg_label);

	Environment e;
	prepare_ooc_seg_env (e, p, false, D3_GLDZM_feature::featureset);
	const int side = 131069;	// (side + 1) / 2 == 65535, the 16-bit unsettled mark
	LR& r = e.roiData[1];
	r.aabb.init_x (0); r.aabb.update_x (side - 1);
	r.aabb.init_y (0); r.aabb.update_y (side - 1);
	r.aabb.init_z (0); r.aabb.update_z (side - 1);

	testing::internal::CaptureStderr();
	bool ok = true, threw = false;
	try
	{
		ok = Nyxus::processNontrivialRois_3D (e, { 1 }, p.inten.string(), p.mask.string(), 0, 0);
	}
	catch (...)
	{
		threw = true;
	}
	// ends the capture on every path: leaving it open would swallow other tests' failure output
	const std::string err = testing::internal::GetCapturedStderr();

	ASSERT_FALSE(threw) << "the CLI build reports the failure by status";
	EXPECT_FALSE(ok) << "a failed feature must fail the pair";
	EXPECT_NE(err.find ("border distance"), std::string::npos) << "the failure names its cause:\n" << err;
}
