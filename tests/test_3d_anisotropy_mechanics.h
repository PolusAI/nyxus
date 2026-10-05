#pragma once

// Voxel spacing on the volumetric paths: what it changes and what it leaves alone.
//
// Every volumetric path caches a ROI's voxels as acquired and hands the spacing to the shape family
// alone. So on any of them, a run with anisotropy has to report the same first-order and texture
// values as a run without it, and shape values scaled by the spacing. The paths are the segmented
// in-RAM pass, the segmented out-of-core pass, the whole-volume in-RAM and out-of-core passes, and
// the 2.5D pass over a stack of 2D files.
//
// The fixture is a 16x16x6 tiled volume with two labels of different shapes inside a background
// border: an ellipsoid and a notched block. The spacing (1.3, 0.8, 2.5) is unequal on every axis and
// inexact on each, so a pass that resampled the voxels would duplicate some and drop others, and
// every count-based value would move.

#include <algorithm>
#include <cmath>
#include <map>
#include <string>
#include <vector>
#include "test_main_nyxus.h"		// gtest, libtiff, fs, globals.h, roi_cache.h
#include "../src/nyx/environment.h"
#include "../src/nyx/features/3d_intensity.h"
#include "../src/nyx/features/3d_glcm.h"
#include "../src/nyx/features/3d_gldm.h"
#include "../src/nyx/features/3d_gldzm.h"
#include "../src/nyx/features/3d_glrlm.h"
#include "../src/nyx/features/3d_glszm.h"
#include "../src/nyx/features/3d_ngldm.h"
#include "../src/nyx/features/3d_ngtdm.h"
#include "../src/nyx/features/3d_surface.h"

namespace Nyxus
{
	bool featurize_wholevolume (Environment& env, size_t sidx, ImageLoader& imlo, LR& vroi, size_t channel, size_t timeframe);
}

namespace
{
	const uint32_t aniso_W = 16, aniso_H = 16, aniso_D = 6, aniso_tile = 16;
	const double aniso_sx = 1.3, aniso_sy = 0.8, aniso_sz = 2.5;

	// label 1: an ellipsoid; label 2: a block with one corner notched out; 0 elsewhere
	uint16_t aniso_label (uint32_t x, uint32_t y, uint32_t z)
	{
		const double ex = (x - 4.5) / 3.6, ey = (y - 7.5) / 5.2, ez = (z - 2.5) / 2.4;
		if (ex * ex + ey * ey + ez * ez <= 1.0)
			return 1;
		if (x >= 9 && x <= 14 && y >= 3 && y <= 12 && z >= 1 && z <= 4 && ! (x >= 12 && y >= 9 && z >= 3))
			return 2;
		return 0;
	}

	uint16_t aniso_inten (uint32_t x, uint32_t y, uint32_t z)
	{
		switch (aniso_label (x, y, z))
		{
		case 1:
			return (uint16_t) (100 + (x * 37 + y * 17 + z * 53) % 61);
		case 2:
			return (uint16_t) (400 + (x * 13 + y * 29 + z * 7) % 47);
		default:
			return (uint16_t) (3 + (x + 2 * y + 3 * z) % 5);
		}
	}

	// One tiled 16-bit page per Z plane in [z0, z1): a plain TIFF whose pages share a shape reads as
	// a Z-stack, and a one-page file is one plane of a 2.5D stack
	void write_aniso_pages (const fs::path& f, uint16_t (*val) (uint32_t, uint32_t, uint32_t), uint32_t z0, uint32_t z1)
	{
		TIFF* t = TIFFOpen (f.string().c_str(), "w");
		ASSERT_NE(t, nullptr) << f.string();
		std::vector<uint16_t> buf (aniso_tile * aniso_tile);
		for (uint32_t z = z0; z < z1; z++)
		{
			TIFFSetField (t, TIFFTAG_IMAGEWIDTH, aniso_W);
			TIFFSetField (t, TIFFTAG_IMAGELENGTH, aniso_H);
			TIFFSetField (t, TIFFTAG_SAMPLESPERPIXEL, 1);
			TIFFSetField (t, TIFFTAG_BITSPERSAMPLE, 16);
			TIFFSetField (t, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_UINT);
			TIFFSetField (t, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
			TIFFSetField (t, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
			TIFFSetField (t, TIFFTAG_TILEWIDTH, aniso_tile);
			TIFFSetField (t, TIFFTAG_TILELENGTH, aniso_tile);
			for (uint32_t r = 0; r < aniso_tile; r++)
				for (uint32_t c = 0; c < aniso_tile; c++)
					buf[r * aniso_tile + c] = val (c, r, z);
			ASSERT_GE(TIFFWriteTile (t, buf.data(), 0, 0, 0, 0), 0) << f.string();
			ASSERT_EQ(TIFFWriteDirectory (t), 1) << f.string();
		}
		TIFFClose (t);
	}

	// A Z-stack pair, and the same volume as a 2.5D stack of one-plane pairs i_<z>.tif / m_<z>.tif
	struct AnisoFixture
	{
		fs::path dir, inten, mask;
		std::vector<std::string> z_indices;

		AnisoFixture (const char* name)
		{
			dir = fs::temp_directory_path() / name;
			std::error_code ec;
			fs::remove_all (dir, ec);
			fs::create_directories (dir);
			inten = dir / "i.tif";
			mask = dir / "m.tif";
			write_aniso_pages (inten, aniso_inten, 0, aniso_D);
			write_aniso_pages (mask, aniso_label, 0, aniso_D);
			for (uint32_t z = 0; z < aniso_D; z++)
			{
				z_indices.push_back (std::to_string (z));
				write_aniso_pages (dir / ("i_" + z_indices.back() + ".tif"), aniso_inten, z, z + 1);
				write_aniso_pages (dir / ("m_" + z_indices.back() + ".tif"), aniso_label, z, z + 1);
			}
		}
		~AnisoFixture()
		{
			std::error_code ec;
			fs::remove_all (dir, ec);
		}
		std::string inten_25d() const { return (dir / "i_*.tif").string(); }
		std::string mask_25d() const { return (dir / "m_*.tif").string(); }
	};

	// Every 3D family but shape: the ones a voxel's spacing must not reach
	const std::initializer_list<std::initializer_list<Nyxus::Feature3D>> aniso_invariant_families =
	{
		D3_VoxelIntensityFeatures::featureset, D3_GLCM_feature::featureset, D3_GLDM_feature::featureset,
		D3_GLDZM_feature::featureset, D3_GLRLM_feature::featureset, D3_GLSZM_feature::featureset,
		D3_NGLDM_feature::featureset, D3_NGTDM_feature::featureset
	};

	std::vector<Nyxus::Feature3D> aniso_invariant_features()
	{
		std::vector<Nyxus::Feature3D> v;
		for (auto fs : aniso_invariant_families)
			v.insert (v.end(), fs.begin(), fs.end());
		return v;
	}

	// An environment with every 3D family requested and the fixture's spacing given as --aniso*
	// when 'anisotropic', prescanned over slide 'ipath'/'mpath' ('mpath' empty for a whole volume)
	void prepare_aniso_env (Environment& e, bool anisotropic, const std::string& ipath, const std::string& mpath)
	{
		e.set_dim (3);
		e.singleROI = false;
		e.theFeatureSet.enableAll (false);
		for (auto fs : aniso_invariant_families)
			e.theFeatureSet.enableFeatures (fs);
		e.theFeatureSet.enableFeatures (D3_SurfaceFeature::featureset);
		ASSERT_TRUE(e.theFeatureMgr.compile());
		e.theFeatureMgr.apply_user_selection (e.theFeatureSet);
		ASSERT_TRUE(e.theFeatureMgr.init_feature_classes());
		e.compile_feature_settings();
		e.refresh_feature_settings_singleroi();
		ASSERT_TRUE(e.set_ram_limit (64));	// the fixture is kilobytes; a bigger limit is one a busy box can refuse
		if (anisotropic)
		{
			e.anisoOptions.set_aniso_x (aniso_sx);
			e.anisoOptions.set_aniso_y (aniso_sy);
			e.anisoOptions.set_aniso_z (aniso_sz);
		}

		SlideProps& sp = e.dataset.dataset_props.emplace_back (ipath, mpath);
		ASSERT_TRUE(Nyxus::scan_slide_props (sp, 3, e.anisoOptions, e.use_physical_spacing(),
			e.fpimageOptions, e.resultOptions.need_annotation()));
		e.dataset.update_dataset_props_extrema();
	}

	// Values of one ROI by feature code, every sub-value of each
	using AnisoValues = std::vector<std::vector<double>>;

	enum class AnisoPath { in_ram, out_of_core };

	// Segmented pass over the Z-stack pair: label -> values
	std::map<int, AnisoValues> run_aniso_segmented (const AnisoFixture& f, bool anisotropic, AnisoPath path)
	{
		std::map<int, AnisoValues> out;
		Environment e;
		prepare_aniso_env (e, anisotropic, f.inten.string(), f.mask.string());
		EXPECT_TRUE(Nyxus::gatherRoisMetrics_3D (e, 0, f.inten.string(), f.mask.string(), 0, 0));
		std::vector<int> labels (e.uniqueLabels.begin(), e.uniqueLabels.end());
		std::sort (labels.begin(), labels.end());
		EXPECT_EQ(labels, std::vector<int>({ 1, 2 }));
		for (auto lab : labels)
			e.roiData[lab].initialize_fvals();

		bool ok = path == AnisoPath::in_ram
			? Nyxus::processTrivialRois_3D (e, 0, 0, 0, labels, f.inten.string(), f.mask.string(), e.get_ram_limit())
			: Nyxus::processNontrivialRois_3D (e, labels, f.inten.string(), f.mask.string(), 0, 0);
		EXPECT_TRUE(ok);
		for (auto lab : labels)
			out[lab] = e.roiData[lab].fvals;
		return out;
	}

	// Whole-volume pass over the intensity stack; the RAM limit decides in-RAM or out-of-core
	AnisoValues run_aniso_wholevolume (const AnisoFixture& f, bool anisotropic, AnisoPath path)
	{
		Environment e;
		prepare_aniso_env (e, anisotropic, f.inten.string(), "");
		if (path == AnisoPath::out_of_core)
			EXPECT_TRUE(e.set_ram_limit (0));	// every footprint is at or above it

		ImageLoader imlo;
		EXPECT_TRUE(imlo.open (e.dataset.dataset_props[0], e.fpimageOptions));
		LR vroi (1);
		EXPECT_TRUE(Nyxus::featurize_wholevolume (e, 0, imlo, vroi, 0, 0));
		imlo.close();
		return vroi.fvals;
	}

	// 2.5D pass over the stack of one-plane pairs. 'physical' gives the slide a physical voxel size
	// in the fixture's proportions and turns --use-physical-spacing on instead of passing --aniso*.
	std::map<int, AnisoValues> run_aniso_25d (const AnisoFixture& f, bool anisotropic, bool physical = false)
	{
		std::map<int, AnisoValues> out;
		Environment e;
		prepare_aniso_env (e, anisotropic && ! physical, (f.dir / "i_0.tif").string(), (f.dir / "m_0.tif").string());
		if (anisotropic && physical)
		{
			e.use_physical_spacing_ = true;
			SlideProps& p = e.dataset.dataset_props[0];
			p.phys_x = aniso_sx * 0.4;	// the fixture's proportions in some physical unit, which
			p.phys_y = aniso_sy * 0.4;	// the ratio normalization to the smallest axis (y) cancels
			p.phys_z = aniso_sz * 0.4;
		}
		EXPECT_TRUE(Nyxus::gatherRoisMetrics_25D (e, 0, f.inten_25d(), f.mask_25d(), f.z_indices));
		std::vector<int> labels (e.uniqueLabels.begin(), e.uniqueLabels.end());
		std::sort (labels.begin(), labels.end());
		EXPECT_EQ(labels, std::vector<int>({ 1, 2 }));
		for (auto lab : labels)
			e.roiData[lab].initialize_fvals();

		EXPECT_TRUE(Nyxus::processTrivialRois_25D (e, labels, f.inten_25d(), f.mask_25d(), e.get_ram_limit(), f.z_indices));
		for (auto lab : labels)
			out[lab] = e.roiData[lab].fvals;
		return out;
	}

	bool same_value (double a, double b)
	{
		return (std::isnan (a) && std::isnan (b)) || a == b;
	}

	// The first-order and texture values of a run with anisotropy are those of the run without it
	void expect_aniso_invariant (const AnisoValues& iso, const AnisoValues& aniso, const std::string& what)
	{
		for (auto fcode : aniso_invariant_features())
		{
			const auto& want = iso[(int) fcode];
			const auto& got = aniso[(int) fcode];
			ASSERT_EQ(got.size(), want.size()) << what << " feature " << (int) fcode;
			for (size_t i = 0; i < want.size(); i++)
				EXPECT_TRUE(same_value (got[i], want[i]))
					<< what << " feature " << (int) fcode << "[" << i << "]: " << got[i] << " with anisotropy, "
					<< want[i] << " without";
		}
	}

	// The voxel volume is the voxel count times a voxel's volume, and the mesh and its convex hull are
	// built on the lattice, whose linear image a grid of sx*sy*sz voxels is: the spacing scales all
	// three by sx*sy*sz exactly. A volume that never saw the spacing stays where it was.
	void expect_aniso_scales_shape (const AnisoValues& iso, const AnisoValues& aniso, const std::string& what,
		double voxel_volume = aniso_sx * aniso_sy * aniso_sz)
	{
		const std::pair<Nyxus::Feature3D, const char*> volumes[] =
		{
			{ Nyxus::Feature3D::VOXEL_VOLUME, "3VOXEL_VOLUME" },
			{ Nyxus::Feature3D::MESH_VOLUME, "3MESH_VOLUME" },
			{ Nyxus::Feature3D::VOLUME_CONVEXHULL, "3VOLUME_CONVEXHULL" }
		};
		for (const auto& [fcode, name] : volumes)
		{
			const int f = (int) fcode;
			ASSERT_GT(iso[f][0], 0.0) << what << ": " << name << " without anisotropy";
			const double want = iso[f][0] * voxel_volume;
			EXPECT_NEAR(aniso[f][0], want, 1e-12 * want) << what << ": " << name << " does not carry the spacing";
		}
	}
}

// Segmented in-RAM pass. What this discriminates: a scan that resamples the volume by the spacing
// caches some voxels twice and others not at all, and every first-order and texture value moves.
void test_3d_anisotropy_in_ram_leaves_intensity_and_texture_alone_mechanics()
{
	AnisoFixture f ("nyxus_3d_aniso_in_ram");
	auto iso = run_aniso_segmented (f, false, AnisoPath::in_ram),
		aniso = run_aniso_segmented (f, true, AnisoPath::in_ram);
	for (int lab : { 1, 2 })
	{
		expect_aniso_invariant (iso[lab], aniso[lab], "in-RAM ROI " + std::to_string (lab));
		expect_aniso_scales_shape (iso[lab], aniso[lab], "in-RAM ROI " + std::to_string (lab));
	}
}

// Segmented out-of-core pass, which streams its own cloud and so can resample on its own
void test_3d_anisotropy_out_of_core_leaves_intensity_and_texture_alone_mechanics()
{
	AnisoFixture f ("nyxus_3d_aniso_ooc");
	auto iso = run_aniso_segmented (f, false, AnisoPath::out_of_core),
		aniso = run_aniso_segmented (f, true, AnisoPath::out_of_core);
	for (int lab : { 1, 2 })
	{
		expect_aniso_invariant (iso[lab], aniso[lab], "out-of-core ROI " + std::to_string (lab));
		expect_aniso_scales_shape (iso[lab], aniso[lab], "out-of-core ROI " + std::to_string (lab));
	}
}

// Whole-volume in-RAM and out-of-core passes, which take their spacing and their box from the
// prescan rather than from phase 1
void test_3d_anisotropy_whole_volume_leaves_intensity_and_texture_alone_mechanics()
{
	AnisoFixture f ("nyxus_3d_aniso_wholevolume");
	for (auto path : { AnisoPath::in_ram, AnisoPath::out_of_core })
	{
		const std::string what = path == AnisoPath::in_ram ? "whole volume in-RAM" : "whole volume out-of-core";
		auto iso = run_aniso_wholevolume (f, false, path),
			aniso = run_aniso_wholevolume (f, true, path);
		expect_aniso_invariant (iso, aniso, what);
		expect_aniso_scales_shape (iso, aniso, what);
	}
}

// The 2.5D pass over a stack of 2D files
void test_3d_anisotropy_25d_leaves_intensity_and_texture_alone_mechanics()
{
	AnisoFixture f ("nyxus_3d_aniso_25d");
	auto iso = run_aniso_25d (f, false),
		aniso = run_aniso_25d (f, true);
	for (int lab : { 1, 2 })
	{
		expect_aniso_invariant (iso[lab], aniso[lab], "2.5D ROI " + std::to_string (lab));
		expect_aniso_scales_shape (iso[lab], aniso[lab], "2.5D ROI " + std::to_string (lab));
	}
}

// --use-physical-spacing on a 2.5D run: the slide's physical voxel size reaches the shape family as
// it does on the 3D paths, normalized so the smallest axis (y) is 1. What this discriminates: a 2.5D
// pass that reads only --aniso* reports the shape of cubic voxels, so the voxel volume stays at the
// isotropic run's.
void test_3d_anisotropy_25d_takes_physical_spacing_mechanics()
{
	AnisoFixture f ("nyxus_3d_aniso_25d_physical");
	auto iso = run_aniso_25d (f, false),
		physical = run_aniso_25d (f, true, /*physical=*/ true);
	for (int lab : { 1, 2 })
	{
		expect_aniso_invariant (iso[lab], physical[lab], "2.5D physical-spacing ROI " + std::to_string (lab));
		expect_aniso_scales_shape (iso[lab], physical[lab], "2.5D physical-spacing ROI " + std::to_string (lab),
			(aniso_sx / aniso_sy) * (aniso_sz / aniso_sy));
	}
}

// A ROI one voxel thick along an axis whose spacing is below 1. Resampling by 0.4 maps a one-plane
// ROI to no virtual plane at all, so a resampling pass either has nothing to featurize or reads
// past an empty cloud; on the voxels as acquired it is an ordinary ROI. What this discriminates: any
// refusal or emptied cloud leaves the anisotropic run without the isotropic run's values.
void test_3d_anisotropy_thin_roi_is_featurized_mechanics()
{
	struct Thin
	{
		static uint16_t label (uint32_t x, uint32_t y, uint32_t z) { return (z == 2 && x >= 3 && x <= 11 && y >= 4 && y <= 9) ? 1 : 0; }
		static uint16_t inten (uint32_t x, uint32_t y, uint32_t z) { return (uint16_t) (50 + (x * 11 + y * 5 + z) % 23); }
	};
	const fs::path dir = fs::temp_directory_path() / "nyxus_3d_aniso_thin";
	std::error_code ec;
	fs::remove_all (dir, ec);
	fs::create_directories (dir);
	const fs::path ip = dir / "i.tif", mp = dir / "m.tif";
	write_aniso_pages (ip, Thin::inten, 0, aniso_D);
	write_aniso_pages (mp, Thin::label, 0, aniso_D);

	AnisoValues vals[2];
	for (int k = 0; k < 2; k++)
	{
		Environment e;
		prepare_aniso_env (e, false, ip.string(), mp.string());
		if (k == 1)
			e.anisoOptions.set_aniso_z (0.4);
		ASSERT_TRUE(Nyxus::gatherRoisMetrics_3D (e, 0, ip.string(), mp.string(), 0, 0));
		ASSERT_EQ(e.uniqueLabels.size(), 1u);
		e.roiData[1].initialize_fvals();
		ASSERT_TRUE(Nyxus::processTrivialRois_3D (e, 0, 0, 0, { 1 }, ip.string(), mp.string(), e.get_ram_limit()));
		vals[k] = e.roiData[1].fvals;
	}
	fs::remove_all (dir, ec);

	expect_aniso_invariant (vals[0], vals[1], "one-plane ROI at z spacing 0.4");
	const int f = (int) Nyxus::Feature3D::VOXEL_VOLUME;
	EXPECT_NEAR(vals[1][f][0], 0.4 * vals[0][f][0], 1e-12 * vals[0][f][0]);
}
