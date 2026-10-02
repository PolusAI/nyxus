#pragma once

// 3D shape on a non-cubic voxel grid against pyradiomics 3.0.1 (RadiomicsShape), which builds its
// marching-cubes mesh and its covariance on the voxels as acquired, scaled by the image spacing.
// 3AREA and the ratios built on it are not pinned here: nyxus' 3AREA is the exposed-voxel-face
// count, which pyradiomics does not compute.
//
// Fixture: the segmented phantom (ut_inten.nii / ut_mask57.nii, ROI 57) at voxel spacing
// (1.3, 0.8, 2.5) in x, y, z, given to nyxus as --aniso* and to pyradiomics as the SimpleITK image
// spacing in the same axis order. The spacing is unequal on every axis, so a value that took any
// axis's spacing from another, or left one out, misses its pin.
//
// Goldens: pyradiomics 3.0.1 + SimpleITK 2.3.1, shape.RadiomicsShape(image, mask, label=57) with
// image.SetSpacing((1.3, 0.8, 2.5)) on both images. At unit spacing the same call gives MeshVolume
// 274338.333..., and the pin below is 1.3*0.8*2.5 = 2.6 times that, as a linear image of the mesh is.

#include "test_3d_morphology_common.h"

namespace
{
	// The phantom's ROI 57 through phase 1 and the in-RAM scan at the fixture's spacing, then the
	// shape family
	void calculate_3d_morphology_aniso_value (const std::string& fname, const Nyxus::Feature3D& expecting_fcode, double& out)
	{
		auto [ipath, mpath, label] = get_3d_segmented_phantom();
		ASSERT_TRUE(fs::exists(ipath));
		ASSERT_TRUE(fs::exists(mpath));

		Environment e;
		e.anisoOptions.set_aniso_x (1.3);
		e.anisoOptions.set_aniso_y (0.8);
		e.anisoOptions.set_aniso_z (2.5);
		e.dataset.dataset_props.reserve(1);
		SlideProps& sp = e.dataset.dataset_props.emplace_back(ipath, mpath);
		ASSERT_TRUE(scan_slide_props(sp, 3, e.anisoOptions, e.use_physical_spacing(), e.fpimageOptions, e.resultOptions.need_annotation()));
		e.dataset.update_dataset_props_extrema();
		clear_slide_rois(e.uniqueLabels, e.roiData);
		ASSERT_TRUE(gatherRoisMetrics_3D(e, 0, ipath, mpath, 0, 0));
		std::vector<int> batch = { label };
		ASSERT_TRUE(scanTrivialRois_3D(e, batch, ipath, mpath, 0, 0));
		ASSERT_NO_THROW(allocateTrivialRoisBuffers_3D(batch, e.roiData, e.hostCache));

		Fsettings s;
		s.resize((int)NyxSetting::__COUNT__);
		s[(int)NyxSetting::SOFTNAN].rval = 0.0;
		s[(int)NyxSetting::TINY].rval = 0.0;
		s[(int)NyxSetting::SINGLEROI].bval = false;
		s[(int)NyxSetting::GREYDEPTH].ival = 128;
		s[(int)NyxSetting::PIXELSIZEUM].rval = 100;
		s[(int)NyxSetting::PIXELDISTANCE].ival = 5;
		s[(int)NyxSetting::USEGPU].bval = false;
		s[(int)NyxSetting::VERBOSLVL].ival = 0;
		s[(int)NyxSetting::IBSI].bval = true;

		int fcode = -1;
		ASSERT_TRUE(e.theFeatureSet.find_3D_FeatureByString(fname, fcode));
		ASSERT_TRUE((int)expecting_fcode == fcode);

		LR& r = e.roiData[label];
		ASSERT_NO_THROW(r.initialize_fvals());
		D3_SurfaceFeature f;
		ASSERT_NO_THROW(f.calculate(r, s));
		f.save_value(r.fvals);
		out = r.fvals[fcode][0];
	}

	void expect_3d_morphology_aniso_pyradiomics (const std::string& fname, Nyxus::Feature3D fcode, double golden, double rel)
	{
		double v = 0;
		calculate_3d_morphology_aniso_value (fname, fcode, v);
		EXPECT_NEAR(v, golden, rel * std::abs (golden)) << fname << " at spacing (1.3, 0.8, 2.5)";
	}
}

// The mesh volume is the lattice mesh volume times a voxel's volume, exact up to summation order
void test_3d_morphology_aniso_mesh_volume_pyradiomics()
{
	expect_3d_morphology_aniso_pyradiomics ("3MESH_VOLUME", Nyxus::Feature3D::MESH_VOLUME, 713279.66666661727, 1e-9);
}

// The axis lengths are 4*sqrt of the covariance eigenvalues of the physical voxel coordinates.
// pyradiomics normalizes that covariance by the voxel count N, nyxus (as MIRP) by N-1, so its
// lengths are pyradiomics' times sqrt(N/(N-1)), N = 274432 voxels in ROI 57. The ratios
// 3ELONGATION and 3FLATNESS cancel it.
static const double aniso_axis_len_sample_cov = std::sqrt (274432. / 274431.);

void test_3d_morphology_aniso_major_axis_len_pyradiomics()
{
	expect_3d_morphology_aniso_pyradiomics ("3MAJOR_AXIS_LEN", Nyxus::Feature3D::MAJOR_AXIS_LEN, 178.78592361596122 * aniso_axis_len_sample_cov, 1e-9);
}

void test_3d_morphology_aniso_minor_axis_len_pyradiomics()
{
	expect_3d_morphology_aniso_pyradiomics ("3MINOR_AXIS_LEN", Nyxus::Feature3D::MINOR_AXIS_LEN, 114.79168868452528 * aniso_axis_len_sample_cov, 1e-9);
}

void test_3d_morphology_aniso_least_axis_len_pyradiomics()
{
	expect_3d_morphology_aniso_pyradiomics ("3LEAST_AXIS_LEN", Nyxus::Feature3D::LEAST_AXIS_LEN, 83.765297555890839 * aniso_axis_len_sample_cov, 1e-9);
}

void test_3d_morphology_aniso_elongation_pyradiomics()
{
	expect_3d_morphology_aniso_pyradiomics ("3ELONGATION", Nyxus::Feature3D::ELONGATION, 0.64206222930112822, 1e-9);
}

void test_3d_morphology_aniso_flatness_pyradiomics()
{
	expect_3d_morphology_aniso_pyradiomics ("3FLATNESS", Nyxus::Feature3D::FLATNESS, 0.46852288961977678, 1e-9);
}
