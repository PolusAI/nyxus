#pragma once

#include "test_3d_morphology_common.h"
#include "test_ref_vals.h"

// MATLAB regionprops3 assertions for the two feature/config pairs it can verify on the shared
// segmented phantom. No covariance helpers or other morphology features belong in this file.
//
// Provenance (SPEC 6.4):
//   tool       = MATLAB Image Processing Toolbox regionprops3
//   version    = MATLAB R2026a
//   properties = Volume -> 3VOXEL_VOLUME
//                ConvexVolume -> 3VOLUME_CONVEXHULL
//   fixture    = tests/data/nifti/phantoms/ut_mask57.nii, label 57, native 1x1x1 spacing
//   recipe     = morphology3d.matlab_regionprops3
//   generator  = tests/vetting/oracles/gen_morphology3d_matlab.m
//
// ConvexVolume is the number of voxels in regionprops3's ConvexImage, the hull rasterised back onto
// the lattice, where 3VOLUME_CONVEXHULL is the volume of the hull of the mesh vertices. The two are
// different conventions for the same body: measured 497824 against 496958.33, 0.174% apart, held to
// the 0.5% band in test_3d_morphology_common.h.
//
// regionprops3 carries no mesh volume, so it cannot judge 3MESH_VOLUME. That feature is an integral
// of the ROI surface mesh and is asserted against MIRP's morph_volume in test_3d_morphology_mirp.h
// and against closed-form solids in test_3d_morphology_analytic.h.
static const ref_vals_map<double> morphology_3d_matlab_ref_vals
{
    { "3VOXEL_VOLUME", 274432.0 },
    { "3VOLUME_CONVEXHULL", 497824.0 }
};

static void assert_3d_morphology_feature_matlab (
    const std::string& fname,
    const Nyxus::Feature3D& expecting_fcode)
{
    SCOPED_TRACE(std::string("MATLAB_ORACLE__") + fname);
    ASSERT_TRUE(morphology_3d_matlab_ref_vals.count(fname) > 0) << fname;

    double actual = 0.0;
    calculate_3d_morphology_feature_value(fname, expecting_fcode, actual);

    const double expected = morphology_3d_matlab_ref_vals.at(fname);
    const double band = morphology_3d_volume_ref_tol_pct(fname);
    const double pct = 100.0 * std::abs(actual - expected) / std::abs(expected);
    ASSERT_LE(pct, band)
        << fname << " actual=" << actual << " MATLAB regionprops3=" << expected
        << " band=" << band << "%";
}

void test_3d_morphology_voxel_volume_matlab() {
    assert_3d_morphology_feature_matlab("3VOXEL_VOLUME", Feature3D::VOXEL_VOLUME);
}

void test_3d_morphology_volume_convex_hull_matlab() {
    assert_3d_morphology_feature_matlab("3VOLUME_CONVEXHULL", Feature3D::VOLUME_CONVEXHULL);
}
