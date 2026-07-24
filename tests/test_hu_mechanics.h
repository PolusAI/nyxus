#pragma once

#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include "../src/nyx/cli_fpimage_options.h"
#include "../src/nyx/grayscale_tiff.h"
#include "../src/nyx/helpers/fsystem.h"

// ---------------------------------------------------------------------------
// Hounsfield-Unit (HU) — MECHANICS tests (SPEC.md §2: plumbing, no correctness
// claim). Covers two plumbing surfaces:
//   1) --preserve-hu option parsing into the preserve_hu() flag;
//   2) the TIFF/DICOM loader I/O path applying the HU offset mapping correctly.
// The numeric feature behavior these switch on is vetted in test_hu_analytic.h
// (analytic) and test_hu_ct_small_pydicom.py (pydicom oracle).
// ---------------------------------------------------------------------------

// FpImageOptions parses the --preserve-hu raw string into the preserve_hu() flag
// via Nyxus::parse_as_bool (accepts TRUE/FALSE/T/F, case-insensitive).
// Default is off; empty leaves it off; a non-boolean is a parse error.
void test_hu_fpimage_options_parse()
{
    FpImageOptions off;
    ASSERT_TRUE(off.parse_input());
    EXPECT_FALSE(off.preserve_hu());             // absent -> default off

    FpImageOptions on;
    on.raw_preserve_hu = "true";
    ASSERT_TRUE(on.parse_input());
    EXPECT_TRUE(on.preserve_hu());

    FpImageOptions onF;
    onF.raw_preserve_hu = "False";               // case-insensitive
    ASSERT_TRUE(onF.parse_input());
    EXPECT_FALSE(onF.preserve_hu());

    FpImageOptions bad;
    bad.raw_preserve_hu = "banana";
    EXPECT_FALSE(bad.parse_input());             // non-boolean -> parse error
}

// ---------------------------------------------------------------------------
// HU loader I/O tests against real tiled TIFF/DICOM fixtures.
//
// Fixtures (committed under tests/data/hounsfield) are 16x16, single tile, with
// pixel(r,c) = -1024 + idx*8, idx = r*16 + c (idx 0..255), so values span
// -1024..1016 — a CT/HU-like signed range crossing 0 (water).
//
//   ct_int16.tif : signed int16 (SampleFormat=2) -> loadTile<int16_t>
//   ct_float.tif : float32      (SampleFormat=3) -> loadTile_real_intens<float>
//
// In HU mode (preserve_hu=true, fpmin=-1024) the offset map yields
//   stored = value - floor(-1024) = value + 1024 = idx*8,
// so the feature-domain pixel at idx equals idx*8, with NO wraparound for the
// negative int16 values — the core of the fix. This exercises the actual TIFF
// decode + dispatch, not just the arithmetic primitive.
// ---------------------------------------------------------------------------

static inline fs::path hu_data_path(const char* name)
{
    fs::path p(__FILE__);
    return p.parent_path() / "data" / "hounsfield" / name;
}

// Load tile 0 of a fixture with the given HU/fp settings and return the buffer.
static std::vector<uint32_t> hu_load_tile0(const char* fixture, bool preserve_hu,
                                           double fpmin, double fpmax, double dr)
{
    fs::path ds = hu_data_path(fixture);
    EXPECT_TRUE(fs::exists(ds)) << "missing fixture: " << ds.string();

    NyxusGrayscaleTiffTileLoader<uint32_t> ldr(
        1, ds.string(), /*permit_fp*/ true, fpmin, fpmax, dr, preserve_hu);

    size_t th = ldr.tileHeight(0), tw = ldr.tileWidth(0);
    auto tile = std::make_shared<std::vector<uint32_t>>(th * tw, 0u);
    EXPECT_NO_THROW(ldr.loadTileFromFile(tile, 0, 0, 0, 0/*channel*/, 0/*timeframe*/, 0));
    return *tile;
}

// HU mode on signed int16 CT: value+1024 == idx*8, negatives do NOT wrap.
void test_hu_loader_int16_preserve()
{
    auto buf = hu_load_tile0("ct_int16.tif", /*preserve_hu*/ true, -1024.0, 1016.0, 10000.0);
    const size_t tw = 16;
    EXPECT_EQ(buf[0 * tw + 0], 0u);        // value -1024 (air) -> 0  (was wrapping to ~2^32)
    EXPECT_EQ(buf[0 * tw + 1], 8u);        // idx 1
    EXPECT_EQ(buf[8 * tw + 0], 1024u);     // idx 128, value 0 (water) -> 1024
    EXPECT_EQ(buf[15 * tw + 15], 2040u);   // idx 255, value 1016 (bone) -> 2040
}

// HU mode on float32 CT: same offset mapping through the real-intensity path.
void test_hu_loader_float_preserve()
{
    auto buf = hu_load_tile0("ct_float.tif", /*preserve_hu*/ true, -1024.0, 1016.0, 10000.0);
    const size_t tw = 16;
    EXPECT_EQ(buf[0 * tw + 0], 0u);
    EXPECT_EQ(buf[0 * tw + 1], 8u);
    EXPECT_EQ(buf[8 * tw + 0], 1024u);
    EXPECT_EQ(buf[15 * tw + 15], 2040u);
}

// Non-HU baseline on the float fixture: min-max rescale into [0, DR] unchanged.
// buf = DR*(value+1024)/2040 = idx*8*10000/2040; exact at chosen indices.
void test_hu_loader_float_nonpreserve_baseline()
{
    auto buf = hu_load_tile0("ct_float.tif", /*preserve_hu*/ false, -1024.0, 1016.0, 10000.0);
    const size_t tw = 16;
    EXPECT_EQ(buf[0 * tw + 0], 0u);        // min -> 0
    EXPECT_EQ(buf[3 * tw + 3], 2000u);     // idx 51 -> 51*80000/2040 = 2000
    EXPECT_EQ(buf[15 * tw + 15], 10000u);  // max -> DR
}

#ifdef DICOM_SUPPORT
#include "../src/nyx/nyxus_dicom_loader.h"

// Load tile 0 of a CT DICOM fixture through the feature-path loader (which applies
// RescaleSlope/Intercept then the HU offset). fpmin is the scanned HU-domain min.
static std::vector<uint32_t> hu_load_dicom_tile0(const char* fixture, bool preserve_hu, double fpmin)
{
    fs::path ds = hu_data_path(fixture);
    EXPECT_TRUE(fs::exists(ds)) << "missing fixture: " << ds.string();
    NyxusGrayscaleDicomLoader<uint32_t> ldr(1, ds.string(), fpmin, preserve_hu);
    size_t th = ldr.tileHeight(0), tw = ldr.tileWidth(0);
    auto tile = std::make_shared<std::vector<uint32_t>>(th * tw, 0u);
    EXPECT_NO_THROW(ldr.loadTileFromFile(tile, 0, 0, 0, 0/*channel*/, 0/*timeframe*/, 0));
    return *tile;
}

// Textbook CT: unsigned uint16 stored with RescaleIntercept=-1024. The loader must
// rescale (HU = stored - 1024) THEN offset by floor(-1024), giving idx*8.
void test_hu_loader_dicom_u16_preserve()
{
    auto buf = hu_load_dicom_tile0("ct_u16.dcm", /*preserve_hu*/ true, -1024.0);
    const size_t tw = 16;
    EXPECT_EQ(buf[0 * tw + 0], 0u);        // HU -1024 (air) -> 0
    EXPECT_EQ(buf[0 * tw + 1], 8u);
    EXPECT_EQ(buf[8 * tw + 0], 1024u);     // HU 0 (water) -> 1024
    EXPECT_EQ(buf[15 * tw + 15], 2040u);   // HU 1016 (bone) -> 2040
}

// Signed int16 stored (intercept 0): negative stored values must NOT wrap.
void test_hu_loader_dicom_i16_preserve()
{
    auto buf = hu_load_dicom_tile0("ct_i16.dcm", /*preserve_hu*/ true, -1024.0);
    const size_t tw = 16;
    EXPECT_EQ(buf[0 * tw + 0], 0u);        // stored -1024 -> 0 (no wraparound)
    EXPECT_EQ(buf[8 * tw + 0], 1024u);
    EXPECT_EQ(buf[15 * tw + 15], 2040u);
}

// Real scanner CT slice (pydicom's CT_small.dcm): 128x128 signed int16,
// RescaleSlope=1, RescaleIntercept=-1024, HU range -896..1167. With the scanned
// HU min (-896) as the offset base, the loader yields HU - floor(-896) = HU + 896.
// Reference values computed independently with pydicom (see data/hounsfield/README.md).
void test_hu_loader_dicom_ct_small_preserve()
{
    auto buf = hu_load_dicom_tile0("ct_small.dcm", /*preserve_hu*/ true, -896.0);
    const size_t tw = 128;
    EXPECT_EQ(buf[0   * tw + 0  ], 47u);     // stored 175  -> HU -849 -> 47
    EXPECT_EQ(buf[64  * tw + 64 ], 1800u);   // stored 1928 -> HU  904 -> 1800
    EXPECT_EQ(buf[100 * tw + 100], 889u);    // stored 1017 -> HU   -7 -> 889
    EXPECT_EQ(buf[127 * tw + 127], 781u);    // stored 909  -> HU -115 -> 781
}

// Same real slice WITHOUT HU mode: raw stored values, no rescale/offset (baseline).
void test_hu_loader_dicom_ct_small_baseline()
{
    auto buf = hu_load_dicom_tile0("ct_small.dcm", /*preserve_hu*/ false, 0.0);
    const size_t tw = 128;
    EXPECT_EQ(buf[0   * tw + 0  ], 175u);    // raw stored (positive int16, no wrap)
    EXPECT_EQ(buf[64  * tw + 64 ], 1928u);
}
#endif // DICOM_SUPPORT
