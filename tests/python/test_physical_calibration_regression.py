"""End-to-end regression coverage for physical-unit canonicalization through the real Python
API. No prior test exercised phys_x/y/z/phys_unit output at all (not just the unit-conversion
gap) -- this closes both gaps at once: proves the columns come back, and proves a
non-micrometer-declared store is converted to the same values as an equivalent
micrometer-declared store.

Bug fixed: physX/Y/Z were reported verbatim in whatever unit a file declared (nanometer,
millimeter, ...) with no conversion, so two calibrated files using different units produced
numerically incomparable phys_x/y/z columns. Fixed by canonicalizing every axis to micrometer
at OME metadata parse time (ome_axes.h's canonicalize_to_micrometer), using each axis's own
declared unit (a Z axis in a different unit than X/Y previously mislabeled its own value under
X/Y's unit string).

Uses OME-TIFF whole-volume mode (featurize_directory(dir, dir, ...), intensity_dir==label_dir)
-- the same pattern test_ooc_mechanics.py's whole-volume tests already exercise successfully.
Each fixture is copied into its OWN isolated tmp_path subdirectory: featurize_directory matches
every entry under the given dir against file_pattern, and tests/data/ometiff/ has dozens of
other fixtures that would otherwise also match ".*".
"""
import pathlib
import shutil

import nyxus

DATA_OMETIFF = pathlib.Path(__file__).resolve().parent.parent / "data" / "ometiff"


def test_ometiff_micrometer_and_nanometer_stores_agree(tmp_path):
    cal_um = DATA_OMETIFF / "dim5_calibrated.ome.tif"
    cal_nm = DATA_OMETIFF / "dim5_calibrated_nm.ome.tif"
    assert cal_um.exists() and cal_nm.exists()

    um_dir, nm_dir = tmp_path / "um", tmp_path / "nm"
    um_dir.mkdir(); nm_dir.mkdir()
    shutil.copy(cal_um, um_dir / cal_um.name)
    shutil.copy(cal_nm, nm_dir / cal_nm.name)

    # NOTE: deliberately NOT use_physical_spacing=True -- phys_x/y/z/phys_unit are emitted
    # regardless of that flag (it only controls whether ANISOTROPIC RESAMPLING is applied),
    # and enabling it here hit an unrelated hang in the resampling path for this fixture's
    # multi-channel/timeframe + anisotropic (4x Z:XY) combination. Out of scope for this fix
    # (units, not resampling); flagging separately rather than chasing it here.
    n = nyxus.Nyxus3D(["*3D_ALL_INTENSITY*"])
    df_um = n.featurize_directory(str(um_dir), str(um_dir), ".*")
    df_nm = n.featurize_directory(str(nm_dir), str(nm_dir), ".*")

    for df in (df_um, df_nm):
        assert "phys_unit" in df.columns
        assert {"phys_x", "phys_y", "phys_z"} <= set(df.columns)

    # dim5_calibrated: physX/Y=0.5, physZ=2.0 micrometer.
    # dim5_calibrated_nm: same spacing declared in nanometer (X/Y) and millimeter (Z) -- must
    # canonicalize to the SAME values, not the raw declared numbers.
    assert (df_um["phys_x"] == 0.5).all()
    assert (df_um["phys_y"] == 0.5).all()
    assert (df_um["phys_z"] == 2.0).all()
    assert (df_um["phys_unit"] == "micrometer").all()

    assert (df_nm["phys_x"] == df_um["phys_x"]).all()
    assert (df_nm["phys_y"] == df_um["phys_y"]).all()
    assert (df_nm["phys_z"] == df_um["phys_z"]).all()
    assert (df_nm["phys_unit"] == "micrometer").all()
