"""OFFLINE PyRadiomics oracle for the 2D and 3D GLCM features on ROIs whose grey levels leave bins empty.

    python tests/vetting/oracles/gen_glcm_gap_levels_pyradiomics.py     (from the repository root)

Prints the paste-ready goldens of glcm_2d_pyradiomics_gap_levels_ref_vals (test_2d_glcm_pyradiomics.h)
and glcm_3d_pyradiomics_gap_levels_ref_vals (test_3d_glcm_pyradiomics.h), and verifies both tables in
both directions: every pin reproduces, and every feature the mapping covers is pinned. Exits non-zero
on any mismatch.

Recipe `glcm.pyradiomics_gap_levels_bincount20`: PyRadiomics binCount=20, symmetricalGLCM=True,
distances=[1], weightingNorm=None, no resampling, label 1 over the whole crop. 2D runs the crop as a
one-slice volume with force2D=True, so the matrix is built over the 4 in-plane directions; 3D runs all
13 directions. On the Nyxus side that is GLCM_GREYDEPTH=-20 (negative: PyRadiomics binCount binning),
IBSI=false, GLCM_OFFSET=1.

The fixtures are the two crops the test headers hold as literal arrays (glcm_2d_gap_levels_pixels,
glcm_3d_gap_levels_voxels); this script reads them from the headers, so the arrays live in one place.
Both are boxes cut from inside one segmented structure of a contrast CT volume, Hounsfield units shifted
so the box minimum is 1. At binCount 20 their voxels leave interior bins empty and put a few voxels in
the top bin, far from the rest, which is what such ROIs look like in CT. That is the configuration in
which a matrix row index and the grey level it stands for differ: under PyRadiomics binning Nyxus keeps
only the levels present, so row x stands for level I[x]. The fixture would let an implementation that
uses x where it means I[x] land far from these goldens (on the 3D crop: DIFVAR, IV, SUMAVERAGE, DIFAVE).

Nyxus keeps several names for one PyRadiomics quantity (ENERGY is JointEnergy, HOM1 is Id, HOM2 is Idm,
VARIANCE and JVAR are SumSquares, DIS is DifferenceAverage, SUMVARIANCE is ClusterTendency); each name is
pinned to the same PyRadiomics value. GLCM_HOM2 / 3GLCM_HOM2 have no _AVE twin in the feature set.

Provenance: tool=pyradiomics 3.0.1 (SimpleITK 2.3.1, Python 3.8); env=nyxus_oracle (conda, needs
Python <= 3.9). Run offline; CI never invokes it.
"""
import os
import re
import sys

import numpy as np
import SimpleITK as sitk
import radiomics
from radiomics import glcm

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
from gen_glcm_pyradiomics import PYRADIOMICS_TO_NYXUS  # noqa: E402  the one name map, shared

HEADER_2D = os.path.join(TESTS, "test_2d_glcm_pyradiomics.h")
HEADER_3D = os.path.join(TESTS, "test_3d_glcm_pyradiomics.h")
SHAPE_2D = (1, 12, 12)       # (z, y, x)
SHAPE_3D = (5, 9, 9)
TOL = 1e-12   # pins are PyRadiomics' own output at %.17g; the bands against Nyxus live in the headers

SETTINGS = dict(binCount=20, symmetricalGLCM=True, distances=[1], weightingNorm=None, label=1,
                interpolator=None, resampledPixelSpacing=None)


def read_array(header, name, shape):
    """-> the literal array `name` in `header`, as an int array of `shape`."""
    text = re.sub(r"//[^\n]*", "", open(header, encoding="utf-8").read())
    body = text.split(name, 1)[1].split("{", 1)[1].split("};", 1)[0]
    vals = [int(v) for v in re.findall(r"\d+", body)]
    assert len(vals) == int(np.prod(shape)), f"{name}: {len(vals)} values, expected {np.prod(shape)}"
    return np.array(vals, dtype=np.int64).reshape(shape)


def run(arr, force2d):
    image = sitk.GetImageFromArray(arr.astype(np.int32))
    mask = sitk.GetImageFromArray(np.ones(arr.shape, np.int32))
    for im in (image, mask):
        im.SetSpacing((1.0, 1.0, 1.0))
    f = glcm.RadiomicsGLCM(image, mask, force2D=force2d, force2Ddimension=0, **SETTINGS)
    f.enableAllFeatures()
    values = {k: float(v) for k, v in f.execute().items()}
    levels = [int(x) for x in f.coefficients["grayLevels"]]
    return values, levels, int(f.P_glcm.shape[3])


def table(values, prefix):
    """-> {Nyxus feature name: PyRadiomics value}, every name the mapping covers, base and _AVE."""
    out = {}
    for oracle_name, features in PYRADIOMICS_TO_NYXUS.items():
        for feature in features:
            name = prefix + feature
            out[name] = values[oracle_name]
            if feature != "GLCM_HOM2":
                out[name + "_AVE"] = values[oracle_name]
    return out


def verify(header, table_name, computed):
    """Both directions: every pin reproduces, and every computed feature is pinned."""
    text = re.sub(r"//[^\n]*", "", open(header, encoding="utf-8").read())
    if table_name not in text:
        print(f"  {table_name} is not in {os.path.basename(header)} yet")
        return False
    body = text.split(table_name, 1)[1].split("{", 1)[1].split("};", 1)[0]
    pins = {n: float(v) for n, v in re.findall(r'\{\s*"(\w+)"\s*,\s*([-0-9.e+]+)\s*\}', body)}
    ok = True
    for name in sorted(set(pins) | set(computed)):
        if name not in computed:
            print(f"  FAIL {name}: pinned but not produced by this generator"); ok = False
        elif name not in pins:
            print(f"  FAIL {name}: produced but not pinned"); ok = False
        else:
            got, want = computed[name], pins[name]
            if abs(got - want) > TOL * max(abs(want), 1e-300):
                print(f"  FAIL {name}: oracle={got!r} pinned={want!r}"); ok = False
    print(f"  {len(pins)} pins checked against {len(computed)} computed values")
    return ok


def main():
    ok = True
    for dim, header, arr_name, shape, prefix, force2d, tname in (
            ("2D", HEADER_2D, "glcm_2d_gap_levels_pixels", SHAPE_2D, "", True,
             "glcm_2d_pyradiomics_gap_levels_ref_vals"),
            ("3D", HEADER_3D, "glcm_3d_gap_levels_voxels", SHAPE_3D, "3", False,
             "glcm_3d_pyradiomics_gap_levels_ref_vals")):
        arr = read_array(header, arr_name, shape)
        values, levels, n_angles = run(arr, force2d)
        missing = sorted(set(range(1, max(levels) + 1)) - set(levels))
        print(f"\n// {dim}: pyradiomics {radiomics.__version__}, recipe glcm.pyradiomics_gap_levels_bincount20, "
              f"{n_angles} directions")
        print(f"// levels present {levels}; empty below the highest: {missing}")
        computed = table(values, prefix)
        assert missing, "the fixture must leave bins empty"
        for name in sorted(computed):
            print('    {"%s", %.17g},' % (name, computed[name]))
        print(f"\n=== verify {tname} ===")
        ok = verify(header, tname, computed) and ok
    print("\n%s" % ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED -- do not paste goldens"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
