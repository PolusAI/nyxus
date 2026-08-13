"""OFFLINE golden generator for tests/test_2d_glcm_pyradiomics.h (SPEC 6.4).

Runs PyRadiomics GLCM on the pinned dense phantom and prints the golden table to paste into the
oracle test. Run offline; CI never invokes it - the reference tool is not a runtime dependency.

    conda create -n nyxus_oracle -c conda-forge python=3.9 pyradiomics simpleitk
    python tests/vetting/oracles/gen_glcm_pyradiomics.py

Provenance to record at the pinned goldens: tool=pyradiomics, version (printed by this script),
config=glcm.ibsi_identity, fixture=the dense phantom below. The recipe is the IBSI one, not
glcm.pyradiomics_symmetric: binWidth=1 on this integer image is identity binning, so neither tool
discretises and both build the matrix over the image's own grey levels. Under
glcm.pyradiomics_symmetric the fixed bin count re-maps those levels, and every feature that reads
the absolute level - ACOR, SUMAVERAGE, IDN, IDMN - stops being comparable.

The fixture is img[i,j] = ((i + 2j) % 8) + 1 over 8x8 with a one-pixel background border: every grey
level 1..8 occurs and every level pair is populated, so the matrix is denser and larger than the
IBSI phantom's (in-mask levels {1,3,4,6}), which test_2d_glcm_ibsi.h already covers. Both fixtures
were run against this tool; see tests/vetting/audit/glcm_2d_pyradiomics_vetting_report.md.

PyRadiomics reports one value per feature for the whole angle set (it averages the per-angle
values), which is exactly the Nyxus *_AVE aggregation; the per-angle features are checked against
the same golden by averaging the 4 angles in the test.
"""
import json

import numpy as np
import SimpleITK as sitk
import radiomics
from radiomics import glcm

# PyRadiomics feature name -> the Nyxus features that quantity covers. Nyxus keeps several names
# for one quantity (ENERGY is ASM, HOM1 is Id, HOM2 is Idm, VARIANCE is the joint variance
# PyRadiomics calls SumSquares), and PyRadiomics 3.x dropped two features as duplicates of others:
# Dissimilarity == DifferenceAverage and SumVariance == ClusterTendency. Both equalities are
# PyRadiomics' own documented reason for dropping them.
PYRADIOMICS_TO_NYXUS = {
    "Autocorrelation":    ["GLCM_ACOR"],
    "ClusterProminence":  ["GLCM_CLUPROM"],
    "ClusterShade":       ["GLCM_CLUSHADE"],
    "ClusterTendency":    ["GLCM_CLUTEND", "GLCM_SUMVARIANCE"],
    "Contrast":           ["GLCM_CONTRAST"],
    "Correlation":        ["GLCM_CORRELATION"],
    "DifferenceAverage":  ["GLCM_DIFAVE", "GLCM_DIS"],
    "DifferenceEntropy":  ["GLCM_DIFENTRO"],
    "DifferenceVariance": ["GLCM_DIFVAR"],
    "Id":                 ["GLCM_ID", "GLCM_HOM1"],
    "Idm":                ["GLCM_IDM", "GLCM_HOM2"],
    "Idmn":               ["GLCM_IDMN"],
    "Idn":                ["GLCM_IDN"],
    "Imc1":               ["GLCM_INFOMEAS1"],
    "Imc2":               ["GLCM_INFOMEAS2"],
    "InverseVariance":    ["GLCM_IV"],
    "JointAverage":       ["GLCM_JAVE"],
    "JointEnergy":        ["GLCM_ASM", "GLCM_ENERGY"],
    "JointEntropy":       ["GLCM_JE", "GLCM_ENTROPY"],
    "MaximumProbability": ["GLCM_JMAX"],
    "SumAverage":         ["GLCM_SUMAVERAGE"],
    "SumEntropy":         ["GLCM_SUMENTROPY"],
    "SumSquares":         ["GLCM_JVAR", "GLCM_VARIANCE"],
}

# recipe glcm.ibsi_identity, PyRadiomics side: binWidth 1 is identity on an integer image, so
# neither tool discretises and both build the matrix over the same levels
SETTINGS = dict(binWidth=1, symmetricalGLCM=True, distances=[1], force2D=True, force2Ddimension=0,
                weightingNorm=None, label=1, interpolator=None, resampledPixelSpacing=None)


def dense_phantom():
    """8x8, every grey level 1..8 present, one-pixel background border -> (intensity, mask)."""
    i, j = np.meshgrid(np.arange(8), np.arange(8), indexing="ij")
    tile = ((i + 2 * j) % 8) + 1
    assert set(np.unique(tile).tolist()) == set(range(1, 9)), "phantom must be dense in 1..8"
    intensity = np.zeros((10, 10), np.int32)
    mask = np.zeros((10, 10), np.int32)
    intensity[1:9, 1:9] = tile
    mask[1:9, 1:9] = 1
    return intensity, mask


def run():
    intensity, mask = dense_phantom()
    image, label = (sitk.GetImageFromArray(a[None, :, :]) for a in (intensity, mask))
    for im in (image, label):
        im.SetSpacing((1.0, 1.0, 1.0))
    f = glcm.RadiomicsGLCM(image, label, **SETTINGS)
    f.enableAllFeatures()
    return {k: float(v) for k, v in f.execute().items()}


def emit(values, name_map, tool_note):
    """Print the golden table. Each quantity is emitted twice - once for the per-angle feature and
    once for its _AVE twin - because the oracle reports the angle-averaged value, which is what
    _AVE holds and what averaging the 4 angled values of the base feature produces. Both names have
    to appear literally: the coverage report credits a feature only where a test or its table names
    it. GLCM_HOM2 is the one quantity with no _AVE twin in the feature set."""
    print(tool_note)
    for oracle_name in sorted(name_map):
        if oracle_name not in values:
            print("// MISSING from this build of the tool: %s" % oracle_name)
            continue
        for feature in name_map[oracle_name]:
            names = [feature] if feature == "GLCM_HOM2" else [feature, feature + "_AVE"]
            for n in names:
                print('    {"%s", %.17g},   // %s' % (n, values[oracle_name], oracle_name))


def main():
    values = run()
    emit(values, PYRADIOMICS_TO_NYXUS,
         "// pyradiomics %s, recipe glcm.ibsi_identity, dense 8x8 phantom\n"
         "// generated by tests/vetting/oracles/gen_glcm_pyradiomics.py" % radiomics.__version__)
    print("\n// raw:", json.dumps(values, sort_keys=True))


if __name__ == "__main__":
    main()
