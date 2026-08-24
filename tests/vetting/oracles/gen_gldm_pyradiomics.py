"""OFFLINE PyRadiomics oracle for the 2D GLDM features, on the IBSI digital phantom.

    conda create -n nyxus_oracle -c conda-forge python=3.9 pyradiomics simpleitk
    python tests/vetting/oracles/gen_gldm_pyradiomics.py       (from the repository root)

Prints the paste-ready goldens AND re-verifies every golden pinned in test_2d_gldm_pyradiomics.h --
both the per-slice table and the four-slice means -- exiting non-zero on any mismatch or on any pin
this generator cannot produce.

PyRadiomics is the reference that defines GLDM, and it names all 14 of the family's features, so one
run covers the whole family. Worst residual against Nyxus over 13 of those features x 4 slices is
2.2e-16, which is why they assert at the SPEC 7 "exact" tier rather than a cross-tool band. GLDM_DE
is the exception at 1.3e-3, because calc_DE() takes its logarithm through the shared float
fast_log10() approximation; it asserts at rel=2.5e-3. RELTOL below is not that band -- this script
checks the pins against PyRadiomics, which is what produced them, so it stays at 1e-9.

Recipe `gldm.ibsi_phantom_2d`: the four IBSI digital-phantom slices, each featurised on its own.
Each slice is its own ROI and yields its own scalar; their mean is what the IBSI "2D, averaged"
aggregation publishes, and it is what assert_gldm_mean_against_golden_values() compares on the
Nyxus side. Both quantities are pinned, because a mean is weaker than the four values behind it --
errors in two slices that cancel leave it unmoved. `binWidth=1` is identity binning on this integer
phantom, so neither tool discretises; `gldm_a=0` and `distances=[1]` are the alpha=0, d=1 coarseness
Nyxus computes in IBSI mode; `force2D` keeps the neighbourhood inside the slice.

Provenance: tool=pyradiomics (version printed by this script); SimpleITK; numpy; env=nyxus_oracle
(conda, Python 3.9); generator=tests/vetting/oracles/gen_gldm_pyradiomics.py. Run offline; CI never
invokes it.
"""
import importlib.util
import os
import re

import numpy as np
import SimpleITK as sitk
import radiomics
from radiomics import gldm

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.dirname(os.path.dirname(HERE))
TEST_H = os.path.join(TESTS, "test_2d_gldm_pyradiomics.h")

RELTOL = 1e-9          # SPEC 7 exact tier; measured worst residual is 1.0e-15

# PyRadiomics GLDM feature -> Nyxus feature. The names are the same quantities spelled long-form.
PYRADIOMICS_TO_NYXUS = {
    "SmallDependenceEmphasis": "GLDM_SDE",
    "LargeDependenceEmphasis": "GLDM_LDE",
    "GrayLevelNonUniformity": "GLDM_GLN",
    "DependenceNonUniformity": "GLDM_DN",
    "DependenceNonUniformityNormalized": "GLDM_DNN",
    "GrayLevelVariance": "GLDM_GLV",
    "DependenceVariance": "GLDM_DV",
    "DependenceEntropy": "GLDM_DE",
    "LowGrayLevelEmphasis": "GLDM_LGLE",
    "HighGrayLevelEmphasis": "GLDM_HGLE",
    "SmallDependenceLowGrayLevelEmphasis": "GLDM_SDLGLE",
    "SmallDependenceHighGrayLevelEmphasis": "GLDM_SDHGLE",
    "LargeDependenceLowGrayLevelEmphasis": "GLDM_LDLGLE",
    "LargeDependenceHighGrayLevelEmphasis": "GLDM_LDHGLE",
}

SETTINGS = dict(binWidth=1, force2D=True, force2Ddimension=0, gldm_a=0, distances=[1],
                label=1, interpolator=None, resampledPixelSpacing=None)


def phantom_slices():
    """The IBSI phantom, read out of tests/test_data.h by the shared helper."""
    spec = importlib.util.spec_from_file_location(
        "ibsi_phantom", os.path.join(HERE, "ibsi_phantom.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.phantom_slices()


def parse_pins(txt, table):
    # the opening brace may sit on the declaration line or the next one, so match across whitespace
    m = re.search(re.escape(table) + r"\s*\{", txt)
    if not m:
        raise RuntimeError("table %s not found in %s" % (table, os.path.basename(TEST_H)))
    body = txt[m.end():].split("};", 1)[0]
    body = re.sub(r"//[^\n]*", "", body)          # a commented-out golden is not a pin
    return {n: float(v) for n, v in
            re.findall(r'\{\s*"([A-Za-z0-9_]+)"\s*,\s*([-0-9.eE+]+)\s*\}', body)}


def run():
    """-> {nyxus feature: [value per phantom slice, in z order]}."""
    per_slice = {}
    for intensity, mask in phantom_slices():
        image = sitk.GetImageFromArray(intensity[None, :, :].astype(np.float64))
        roi = sitk.GetImageFromArray(mask[None, :, :].astype(np.int32))
        f = gldm.RadiomicsGLDM(image, roi, **SETTINGS)
        f.enableAllFeatures()
        result = f.execute()
        for pr_name, nyxus_name in PYRADIOMICS_TO_NYXUS.items():
            if pr_name not in result:
                raise RuntimeError("pyradiomics produced no %s" % pr_name)
            per_slice.setdefault(nyxus_name, []).append(float(result[pr_name]))
    return per_slice


def verify(pins, got, label):
    """Compares one pinned table against this run. -> (verified, failed, unproducible)."""
    print("")
    print("# verifying %d pinned %s goldens against this run" % (len(pins), label))
    nok = nfail = nmiss = 0
    for name in sorted(pins):
        want = pins[name]
        if name not in got:
            print("  MISSING %s: pinned %r but pyradiomics produces no such value" % (name, want))
            nmiss += 1
            continue
        have = got[name]
        rel = abs(have - want) / max(abs(want), 1e-12)
        if rel <= RELTOL:
            print("  OK   %s: pyradiomics=%r pinned=%r rel=%.3g" % (name, have, want, rel))
            nok += 1
        else:
            print("  FAIL %s: pyradiomics=%r pinned=%r rel=%.3g" % (name, have, want, rel))
            nfail += 1
    return nok, nfail, nmiss


def main():
    slices = run()
    means = {k: float(np.mean(v)) for k, v in slices.items()}
    by_slice = {"%s_z%d" % (k, z + 1): v for k, vs in slices.items() for z, v in enumerate(vs)}

    print("# pyradiomics %s, SimpleITK %s, numpy %s, recipe gldm.ibsi_phantom_2d"
          % (radiomics.__version__, sitk.Version_VersionString(), np.__version__))
    print("# paste-ready goldens (mean over the four IBSI phantom slices)")
    for name in sorted(means):
        print('\t{"%s", %r},' % (name, means[name]))
    print("# paste-ready per-slice goldens")
    for name in sorted(by_slice):
        print('\t{"%s", %r},' % (name, by_slice[name]))

    if not os.path.exists(TEST_H):
        print("")
        print("%s does not exist yet - nothing to verify" % os.path.basename(TEST_H))
        return 0

    # Both tables the header pins, not just the means: a per-slice golden nothing re-derives is
    # exactly the kind of pin that stops being checked the moment it is written.
    text = open(TEST_H, encoding="utf-8", errors="replace").read()
    nok, nfail, nmiss = verify(parse_pins(text, "gldm_2d_pyradiomics_ref_vals"), means, "mean")
    a, b, c = verify(parse_pins(text, "gldm_2d_pyradiomics_slice_ref_vals"), by_slice, "per-slice")
    nok, nfail, nmiss = nok + a, nfail + b, nmiss + c

    print("")
    print("%d verified, %d failed, %d unproducible" % (nok, nfail, nmiss))
    if nfail or nmiss:
        print("SOME CHECKS FAILED -- do not promote")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
