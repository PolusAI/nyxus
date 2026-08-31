"""OFFLINE CellProfiler oracle for the 2D edge-intensity morphology features and
MASS_DISPLACEMENT (SPEC 4, oracle=cellprofiler). Runs the real
cellprofiler.modules.MeasureObjectIntensity module on the gtest fixture
`shape2d_morphology_intensity` / `shape2d_morphology_mask` (test_data.h) and re-verifies EVERY
golden pinned in test_2d_morphology_cellprofiler.h against that run, exiting non-zero on any
mismatch, on any pin this generator cannot produce, and on any feature CellProfiler vets that the
header does not pin.

Both the fixture and the pins are READ OUT OF THE TREE rather than transcribed here. A copy of
either would only ever compare this script against itself: a test_data.h edit would keep driving
CellProfiler with the old fixture while the gtest input moved, and a hand-edited golden in the
header would go unnoticed. The Nyxus stddev pin is bound the same way, out of
test_2d_morphology_regression.h, so the divergence identity below is checked against the number the
C++ side actually asserts.

The six features map onto one module, one measurement each:

  EDGE_INTEGRATED_INTENSITY  <- Intensity_IntegratedIntensityEdge_<image>
  EDGE_MEAN_INTENSITY        <- Intensity_MeanIntensityEdge_<image>
  EDGE_STDDEV_INTENSITY      <- Intensity_StdIntensityEdge_<image>
  EDGE_MAX_INTENSITY         <- Intensity_MaxIntensityEdge_<image>
  EDGE_MIN_INTENSITY         <- Intensity_MinIntensityEdge_<image>
  MASS_DISPLACEMENT          <- Intensity_MassDisplacement_<image>

Definitions that make the comparison exact:

- CellProfiler's edge is skimage.segmentation.find_boundaries(labels, mode="inner"),
  i.e. connectivity=1 (4-neighbourhood): an object pixel is an edge pixel unless all four
  of its N/S/E/W neighbours carry the same label. On this fixture that excludes exactly
  eight pixels -- the ones enclosed by the concavity and the interior hole -- so 18 of the
  26 ROI pixels are edge. Nyxus reaches the same set, which is why the integrated value
  agrees to the last digit rather than approximately.
- Outside the array counts as background for both tools, so the two ROI pixels in column 0
  are edge pixels. The fixture is padded with background here to state that explicitly
  rather than relying on either tool's array-border handling.
- CellProfiler measures intensity on a float image in [0, 1], so the image is fed as
  raw/255 and the four scale-carrying results are multiplied back by 255. MASS_DISPLACEMENT
  is a distance in pixels between two centroids and is invariant to that scaling.

Environment: a dedicated CellProfiler env is required to RUN this generator
(cellprofiler-core + centrosome + the cellprofiler module package, headless). CI never
invokes it -- CellProfiler is not a runtime dependency.

Windows notes: activate the env rather than calling its python.exe by path -- without the
env's Library\\bin and friends on PATH, importing cellprofiler_core.image dies on a DLL
ordinal lookup (exit 0xC06D007E) with nothing on stderr, which reads like a crash in the
generator and is not one. And import the module package from a working directory on the SAME
drive as the env: cellprofiler.modules pulls in cellprofiler.gui.help.content, which calls
os.path.relpath() against the CWD, and that raises ValueError across drive letters.

Provenance: tool=cellprofiler, version=4.2.8 (module package) / cellprofiler-core 4.2.8.1,
centrosome 1.2.3, scikit-image as pinned by that env; python 3.9; module
MeasureObjectIntensity with a single image and a single object set, all settings at their
defaults. generator=tests/vetting/oracles/gen_morphology_cellprofiler.py. Run offline.
"""
import os
import re
import warnings
warnings.filterwarnings("ignore")
import numpy as np

import cellprofiler_core.preferences as cpprefs
cpprefs.set_headless()
import cellprofiler_core.image as cpi
import cellprofiler_core.measurement as cpmeas
import cellprofiler_core.object as cpo
import cellprofiler_core.pipeline as cpp
import cellprofiler_core.workspace as cpw
from cellprofiler.modules import measureobjectintensity as moi

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.dirname(os.path.dirname(HERE))
DATA_H = os.path.join(TESTS, "test_data.h")
INTENSITY_FIXTURE = "shape2d_morphology_intensity"
MASK_FIXTURE = "shape2d_morphology_mask"
TEST_H = os.path.join(TESTS, "test_2d_morphology_cellprofiler.h")
TABLE = "morphology_2d_cellprofiler_ref_vals"
REGRESSION_H = os.path.join(TESTS, "test_2d_morphology_regression.h")
REGRESSION_TABLE = "morphology_2d_regression_ref_vals"

PAD = 1           # background border, so "outside is background" is stated by the fixture
SCALE = 255.0     # CP measures on [0,1]; the fixture's raw values are 8-bit
IMAGE = "img"
OBJECTS = "objs"

# the five features CP and Nyxus compute the same way; the header pins CP's own digits for them
CP_VETTED = ("MASS_DISPLACEMENT", "EDGE_MEAN_INTENSITY", "EDGE_MAX_INTENSITY",
             "EDGE_MIN_INTENSITY", "EDGE_INTEGRATED_INTENSITY")
# EDGE_STDDEV_INTENSITY is NOT among them: the two tools use different estimators over the
# identical 18 pixels. Nyxus divides by n-1 (Moments4::std() is sqrt(M2/(n-1)), a shared
# helper, so this is a house convention rather than a slip here); CellProfiler divides by n.
# The ratio is therefore exactly sqrt(n/(n-1)) and the divergence is checked as that identity,
# not waved through as "close enough". Its pin lives in the regression header and is read from
# there, so an edit to that number fails this generator instead of quietly redefining the gap.
STDDEV_FEATURE = "EDGE_STDDEV_INTENSITY"

# measurement name and whether the value carries the image's intensity scale
MEASUREMENTS = {
    "MASS_DISPLACEMENT":         (moi.MASS_DISPLACEMENT,         False),
    "EDGE_MEAN_INTENSITY":       (moi.MEAN_INTENSITY_EDGE,       True),
    "EDGE_STDDEV_INTENSITY":     (moi.STD_INTENSITY_EDGE,        True),
    "EDGE_MAX_INTENSITY":        (moi.MAX_INTENSITY_EDGE,        True),
    "EDGE_MIN_INTENSITY":        (moi.MIN_INTENSITY_EDGE,        True),
    "EDGE_INTEGRATED_INTENSITY": (moi.INTEGRATED_INTENSITY_EDGE, True),
}
# Relative band. The two tools do the same arithmetic on the same 18 pixels, so the only
# residual is CellProfiler storing the image as float32: Image(float64).pixel_data.dtype is
# float32, so a raw value round-trips as raw/255 -> float32 -> *255 and comes back within
# ~1 ulp of float32 (measured: 5.2e-8 relative on the max, 2.1e-8 on the integrated sum).
# 1e-6 sits above that and far below any real disagreement. It is the band the header
# asserts at, so this check and the gtest one cannot disagree about what "agrees" means.
TOL = 1e-6


def parse_grid(txt, name):
    """The {x, y, value} pixel array `name` from test_data.h, as a dense 2D numpy array.

    Read out of the checked-in fixture rather than transcribed here: a copy in this file would
    keep driving CellProfiler with the old fixture after a test_data.h edit, so the gtest input
    could move while this oracle stayed green.
    """
    body = txt.split(name + "[] = {", 1)[1].split("};", 1)[0]
    body = re.sub(r"//[^\n]*", "", body)
    px = [(int(x), int(y), int(v)) for x, y, v in
          re.findall(r"\{\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\}", body)]
    if not px:
        raise RuntimeError("fixture %s not found in %s" % (name, os.path.basename(DATA_H)))
    grid = np.zeros((max(y for _, y, _ in px) + 1, max(x for x, _, _ in px) + 1), dtype=np.int64)
    for x, y, v in px:
        grid[y, x] = v                              # test_data.h is {x, y, ...}; numpy is [row, col]
    return grid


def parse_pins(path, table):
    """The header's own reference table, as {feature: value}."""
    txt = open(path, encoding="utf-8", errors="replace").read()
    m = re.search(re.escape(table) + r"\s*\{", txt)
    if not m:
        raise RuntimeError("table %s not found in %s" % (table, os.path.basename(path)))
    body = txt[m.end():].split("\n};", 1)[0]
    body = re.sub(r"//[^\n]*", "", body)            # a commented-out golden is not a pin
    return {n: float(v) for n, v in
            re.findall(r'\{\s*"(\w+)"\s*,\s*([-0-9.eE+]+)\s*\}', body)}


def build():
    txt = open(DATA_H, encoding="utf-8", errors="replace").read()
    inten = np.pad(parse_grid(txt, INTENSITY_FIXTURE).astype(float), PAD)
    labels = np.pad(parse_grid(txt, MASK_FIXTURE).astype(np.int32), PAD)
    if inten.shape != labels.shape:
        raise RuntimeError("intensity and mask fixtures differ in shape")
    return inten, labels


def run_cp(inten, labels):
    image = inten / SCALE

    module = moi.MeasureObjectIntensity()
    module.images_list.value = IMAGE
    module.objects_list.value = OBJECTS

    objects = cpo.Objects()
    objects.segmented = labels                      # CP indexes [row=y, col=x]
    oset = cpo.ObjectSet()
    oset.add_objects(objects, OBJECTS)

    isl = cpi.ImageSetList()
    iset = isl.get_image_set(0)
    iset.add(IMAGE, cpi.Image(image))

    m = cpmeas.Measurements()
    ws = cpw.Workspace(cpp.Pipeline(), module, iset, oset, m, isl)
    module.run(ws)

    out = {}
    for feat, (meas, scaled) in MEASUREMENTS.items():
        v = m.get_current_measurement(OBJECTS, "%s_%s_%s" % (moi.INTENSITY, meas, IMAGE))
        out[feat] = float(v[0]) * (SCALE if scaled else 1.0)
    return out


def edge_pixels(inten, labels):
    """The edge set, derived independently of CellProfiler, so a disagreement can be read as
    'different pixels' or 'same pixels, different arithmetic' rather than just a number."""
    mask = labels.astype(bool)
    padded = np.pad(mask, 1)
    interior = (padded[:-2, 1:-1] & padded[2:, 1:-1] &     # N, S
                padded[1:-1, :-2] & padded[1:-1, 2:])      # W, E
    edge = mask & ~interior
    return inten[edge], inten[mask & ~edge]


def main():
    inten, labels = build()
    cp = run_cp(inten, labels)
    edge_vals, interior_vals = edge_pixels(inten, labels)

    print("=== fixture (read from tests/test_data.h) ===")
    print("  %s / %s, padded by %d" % (INTENSITY_FIXTURE, MASK_FIXTURE, PAD))
    print("  ROI pixels %d, edge %d, interior %d"
          % (int(labels.astype(bool).sum()), edge_vals.size, interior_vals.size))
    print("  ROI intensity sum %g, edge sum %g, interior sum %g"
          % (inten[labels.astype(bool)].sum(), edge_vals.sum(), interior_vals.sum()))

    print("\n=== paste-ready goldens ===")
    for f in CP_VETTED:
        print('\t{"%s", %r},' % (f, cp[f]))

    # ---- forward: every pin in the header must be a feature CP vets, and must match this run.
    pins = parse_pins(TEST_H, TABLE)
    print("\n=== verifying %d pinned goldens in %s against this run ==="
          % (len(pins), os.path.basename(TEST_H)))
    nok = nfail = nmiss = 0
    for f in sorted(pins):
        want = pins[f]
        if f not in CP_VETTED:
            print("  EXTRA  %-26s pinned %r but this recipe does not vet it" % (f, want))
            nmiss += 1
            continue
        got = cp[f]
        err = abs(got - want) / max(1.0, abs(want))
        if err <= TOL:
            print("  OK     %-26s cp=%-22r pinned=%-22r rel=%.3g" % (f, got, want, err))
            nok += 1
        else:
            print("  FAIL   %-26s cp=%-22r pinned=%-22r rel=%.3g" % (f, got, want, err))
            nfail += 1

    # ---- reverse: every feature CP vets must be pinned, or the header quietly lost coverage.
    unpinned = [f for f in CP_VETTED if f not in pins]
    for f in unpinned:
        print("  UNPINNED %-24s CP vets it but the header pins nothing" % f)

    # ---- the one divergence, checked as an identity against the pin the C++ side asserts.
    reg = parse_pins(REGRESSION_H, REGRESSION_TABLE)
    nyxus_std = reg.get(STDDEV_FEATURE)
    n = edge_vals.size
    pop = float(np.std(edge_vals, ddof=0))
    smp = float(np.std(edge_vals, ddof=1))
    bessel = (n / (n - 1.0)) ** 0.5
    cp_std = cp[STDDEV_FEATURE]
    print("\n=== %s -- divergence by estimator, NOT vetted vs CP ===" % STDDEV_FEATURE)
    if nyxus_std is None:
        print("  UNPINNED %s: not in %s::%s -- the divergence has no pinned Nyxus side"
              % (STDDEV_FEATURE, os.path.basename(REGRESSION_H), REGRESSION_TABLE))
        ident_ok = False
    elif STDDEV_FEATURE in pins:
        print("  MISPLACED %s is pinned in %s, which claims CellProfiler backs it -- it does not"
              % (STDDEV_FEATURE, os.path.basename(TEST_H)))
        ident_ok = False
    else:
        ident_ok = (abs(cp_std - pop) <= TOL * pop
                    and abs(nyxus_std - smp) <= TOL * smp
                    and abs(smp / pop - bessel) <= 1e-12)
        print("  cellprofiler %r  == population std (/n)   %r" % (cp_std, pop))
        print("  nyxus        %r  == sample std     (/n-1) %r  (pinned in %s)"
              % (nyxus_std, smp, os.path.basename(REGRESSION_H)))
        print("  ratio %r == sqrt(n/(n-1)) %r for n=%d" % (smp / pop, bessel, n))
        print("  %s" % ("identity holds" if ident_ok else "IDENTITY BROKEN -- investigate"))

    print("\n%d verified, %d failed, %d unproducible, %d unpinned" % (nok, nfail, nmiss, len(unpinned)))
    if nfail or nmiss or unpinned or not ident_ok:
        print("SOME CHECKS FAILED -- do not promote")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
