"""OFFLINE CellProfiler oracle for the 2D edge-intensity morphology features and
MASS_DISPLACEMENT (SPEC 4, oracle=cellprofiler). Runs the real
cellprofiler.modules.MeasureObjectIntensity module on the gtest fixture
`shape2d_morphology_intensity` / `shape2d_morphology_mask` (test_data.h) and validates it
against the goldens pinned in test_2d_morphology_cellprofiler.h.

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
env's Library\bin and friends on PATH, importing cellprofiler_core.image dies on a DLL
ordinal lookup (exit 0xC06D007E) with nothing on stderr, which reads like a crash in the
generator and is not one. And import the module package from a working directory on the SAME
drive as the env: cellprofiler.modules pulls in cellprofiler.gui.help.content, which calls
os.path.relpath() against the CWD, and that raises ValueError across drive letters.

Provenance: tool=cellprofiler, version=4.2.8 (module package) / cellprofiler-core 4.2.8.1,
centrosome 1.2.3, scikit-image as pinned by that env; python 3.9; module
MeasureObjectIntensity with a single image and a single object set, all settings at their
defaults. generator=tests/vetting/oracles/gen_morphology_cellprofiler.py. Run offline.
"""
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

# tests/test_data.h: shape2d_morphology_intensity and shape2d_morphology_mask, as rows y=0..7
# of columns x=0..7. A single irregular concave ROI with one interior hole at (x=3, y=3).
INTENSITY = [
    [0,  0, 12, 14,  0,  0, 0, 0],
    [0, 18, 20, 24, 26,  0, 0, 0],
    [30, 32, 35, 38, 42, 45, 0, 0],
    [34, 37, 40,  0, 48, 52, 0, 0],
    [0, 44, 47, 51, 55,  0, 0, 0],
    [0,  0, 53, 58, 62,  0, 0, 0],
    [0,  0,  0, 63, 68,  0, 0, 0],
    [0,  0,  0,  0,  0,  0, 0, 0],
]
MASK = [
    [0, 0, 1, 1, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 0, 0],
    [1, 1, 1, 0, 1, 1, 0, 0],
    [0, 1, 1, 1, 1, 0, 0, 0],
    [0, 0, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
]

PAD = 1           # background border, so "outside is background" is stated by the fixture
SCALE = 255.0     # CP measures on [0,1]; the fixture's raw values are 8-bit
IMAGE = "img"
OBJECTS = "objs"

# goldens CellProfiler reproduces -- pinned in tests/test_2d_morphology_cellprofiler.h
GOLDENS = {
    "MASS_DISPLACEMENT":         0.634476074243407,
    "EDGE_MEAN_INTENSITY":       41.8333333333333,
    "EDGE_MAX_INTENSITY":        68.0,
    "EDGE_MIN_INTENSITY":        12.0,
    "EDGE_INTEGRATED_INTENSITY": 753.0,
}
# EDGE_STDDEV_INTENSITY is NOT in that set: the two tools use different estimators over the
# identical 18 pixels. Nyxus divides by n-1 (Moments4::std() is sqrt(M2/(n-1)), a shared
# helper, so this is a house convention rather than a slip here); CellProfiler divides by n.
# The ratio is therefore exactly sqrt(n/(n-1)) and the divergence is checked as that identity,
# not waved through as "close enough". The row stays regression until an oracle is found that
# uses the same estimator, or Nyxus states which one it means to report.
NYXUS_EDGE_STDDEV = 16.7691944455582
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
# 1e-6 sits above that and far below any real disagreement.
TOL = 1e-6


def build():
    inten = np.pad(np.array(INTENSITY, dtype=float), PAD)
    labels = np.pad(np.array(MASK, dtype=np.int32), PAD)
    return inten / SCALE, labels


def run_cp():
    image, labels = build()

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


def edge_pixels():
    """The edge set, derived independently of CellProfiler, so a disagreement can be read as
    'different pixels' or 'same pixels, different arithmetic' rather than just a number."""
    inten = np.array(INTENSITY, dtype=float)
    mask = np.array(MASK, dtype=bool)
    padded = np.pad(mask, 1)
    interior = (padded[:-2, 1:-1] & padded[2:, 1:-1] &     # N, S
                padded[1:-1, :-2] & padded[1:-1, 2:])      # W, E
    edge = mask & ~interior[0:mask.shape[0], 0:mask.shape[1]]
    return inten[edge], inten[mask & ~edge]


def main():
    cp = run_cp()
    edge_vals, interior_vals = edge_pixels()

    print("=== fixture ===")
    print(f"  ROI pixels {int(np.array(MASK).sum())}, edge {edge_vals.size}, "
          f"interior {interior_vals.size}")
    print(f"  ROI intensity sum {np.array(INTENSITY)[np.array(MASK, dtype=bool)].sum():g}, "
          f"edge sum {edge_vals.sum():g}, interior sum {interior_vals.sum():g}")

    print("\n=== CellProfiler MeasureObjectIntensity vs the pinned goldens ===")
    all_ok = True
    for feat, gold in GOLDENS.items():
        got = cp[feat]
        ok = abs(got - gold) <= TOL * max(1.0, abs(gold))
        all_ok &= ok
        print(f"  {'OK  ' if ok else 'FAIL'} {feat:<26} cp={got!r:<22} golden={gold!r}")

    # The one divergence, checked as an identity rather than reported as a number.
    n = edge_vals.size
    pop = float(np.std(edge_vals, ddof=0))
    smp = float(np.std(edge_vals, ddof=1))
    bessel = (n / (n - 1.0)) ** 0.5
    cp_std = cp["EDGE_STDDEV_INTENSITY"]
    ident_ok = (abs(cp_std - pop) <= TOL * pop
                and abs(NYXUS_EDGE_STDDEV - smp) <= TOL * smp
                and abs(smp / pop - bessel) <= 1e-12)
    all_ok &= ident_ok
    print("\n=== EDGE_STDDEV_INTENSITY -- divergence by estimator, NOT vetted vs CP ===")
    print(f"  cellprofiler {cp_std!r}  == population std (/n)      {pop!r}")
    print(f"  nyxus        {NYXUS_EDGE_STDDEV!r}  == sample std     (/n-1)    {smp!r}")
    print(f"  ratio {smp / pop!r} == sqrt(n/(n-1)) {bessel!r} for n={n}")
    print(f"  {'identity holds' if ident_ok else 'IDENTITY BROKEN -- investigate'}")

    print(f"\n{'ALL CHECKS PASSED' if all_ok else 'SOME CHECKS FAILED -- do not promote'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
