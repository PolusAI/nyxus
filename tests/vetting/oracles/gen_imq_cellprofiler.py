"""OFFLINE CellProfiler oracle for the image-quality saturation features (SPEC 4,
oracle=cellprofiler).

Runs the real cellprofiler.modules.MeasureImageQuality module on the gtest fixture
`im_quality_intensity` / `im_quality_mask` and re-verifies EVERY golden pinned in
test_imq_cellprofiler.h against that run, exiting non-zero on any mismatch, on any pin this
generator cannot produce, and on any value it produces that the header pins nothing for. Both the
fixture and the pins are parsed out of the checked-in files: a copy of either kept here would only
ever compare this script against itself.

Vets (2):
  MIN_SATURATION -- CP Image_ImageQuality_PercentMinimal / 100
  MAX_SATURATION -- CP Image_ImageQuality_PercentMaximal / 100

Both tools use the same convention: the fraction of pixels equal to the image's OWN observed
extremum (CP: numpy.min/numpy.max of the pixel data, masked when a mask is present), not a fixed
bit-depth threshold such as 255. CP's calculate_saturation():

    number_pixels_maximal = numpy.sum(pixel_data == numpy.max(pixel_data))
    number_pixels_minimal = numpy.sum(pixel_data == numpy.min(pixel_data))
    percent_maximal = 100.0 * number_pixels_maximal / pixel_count
    percent_minimal = 100.0 * number_pixels_minimal / pixel_count

The metric is scale-invariant (it only compares pixels to the extremum), so CP's usual [0,1] float
images and Nyxus' integer PixIntens give the same fraction; this generator checks that explicitly by
running raw and max-normalized inputs.

WHY MAX_SATURATION IS PINNED AT ...669 AND NOT ...666. The two tools count the same 16 of 96 pixels.
CP reports a PERCENTAGE, so the fraction reaches this script as 100.0*16/96 divided by 100.0, which
is one ulp above the 16/96 Nyxus computes -- 2.8e-17 absolute, 1.7e-16 relative. The pin carries
CellProfiler's digits, as the pin of an oracle table should. MIN_SATURATION has no such gap because
18/96 = 0.1875 is exact in binary.

SCOPE OF THE CLAIM -- two places where the two implementations are NOT interchangeable, neither of
them exercised by this fixture:
  * Constant ROI (min == max): CP counts minimal and maximal independently and reports 100% for
    both; Nyxus' get_percent_max_pixels() uses `else if`, so a pixel equal to both extrema is
    counted only as maximal. Measured on a constant 4x4 ROI, Nyxus returns MIN_SATURATION=0 and
    MAX_SATURATION=1. This assertion does not cover that case.
  * Nyxus computes over the ROI's bounding-box image matrix, in which pixels inside the box but
    outside the mask are 0 and DO take part in the min/extremum count; CP restricts to image.mask
    when a mask is present. They coincide here only because im_quality_mask covers the whole 8x12
    box.

NOT vetted here: FOCUS_SCORE / LOCAL_FOCUS_SCORE. CellProfiler has features by those names, but they
are a different statistic -- CP's FocusScore is the *normalized variance of the raw image*,
sum((x-mean)^2)/(N*mean), and CP's LocalFocusScore is var(local_norm_var)/median(local_norm_var)
over a grid. Nyxus implements the Pech-Pacheco variance-of-Laplacian instead, which is vetted
against OpenCV in gen_imq_opencv.py. Do not point a CP oracle at those two names; this generator
fails if either appears in the header's CellProfiler table.

Environment: a dedicated CellProfiler env is required to RUN this generator (cellprofiler-core +
centrosome + the cellprofiler module package, headless -- see tests/vetting/TOOLS.md). CI never
invokes it: CellProfiler is not a runtime dependency.

Provenance of the run behind the pins: tool=cellprofiler, version=4.2.8 (module package) /
cellprofiler-core 4.2.8.1, centrosome 1.2.3, numpy 1.26.4, python 3.9, conda env
nyxus_cellprofiler. Every run prints its own installed versions, so this line describes the run
that produced the goldens rather than claiming anything about a later one; MeasureImageQuality,
calculate_saturation() on a single grayscale image, no mask.
generator=tests/vetting/oracles/gen_imq_cellprofiler.py. Run offline.

Note: importing cellprofiler.modules.measureimagequality pulls in cellprofiler.gui.help.content,
which calls os.path.relpath() against the current working directory. Run this generator from a
directory on the same drive as the CellProfiler install, or the import dies with "path is on mount
'C:', start on mount 'D:'".
"""
import os
import re
import warnings
from importlib import metadata

warnings.filterwarnings("ignore")
import numpy as np

import cellprofiler_core.preferences as cpprefs

cpprefs.set_headless()
import cellprofiler_core.image as cpi
import cellprofiler_core.measurement as cpmeas
import cellprofiler_core.object as cpo
import cellprofiler_core.pipeline as cpp
import cellprofiler_core.workspace as cpw
from cellprofiler.modules import measureimagequality as miq

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.dirname(os.path.dirname(HERE))
DATA_H = os.path.join(TESTS, "test_data.h")
TEST_H = os.path.join(TESTS, "test_imq_cellprofiler.h")
TABLE = "imq_cellprofiler_ref_vals"

IMAGE_NAME = "imq"
# The pins in the header are the TOOL's own digits printed at %.17g, so re-verifying them against a
# fresh run must ROUND-TRIP EXACTLY: a decimal literal with 17 significant digits parses back to the
# same double. This is deliberately NOT the band the C++ test asserts at (SPEC 7's exact tier, an
# absolute 1e-9). Conflating the two -- which is what this constant used to do -- would let a
# hand-edited golden drift by up to 1e-9 and still be reported as verified. A non-zero residual here
# means the tool's own output moved, and the pin has to be regenerated rather than tolerated.
ROUND_TRIP_ABSTOL = 0.0

# CP publishes features under these names too, but they are a different statistic from Nyxus'
# (see the module docstring). A CellProfiler table that pins either is claiming an oracle CP does
# not provide, so their presence is an error rather than a warning.
NOT_A_CP_ORACLE = ("FOCUS_SCORE", "LOCAL_FOCUS_SCORE")


def installed(dist):
    """The installed version of `dist`, or "unknown". None of these expose __version__."""
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return "unknown"


def parse_pixels(txt, name):
    """The {x, y, value} array `name` from test_data.h, as (x, y, value) triples.

    Read out of the checked-in fixture rather than transcribed here: a copy in this file would keep
    driving CellProfiler with the old ROI after a test_data.h edit, and the goldens it printed would
    silently stop describing what the C++ side computes.
    """
    body = txt.split("const static NyxusPixel " + name + "[] = {", 1)[1].split("};", 1)[0]
    body = re.sub(r"//[^\n]*", "", body)
    trips = [(int(a), int(b), int(c)) for a, b, c in
             re.findall(r"\{\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\}", body)]
    if not trips:
        raise RuntimeError("fixture %s not found in %s" % (name, os.path.basename(DATA_H)))
    return trips


def build_roi():
    """The ROI image matrix Nyxus builds: the AABB of the masked pixels, out-of-ROI left at 0."""
    txt = open(DATA_H, encoding="utf-8", errors="replace").read()
    inten = parse_pixels(txt, "im_quality_intensity")
    mask = parse_pixels(txt, "im_quality_mask")
    if len(inten) != len(mask):
        raise RuntimeError("intensity/mask fixtures differ in length: %d vs %d"
                           % (len(inten), len(mask)))
    kept = [p for p, m in zip(inten, mask) if m[2] != 0]
    x0 = min(x for x, _, _ in kept)
    y0 = min(y for _, y, _ in kept)
    w = max(x for x, _, _ in kept) - x0 + 1
    h = max(y for _, y, _ in kept) - y0 + 1
    img = np.zeros((h, w), float)
    for x, y, v in kept:
        img[y - y0, x - x0] = v
    return img


def parse_pins(path, table):
    """The header's own table, {feature: value}.

    Brace-counted rather than matched with a non-greedy regex: a regex that stops at the first "}}"
    swallows the closing brace of the last entry, which is how a header parser silently dropped a
    pin once already.
    """
    txt = open(path, encoding="utf-8", errors="replace").read()
    m = re.search(re.escape(table) + r"\s*\{", txt)
    if not m:
        raise RuntimeError("table %s not found in %s" % (table, os.path.basename(path)))
    depth, j = 1, m.end()
    while j < len(txt) and depth:
        if txt[j] == "{":
            depth += 1
        elif txt[j] == "}":
            depth -= 1
        j += 1
    body = re.sub(r"//[^\n]*", "", txt[m.end():j - 1])   # a commented-out golden is not a pin
    pins = {n: float(v) for n, v in
            re.findall(r'\{\s*"(\w+)"\s*,\s*([-0-9.eE+]+)\s*\}', body)}
    if not pins:
        raise RuntimeError("table %s in %s holds no pins" % (table, os.path.basename(path)))
    return pins


def run_cp(pixel_data):
    """Run MeasureImageQuality's saturation metrics on one grayscale image."""
    module = miq.MeasureImageQuality()
    group = module.image_groups[0]
    module.images_choice.value = miq.O_SELECT
    group.image_names.value = IMAGE_NAME  # ImageListSubscriber parses a ", "-joined string
    group.check_blur.value = False
    group.check_saturation.value = True
    group.check_intensity.value = False
    group.calculate_threshold.value = False

    isl = cpi.ImageSetList()
    image_set = isl.get_image_set(0)
    image_set.add(IMAGE_NAME, cpi.Image(pixel_data))

    m = cpmeas.Measurements()
    ws = cpw.Workspace(cpp.Pipeline(), module, image_set, cpo.ObjectSet(), m, isl)
    module.calculate_saturation(group, ws)

    def get(f):
        return float(m.get_current_measurement(
            "Image", "{}_{}_{}".format(miq.C_IMAGE_QUALITY, f, IMAGE_NAME)))

    return {"MIN_SATURATION": get(miq.F_PERCENT_MINIMAL) / 100.0,
            "MAX_SATURATION": get(miq.F_PERCENT_MAXIMAL) / 100.0}


def main():
    img = build_roi()
    pins = parse_pins(TEST_H, TABLE)
    produced = run_cp(img)
    normalized = run_cp(img / img.max())   # CP's usual [0,1] float convention

    n = img.size
    n_min = int((img == img.min()).sum())
    n_max = int((img == img.max()).sum())

    all_ok = True
    print("=== CellProfiler MeasureImageQuality (saturation) vs the pins in "
          "test_imq_cellprofiler.h ===")
    # The INSTALLED versions, not the docstring's. A generator that labels every run with a
    # literal reports a provenance it did not produce, which is what #448 found in the mirp one.
    print("    %s, numpy %s"
          % (", ".join("%s %s" % (d, installed(d))
                       for d in ("cellprofiler", "cellprofiler-core", "centrosome")),
             np.__version__))
    print("    fixture: %dx%d ROI, min=%.0f max=%.0f, %d at min, %d at max, %d pixels"
          % (img.shape[1], img.shape[0], img.min(), img.max(), n_min, n_max, n))

    # (1) range and identity checks: a saturation fraction is a count over the pixel total
    for name, count in (("MIN_SATURATION", n_min), ("MAX_SATURATION", n_max)):
        value = produced[name]
        ok = 0.0 <= value <= 1.0 and abs(value - count / n) <= 1e-15
        all_ok &= ok
        print("  %s range: %s = %.17g in [0,1] and equals %d/%d"
              % ("OK " if ok else "FAIL", name, value, count, n))
    ok = produced["MIN_SATURATION"] + produced["MAX_SATURATION"] <= 1.0 + 1e-15
    all_ok &= ok
    print("  %s identity: MIN + MAX <= 1 (disjoint pixel sets, min != max here)"
          % ("OK " if ok else "FAIL"))

    # (2) scale invariance, the property that lets CP's [0,1] floats vet Nyxus' integers
    for name in sorted(produced):
        ok = abs(normalized[name] - produced[name]) <= 1e-15
        all_ok &= ok
        print("  %s scale-invariant: %s unchanged on max-normalized input"
              % ("OK " if ok else "FAIL", name))

    # (3) every pin re-verified against this run
    for name in sorted(pins):
        if name in NOT_A_CP_ORACLE:
            all_ok = False
            print("  FAIL %s is pinned in %s under a CellProfiler table, but CP's feature of that "
                  "name is a different statistic" % (name, os.path.basename(TEST_H)))
            continue
        if name not in produced:
            all_ok = False
            print("  FAIL %s is pinned in %s but this generator cannot produce it"
                  % (name, os.path.basename(TEST_H)))
            continue
        resid = abs(produced[name] - pins[name])
        ok = resid <= ROUND_TRIP_ABSTOL
        all_ok &= ok
        rel = resid / abs(pins[name]) if pins[name] else 0.0
        print("  %s %s: cellprofiler=%.17g pinned=%.17g  abs=%.3g rel=%.3g"
              % ("OK " if ok else "FAIL", name, produced[name], pins[name], resid, rel))

    # (4) the reverse direction: a value this oracle produces that the header pins nothing for
    for name in sorted(set(produced) - set(pins)):
        all_ok = False
        print("  FAIL %s: cellprofiler produces %.17g and %s pins nothing for it"
              % (name, produced[name], os.path.basename(TEST_H)))

    print("\n%s" % ("ALL CP-VET CHECKS PASSED" if all_ok
                    else "SOME CHECKS FAILED -- do not promote"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
