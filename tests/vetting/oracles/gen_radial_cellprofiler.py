"""OFFLINE CellProfiler oracle probe for the 2D radial intensity distribution
(FRAC_AT_D, MEAN_FRAC, RADIAL_CV; SPEC 4, candidate oracle=cellprofiler).

Runs the real cellprofiler.modules.measureobjectintensitydistribution module on the
gtest fixture `shape2d_morphology_{intensity,mask}` (test_data.h) at 8 radial bins,
and compares it against the goldens pinned in test_2d_radial_regression.h.

Outcome, reproduced by every run of this script: CellProfiler does NOT vet this
family. Nyxus computes a different quantity under each of the three CellProfiler
names -- see tests/vetting/audit/radial_2d_cellprofiler_vetting_report.md for the
six divergences and the numbers. The family therefore stays status=regression, and
this generator exists to keep that verdict honest rather than to produce goldens:
it fails if CellProfiler ever starts agreeing (the family would then be promotable)
just as loudly as it fails if a pin stops reproducing.

Nothing here carries a literal copy of anything. The fixture is parsed out of
tests/test_data.h, the pinned feature vectors out of tests/test_2d_radial_regression.h,
and the centre pixel and normalising radius out of tests/test_2d_radial_mechanics.h,
so "ALL CHECKS PASSED" cannot mean the script agrees with its own copy.

Environment: a dedicated CellProfiler env is required (cellprofiler 4.2.8 +
cellprofiler-core + centrosome, headless, python 3.9). CI never invokes it --
CellProfiler is not a runtime dependency. On Windows the interpreter's working
directory must be on the SAME DRIVE as the cellprofiler package or importing the
module raises ValueError from os.path.relpath; see tests/vetting/TOOLS.md.

Provenance: tool=cellprofiler, version=4.2.8 (module package) / cellprofiler-core
4.2.8.1; python 3.9; MeasureObjectIntensityDistribution, center_choice="These objects",
bin_count=8, wants_scaled=True, zernikes off. generator=tests/vetting/oracles/
gen_radial_cellprofiler.py. Run offline.

Those versions are read back from the installed distributions on every run and printed
with the results, and a mismatch stops the run: a report that says 4.2.8 has to have
been produced by 4.2.8. `--allow-version-drift` downgrades the stop to a warning, for
deliberately re-probing on a newer CellProfiler -- the printed versions are then what
the run is provenance for, and RECORDED_VERSIONS and the audit report are what need
updating.
"""
import argparse
import math
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.normpath(os.path.join(HERE, "..", ".."))

NBINS = 8
PAD = 3          # keeps the ROI off the label-image border, which CP treats as an edge
FEATURES = ("FRAC_AT_D", "MEAN_FRAC", "RADIAL_CV")

# The versions the docstring above and the audit report claim these numbers came from. Read back
# from the installed distributions on every run by check_versions(), so a report that says 4.2.8
# has to have been produced by 4.2.8.
RECORDED_VERSIONS = {
    "cellprofiler": "4.2.8",
    "cellprofiler-core": "4.2.8.1",
    "centrosome": "1.2.3",
    "numpy": "1.26.4",
    "scipy": "1.10.1",
}

# SPEC 7's band for a cross-tool comparison whose definitional edge differences (the binning rule,
# the centre rule) are documented: rel=1e-2. This is the PROMOTION gate and it is applied PER
# FEATURE at the recipe being run -- a feature is promotable when every one of its bins meets its
# band, whatever the other two features do. One counter over all 24 cells would only fire when the
# whole family agreed, and would leave a promotable row marked regression.
SPEC_BAND = {f: 1e-2 for f in FEATURES}

# Descriptive only: the cutoff the report's "N of the 24 disagree by more than 1%" line counts at.
# Nothing is promoted or rejected on it.
REPORT_CUTOFF = 1e-2


# ---------------------------------------------------------------------------- parsing

def _read(relpath):
    with open(os.path.join(TESTS, relpath), "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def _braced_body(src, start):
    """The text between the brace at/after `start` and its matching close.

    Counts braces rather than matching with a non-greedy regex, which eats the last
    entry's closing brace and silently drops it.
    """
    i = src.index("{", start)
    depth, j = 0, i
    while j < len(src):
        if src[j] == "{":
            depth += 1
        elif src[j] == "}":
            depth -= 1
            if depth == 0:
                return src[i + 1:j]
        j += 1
    raise SystemExit("unbalanced braces while parsing a reference table")


def parse_fixture():
    """The {x, y, value} triples of the two shape2d arrays in test_data.h."""
    src = _read("test_data.h")
    out = {}
    for name in ("shape2d_morphology_intensity", "shape2d_morphology_mask"):
        m = re.search(r"\b%s\s*\[\s*\]\s*=\s*\{" % name, src)
        if not m:
            raise SystemExit("cannot find %s in test_data.h" % name)
        body = _braced_body(src, m.end() - 1)
        out[name] = [tuple(int(v) for v in t)
                     for t in re.findall(r"\{\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\}", body)]
    return out


def parse_pins():
    """The three 8-entry vectors pinned in test_2d_radial_regression.h."""
    src = _read("test_2d_radial_regression.h")
    m = re.search(r"radial_2d_regression_ref_vals\s*\{", src)
    if not m:
        raise SystemExit("cannot find radial_2d_regression_ref_vals in the regression header")
    body = _braced_body(src, m.end() - 1)
    pins = {}
    for name, values in re.findall(r'\{\s*"(\w+)"\s*,\s*\{([^}]*)\}', body):
        pins[name] = [float(v) for v in re.findall(r"[-+0-9.eE]+", values)]
    return pins


def parse_mechanics():
    """The centre pixel and the squared normalising radius pinned in the mechanics header."""
    src = _read("test_2d_radial_mechanics.h")
    # the closing paren, not the semicolon, ends the match: these assertions carry a streamed
    # message naming the defect they pin, so anchoring on ");" stops finding them
    cx = re.search(r"ASSERT_EQ\(pxO\.x,\s*(\d+)\)", src)
    cy = re.search(r"ASSERT_EQ\(pxO\.y,\s*(\d+)\)", src)
    r2 = re.search(r"ASSERT_DOUBLE_EQ\(pxO\.max_sqdist\(K\),\s*([0-9.]+)\)", src)
    if not (cx and cy and r2):
        raise SystemExit("cannot find the centre / radius pins in test_2d_radial_mechanics.h")
    return int(cx.group(1)), int(cy.group(1)), float(r2.group(1))


# ------------------------------------------------------------------- the Nyxus algorithm

def nyxus_model(pixels, cx, cy, r_max, n=NBINS, epsilon=1e-9):
    """RadialDistributionFeature as implemented, written out in numpy-free python.

    Follows src/nyx/features/radial_distribution.cpp: the bin index is scaled by n-1 and
    clamped, FRAC_AT_D is a pixel-count fraction, MEAN_FRAC is a raw bin mean intensity,
    and RADIAL_CV is the population CV of the eight wedge SUMS including empty wedges.
    The centre and r_max are inputs because the feature obtains them from two approximate
    searches over a concatenated contour, which no clean reimplementation reproduces.
    """
    cnt = [0] * n
    isum = [0.0] * n
    wedges = [[0.0] * n for _ in range(n)]
    for x, y, inten in pixels:
        d = math.hypot(x - cx, y - cy)
        bi = int(d / r_max * (n - 1))
        if bi >= n:
            bi = n - 1
        cnt[bi] += 1
        isum[bi] += inten
        ang = math.atan2(y - cy, x - cx)
        if ang < 0:
            ang += 2.0 * math.pi
        wedges[bi][int(ang / (2.0 * math.pi / n))] += inten

    total = float(len(pixels))
    frac = [cnt[i] / (total + epsilon) for i in range(n)]
    mean = [isum[i] / (cnt[i] + epsilon) for i in range(n)]
    cv = []
    for i in range(n):
        mu = sum(wedges[i]) / n
        var = sum((w - mu) ** 2 for w in wedges[i]) / n
        cv.append(math.sqrt(var) / (mu + epsilon))
    return {"FRAC_AT_D": frac, "MEAN_FRAC": mean, "RADIAL_CV": cv}


# ------------------------------------------------------------------------- CellProfiler

def center_tie_set(pixels, grid):
    """The pixels tied for "farthest from the edge", which is how CellProfiler picks its centre.

    Reported because on this fixture the maximum is not unique: which of the tied pixels
    scipy.ndimage.maximum_position returns depends on the label image's shape, so CP's own
    answer moves when the padding around the ROI changes. A tie-free ROI is a precondition
    for this fixture ever vetting the family, independently of the divergences below.
    """
    import numpy as np
    from scipy import ndimage
    lab = np.zeros((grid + 2 * PAD, grid + 2 * PAD), dtype=bool)
    for x, y, _ in pixels:
        lab[y + PAD, x + PAD] = True
    d = ndimage.distance_transform_edt(lab)
    top = d[lab].max()
    return top, sorted((int(j) - PAD, int(i) - PAD)
                       for i, j in zip(*np.where(lab & (d == top))))


def cellprofiler_independent(pixels, grid, n=NBINS):
    """CellProfiler's three measurements rebuilt from the published definitions.

    numpy/scipy only, no centrosome in the path: the distance to the edge is a Euclidean
    distance transform, the distance from the centre is a Dijkstra propagation inside the
    mask (what propagate(weight=1) computes), then the scaled radius, the bin index and the
    three statistics. The one step taken from the tool is which of the tied
    maximum-distance-to-edge pixels is the centre -- the maximum is not unique on this
    fixture (see the tie report above), so an independent tie-break would compare two
    different ROIs rather than two implementations.
    """
    import heapq
    import numpy as np
    from scipy import ndimage

    side = grid + 2 * PAD
    lab = np.zeros((side, side), dtype=bool)
    img = np.zeros((side, side), dtype=float)
    for x, y, inten in pixels:
        lab[y + PAD, x + PAD] = True
        img[y + PAD, x + PAD] = inten / 255.0

    d_to_edge = ndimage.distance_transform_edt(lab)
    ci, cj = (int(v) for v in ndimage.maximum_position(d_to_edge, lab.astype(int), [1])[0])

    INF = float("inf")
    d_from_center = np.full(lab.shape, INF)
    d_from_center[ci, cj] = 0.0
    pq = [(0.0, ci, cj)]
    while pq:
        d, i, j = heapq.heappop(pq)
        if d > d_from_center[i, j]:
            continue
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                ni, nj = i + di, j + dj
                if not (0 <= ni < side and 0 <= nj < side) or not lab[ni, nj]:
                    continue
                nd = d + math.hypot(di, dj)
                if nd < d_from_center[ni, nj]:
                    d_from_center[ni, nj] = nd
                    heapq.heappush(pq, (nd, ni, nj))
    d_from_center[~lab] = 0.0

    nd = np.zeros(lab.shape)
    nd[lab] = d_from_center[lab] / (d_from_center[lab] + d_to_edge[lab] + 0.001)
    bidx = (nd * n).astype(int)
    bidx[bidx > n] = n

    inten_by_bin = np.zeros(n)
    cnt_by_bin = np.zeros(n)
    oct_sum = np.zeros((n, 8))
    oct_cnt = np.zeros((n, 8))
    for i, j in zip(*np.where(lab)):
        b = bidx[i, j]
        inten_by_bin[b] += img[i, j]
        cnt_by_bin[b] += 1
        di, dj = i - ci, j - cj
        oct_sum[b, int(di > 0) + 2 * int(dj > 0) + 4 * int(abs(di) > abs(dj))] += img[i, j]
        oct_cnt[b, int(di > 0) + 2 * int(dj > 0) + 4 * int(abs(di) > abs(dj))] += 1

    frac = inten_by_bin / inten_by_bin.sum()
    meanfrac = frac / (cnt_by_bin / cnt_by_bin.sum() + np.finfo(float).eps)
    cv = np.zeros(n)
    for b in range(n):
        nz = oct_cnt[b] > 0
        if nz.sum():
            means = oct_sum[b][nz] / oct_cnt[b][nz]
            cv[b] = means.std() / means.mean()
    return {"FRAC_AT_D": list(frac), "MEAN_FRAC": list(meanfrac), "RADIAL_CV": list(cv)}


def installed_versions():
    """-> {distribution: version or None} for the packages the provenance line names."""
    try:
        from importlib.metadata import PackageNotFoundError, version   # py3.8+
    except ImportError:                                                # pragma: no cover
        from importlib_metadata import PackageNotFoundError, version
    out = {}
    for dist in RECORDED_VERSIONS:
        try:
            out[dist] = version(dist)
        except PackageNotFoundError:
            out[dist] = None
    return out


def check_versions(allow_drift):
    """Print what is actually installed, and refuse to pass its output off as another version.

    The docstring above and the audit report both name a CellProfiler version; nothing until now
    read the one that is actually importable, so a later environment could produce a run still
    presented as 4.2.8. Returns True if this run may be presented as RECORDED_VERSIONS. A missing
    distribution counts as a mismatch: an absent cellprofiler-core is not evidence that 4.2.8.1
    produced anything.
    """
    got = installed_versions()
    print("=== the CellProfiler environment this run is provenance for ===")
    bad = []
    for dist, recorded in RECORDED_VERSIONS.items():
        have = got[dist]
        same = have == recorded
        print("  %-18s recorded=%-8s installed=%-8s %s"
              % (dist, recorded, have or "MISSING", "" if same else "<-- MISMATCH"))
        if not same:
            bad.append("%s: recorded %s, installed %s" % (dist, recorded, have or "MISSING"))

    if not bad:
        print("  the run matches the recorded provenance")
        return True

    if allow_drift:
        print()
        print("  WARNING: %d version mismatch(es), continuing on --allow-version-drift." % len(bad))
        print("  The numbers below are THIS environment's and not the recorded provenance. Update")
        print("  RECORDED_VERSIONS and radial_2d_cellprofiler_vetting_report.md before quoting them.")
        return False

    print()
    print("  REFUSING to run: %s." % "; ".join(bad))
    print("  The audit report quotes this script's output as cellprofiler %s / cellprofiler-core %s."
          % (RECORDED_VERSIONS["cellprofiler"], RECORDED_VERSIONS["cellprofiler-core"]))
    print("  Install that environment (tests/vetting/TOOLS.md), or pass --allow-version-drift to")
    print("  re-probe deliberately.")
    raise SystemExit(2)


def run_cellprofiler(pixels, grid):
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
    from cellprofiler.modules import measureobjectintensitydistribution as moid

    lab = np.zeros((grid + 2 * PAD, grid + 2 * PAD), dtype=np.int32)
    img = np.zeros((grid + 2 * PAD, grid + 2 * PAD), dtype=float)
    for x, y, inten in pixels:
        lab[y + PAD, x + PAD] = 1
        img[y + PAD, x + PAD] = inten / 255.0   # CP images live in [0,1]; all three
                                                # measurements are scale-invariant ratios

    module = moid.MeasureObjectIntensityDistribution()
    module.images_list.value = "img"
    module.objects[0].object_name.value = "objs"
    module.bin_counts[0].bin_count.value = NBINS
    module.bin_counts[0].wants_scaled.value = True
    module.wants_zernikes.value = moid.Z_NONE

    objects = cpo.Objects()
    objects.segmented = lab
    oset = cpo.ObjectSet()
    oset.add_objects(objects, "objs")
    imgset = cpi.ImageSetList().get_image_set(0)
    imgset.add("img", cpi.Image(img))
    m = cpmeas.Measurements()
    module.run(cpw.Workspace(cpp.Pipeline(), module, imgset, oset, m, None))

    cp_name = {"FRAC_AT_D": "FracAtD", "MEAN_FRAC": "MeanFrac", "RADIAL_CV": "RadialCV"}
    return {f: [m.get_current_measurement(
                    "objs", "RadialDistribution_%s_img_%dof%d" % (cp_name[f], b, NBINS))[0]
                for b in range(1, NBINS + 1)]
            for f in FEATURES}


# -------------------------------------------------------------------------------- main

def pin_checks(pins, n_pixels, total_intensity, lo, hi, n=NBINS):
    """Range and identity checks run over the PINNED LITERALS, not over a fresh run.

    The C++ tests assert the same properties of the computed values, split by whether they survive
    a change of definition: the fraction bounds, the empty-bin zeros and the CV bound live in
    test_2d_radial_invariant.h, while the whole-pixel-count, raw-intensity-range and reconstruction
    checks hold only under Nyxus' current conventions and live in test_2d_radial_regression.h. This
    runs the whole set against what the header actually says, so an edit to the table is caught
    without a rebuild. Returns a list of failure strings.
    """
    bad = []
    frac, mean, cv = pins["FRAC_AT_D"], pins["MEAN_FRAC"], pins["RADIAL_CV"]
    bound = math.sqrt(n - 1)          # population CV of n non-negative values

    for i in range(n):
        if not 0.0 <= frac[i] <= 1.0:
            bad.append("FRAC_AT_D[%d]=%r is not a fraction" % (i, frac[i]))
        count = frac[i] * n_pixels
        if abs(count - round(count)) > 1e-6:
            bad.append("FRAC_AT_D[%d] is %g pixels, not a whole count" % (i, count))
        if cv[i] < 0.0 or cv[i] > bound * (1 + 1e-9):
            bad.append("RADIAL_CV[%d]=%r is outside [0, sqrt(%d)]" % (i, cv[i], n - 1))
        if frac[i] == 0.0:
            if mean[i] != 0.0 or cv[i] != 0.0:
                bad.append("bin %d is empty in FRAC_AT_D but not in MEAN_FRAC/RADIAL_CV" % i)
        elif not lo <= mean[i] <= hi:
            bad.append("MEAN_FRAC[%d]=%r is outside the ROI intensity range [%g, %g]"
                       % (i, mean[i], lo, hi))

    # the feature's epsilon guard in the denominator is why this is not exactly 1
    if abs(sum(frac) - 1.0) > 1e-9:
        bad.append("FRAC_AT_D sums to %.17g, not 1" % sum(frac))

    # the two intensity features must put the ROI's total intensity back together
    got = sum(round(frac[i] * n_pixels) * mean[i] for i in range(n))
    if rel(got, total_intensity) > 1e-9:
        bad.append("FRAC_AT_D and MEAN_FRAC reconstruct %.17g, not the ROI total %g"
                   % (got, total_intensity))
    return bad


def rel(a, b):
    if a == b:
        return 0.0
    if b == 0.0:
        return float("inf")
    return abs(a - b) / abs(b)


def agrees(got, golden, band):
    """Does one (feature, bin) cell meet its band? Two exact zeros agree; one zero never does.

    rel() already returns inf when only the golden is zero, so the band alone would reject that
    case; the explicit both-zero arm is what keeps a bin both tools leave empty from being counted
    as a disagreement.
    """
    if got == 0.0 and golden == 0.0:
        return True
    return rel(got, golden) <= band


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-cellprofiler", action="store_true",
                    help="verify the pins only; skip the part that needs the CP env")
    ap.add_argument("--allow-version-drift", action="store_true",
                    help="run against a CellProfiler env other than the recorded one, and say so")
    args = ap.parse_args()

    fixture = parse_fixture()
    pins = parse_pins()
    cx, cy, r2 = parse_mechanics()
    r_max = math.sqrt(r2)

    inten = {(x, y): v for x, y, v in fixture["shape2d_morphology_intensity"]}
    pixels = [(x, y, inten[(x, y)]) for x, y, v in fixture["shape2d_morphology_mask"] if v]
    # the fixture's own frame, not the ROI's bounding box: CP measures distance to the edge
    # on the label image, so the frame has to be fixed by the fixture and not by the ROI
    grid = 1 + max(max(x for x, _, _ in fixture["shape2d_morphology_intensity"]),
                   max(y for _, y, _ in fixture["shape2d_morphology_intensity"]))
    print("fixture: %d ROI pixels in a %dx%d grid, total intensity %d"
          % (len(pixels), grid, grid, sum(p[2] for p in pixels)))
    print("mechanics pins: centre=(%d,%d) r_max=sqrt(%g)=%.17g\n" % (cx, cy, r2, r_max))

    model = nyxus_model(pixels, cx, cy, r_max)
    ok = True

    # 1. every pinned value must come back out of the written-down algorithm
    print("=== the pinned goldens against an independent model of the implementation ===")
    for f in FEATURES:
        if f not in pins:
            print("  FAIL %s: the header pins nothing for it" % f)
            ok = False
            continue
        if len(pins[f]) != NBINS:
            print("  FAIL %s: header pins %d bins, expected %d" % (f, len(pins[f]), NBINS))
            ok = False
            continue
        for i, (golden, got) in enumerate(zip(pins[f], model[f])):
            good = got == golden
            ok &= good
            print("  %s %s[%d] header=%.17g model=%.17g rel=%.3g"
                  % ("OK  " if good else "FAIL", f, i, golden, got, rel(got, golden)))

    # 2. the reverse direction -- a feature the model produces that nothing pins
    for f in model:
        if f not in pins:
            print("  FAIL the model produces %s and the header pins nothing for it" % f)
            ok = False

    # 3. range and identity checks over the pinned literals themselves
    intensities = [p[2] for p in pixels]
    bad = pin_checks(pins, len(pixels), sum(intensities), min(intensities), max(intensities))
    print("\n=== range and identity checks over the pinned values ===")
    if bad:
        for b in bad:
            print("  FAIL", b)
        ok = False
    else:
        print("  all %d pins in range, empty bins consistent across the three tables,"
              % (3 * NBINS))
        print("  FRAC_AT_D a partition of whole pixel counts, and the two intensity tables")
        print("  reconstruct the ROI total")

    if args.skip_cellprofiler:
        print("\n(skipping the CellProfiler run on request)")
        print("\n%s" % ("ALL PIN CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
        return 0 if ok else 1

    # 4. CellProfiler, on the same fixture, at the recipe radial.cellprofiler_8bin
    provenance_ok = check_versions(args.allow_version_drift)

    top, tied = center_tie_set(pixels, grid)
    print("\n=== CellProfiler's centre rule on this ROI ===")
    print("  max distance-to-edge = %.17g, attained by %d pixels: %s"
          % (top, len(tied), ", ".join("(%d,%d)" % t for t in tied)))
    if len(tied) > 1:
        print("  the centre is NOT unique, so CellProfiler's own answer is tie-break dependent")

    cp = run_cellprofiler(pixels, grid)

    # the tool's own output against the same definitions rebuilt from scratch, so the table
    # below is CellProfiler's answer and not an artefact of how the module was driven
    indep = cellprofiler_independent(pixels, grid)
    worst = max(rel(indep[f][i], cp[f][i]) for f in FEATURES for i in range(NBINS)
                if not (indep[f][i] == 0.0 and cp[f][i] == 0.0))
    print("\n=== the CellProfiler run against an independent numpy/scipy rebuild ===")
    print("  worst relative difference over all %d values: %.3g" % (3 * NBINS, worst))
    if worst > 1e-6:
        print("  FAIL the rebuild does not reproduce the module; the run is not trustworthy")
        ok = False

    print()
    print("=== CellProfiler MeasureObjectIntensityDistribution vs the pinned goldens ===")
    print("  %-10s %3s %22s %22s %10s" % ("feature", "bin", "nyxus", "cellprofiler", "rel"))
    diverging = 0
    for f in FEATURES:
        for i in range(NBINS):
            r = rel(cp[f][i], pins[f][i])
            if r > REPORT_CUTOFF:
                diverging += 1
            print("  %-10s %3d %22.17g %22.17g %10.3g" % (f, i, pins[f][i], cp[f][i], r))

    print()
    print("  %d of the %d (feature x bin) values disagree by more than %g%% -- descriptive only."
          % (diverging, 3 * NBINS, REPORT_CUTOFF * 100))

    # 5. the promotion verdict, decided per (feature x config) rather than once for the family
    print()
    print("=== promotion verdict per feature, at recipe radial.cellprofiler_8bin ===")
    promotable = []
    for f in FEATURES:
        band = SPEC_BAND[f]
        outside = [i for i in range(NBINS) if not agrees(cp[f][i], pins[f][i], band)]
        worst = max(rel(cp[f][i], pins[f][i]) for i in range(NBINS))
        print("  %-10s band rel<=%-8g worst rel=%-10.3g %d/%d bins outside%s"
              % (f, band, worst, len(outside), NBINS,
                 "" if not outside else "  (" + ", ".join(str(i) for i in outside) + ")"))
        if not outside:
            promotable.append(f)

    if promotable:
        print()
        print("  UNEXPECTED: every bin of %s meets its SPEC 7 band against CellProfiler,"
              % ", ".join(promotable))
        print("  so %s promotable at this recipe. The divergence record in"
              % ("that row is" if len(promotable) == 1 else "those rows are"))
        print("  tests/vetting/audit/radial_2d_cellprofiler_vetting_report.md is stale for it --")
        print("  do not ship this verdict unchanged.")
        ok = False
    else:
        print()
        print("  No feature meets its band on every bin, so none of the three rows is promotable at")
        print("  this recipe. That is the recorded outcome: the family is NOT CellProfiler-vetted.")

    if not provenance_ok:
        print()
        print("  NOTE: this environment did not match the recorded provenance printed above.")

    print()
    print("%s" % ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED -- do not ship"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
