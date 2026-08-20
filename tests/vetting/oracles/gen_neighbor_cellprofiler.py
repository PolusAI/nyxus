"""OFFLINE CellProfiler oracle for the 2D neighbor graph/distance features
(SPEC 4, oracle=cellprofiler). Runs the real cellprofiler.modules.
MeasureObjectNeighbors module on the gtest fixture `neighborhood2d_scene_labels`
(test_data.h) and re-verifies EVERY golden pinned in test_2d_neighbor_cellprofiler.h against that
run, exiting non-zero on any mismatch or on any pin this generator cannot produce. The pins are
parsed out of the header: a validation list kept here would only ever compare this script against
itself, so a hand-edited header would go unnoticed.

CellProfiler reproduces Nyxus BIT-IDENTICALLY on both vetted features across all five ROIs --
residual exactly 0 -- which is what lets the header assert at the SPEC 7 exact tier.

Result of this comparison (see the printed table):
  NUM_NEIGHBORS           -- CP == Nyxus  (VETS it)
  CLOSEST_NEIGHBOR1_DIST  -- CP == Nyxus  (VETS it)
  CLOSEST_NEIGHBOR2_DIST  -- CP != Nyxus by DEFINITION: CP reports the 2nd-closest
      of ANY object (global); Nyxus reports the 2nd-closest *neighbor within the
      search radius*, i.e. 0 when an ROI has <2 in-radius neighbors. Not CP-vettable;
      vetted analytically instead (gen_neighbor_analytic.py).
  PERCENT_TOUCHING        -- CP != Nyxus (convention gap, 3/5 ROIs). Nyxus counts
      contour pixels 8-adjacent to a neighbor / contour length; CP counts object
      outline pixels overlapping a disk(distance+0.5)-dilated neighbor / perimeter.
      No CP distance method (Adjacent/Expand/Within) reproduces Nyxus. Stays
      regression with a documented convention divergence.

Environment: a dedicated CellProfiler env is required to RUN this generator
(cellprofiler-core + centrosome + the cellprofiler module package, headless).
CI never invokes it -- CellProfiler is not a runtime dependency.

Provenance: tool=cellprofiler, version=4.2.8 (module package) / cellprofiler-core
4.2.8.1, centrosome 1.2.3; python 3.9; MeasureObjectNeighbors, distance_method=
Adjacent, neighbors_are_objects=True. generator=tests/vetting/oracles/
gen_neighbor_cellprofiler.py. Run offline.
"""
import os
import re
import warnings
warnings.filterwarnings("ignore")
import numpy as np

import cellprofiler_core.preferences as cpprefs
cpprefs.set_headless()
import cellprofiler_core.object as cpo
import cellprofiler_core.measurement as cpmeas
import cellprofiler_core.workspace as cpw
import cellprofiler_core.pipeline as cpp
from cellprofiler.modules import measureobjectneighbors as mon

# fixture neighborhood2d_scene_labels (tests/test_data.h): {x, y, label}
SCENE = [
    (4, 2, 3), (5, 2, 3), (4, 3, 3), (5, 3, 3),
    (2, 4, 2), (3, 4, 2), (4, 4, 1), (5, 4, 1), (6, 4, 1), (7, 4, 4), (8, 4, 4),
    (2, 5, 2), (3, 5, 2), (4, 5, 1), (5, 5, 1), (6, 5, 1), (7, 5, 4), (8, 5, 4),
    (4, 6, 1), (5, 6, 1), (6, 6, 1), (7, 6, 4), (8, 6, 4),
    (5, 7, 5), (6, 7, 5), (5, 8, 5), (6, 8, 5),
]
PAD = 3  # keep every ROI off the image border (CP border handling)
LABELS = [1, 2, 3, 4, 5]

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.dirname(os.path.dirname(HERE))
TEST_H = os.path.join(TESTS, "test_2d_neighbor_cellprofiler.h")
TABLE = "neighbor_2d_cellprofiler_ref_vals_by_label"
REGRESSION_H = os.path.join(TESTS, "test_2d_neighbor_regression.h")
REGRESSION_TABLE = "neighbor_2d_regression_ref_vals_by_label"

# the two features CP and Nyxus compute the same way; the header pins CP's own digits for them
CP_VETTED = ("NUM_NEIGHBORS", "CLOSEST_NEIGHBOR1_DIST")

RELTOL = 1e-9  # SPEC 7 exact tier; the measured residual is exactly 0


def _label_blocks(body):
    """Yields (label, block text) by matching braces, not by a non-greedy regex.

    A regex that stops at the first `}}` swallows the closing brace of the block's last entry, so a
    label block written on a single line silently loses every entry after the first. Counting braces
    is layout-independent.
    """
    i = 0
    while True:
        m = re.compile(r"\{\s*(\d+)\s*,\s*\{").search(body, i)
        if not m:
            return
        label = int(m.group(1))
        depth = 1                      # we are just past the inner '{'
        j = m.end()
        while j < len(body) and depth:
            if body[j] == "{":
                depth += 1
            elif body[j] == "}":
                depth -= 1
            j += 1
        yield label, body[m.end():j - 1]
        i = j


def parse_pins(path, table):
    """The header's own table, keyed {label: {feature: value}}."""
    txt = open(path, encoding="utf-8", errors="replace").read()
    m = re.search(re.escape(table) + r"\s*\{", txt)
    if not m:
        raise RuntimeError("table %s not found in %s" % (table, os.path.basename(path)))
    body = txt[m.end():].split("\n};", 1)[0]
    body = re.sub(r"//[^\n]*", "", body)          # a commented-out golden is not a pin
    pins = {}
    for label, block in _label_blocks(body):
        pins[label] = {n: float(v) for n, v in
                       re.findall(r'\{\s*"(\w+)"\s*,\s*([-0-9.eE+]+)\s*\}', block)}
    return pins


def build_labels():
    maxx = max(x for x, _, _ in SCENE)
    maxy = max(y for _, y, _ in SCENE)
    lab = np.zeros((maxy + 1 + 2 * PAD, maxx + 1 + 2 * PAD), dtype=np.int32)
    for x, y, l in SCENE:
        lab[y + PAD, x + PAD] = l  # CP indexes [row=y, col=x]
    return lab


def run_cp(method):
    module = mon.MeasureObjectNeighbors()
    module.object_name.value = "objs"
    module.neighbors_name.value = "objs"
    module.distance_method.value = method
    if method == mon.D_WITHIN:
        module.distance.value = 1

    objects = cpo.Objects()
    objects.segmented = build_labels()
    oset = cpo.ObjectSet()
    oset.add_objects(objects, "objs")

    m = cpmeas.Measurements()
    ws = cpw.Workspace(cpp.Pipeline(), module, m, oset, m, None)
    module.run(ws)

    scale = mon.S_ADJACENT if method == mon.D_ADJACENT else (
        mon.S_EXPANDED if method == mon.D_EXPAND else "1")
    get = lambda name: m.get_current_measurement("objs", "Neighbors_%s_%s" % (name, scale))
    return {
        "NUM_NEIGHBORS": get(mon.M_NUMBER_OF_NEIGHBORS),
        "CLOSEST_NEIGHBOR1_DIST": get(mon.M_FIRST_CLOSEST_DISTANCE),
        "CLOSEST_NEIGHBOR2_DIST": get(mon.M_SECOND_CLOSEST_DISTANCE),
        "PERCENT_TOUCHING": get(mon.M_PERCENT_TOUCHING),
    }


def main():
    cp = run_cp(mon.D_ADJACENT)

    print("# CellProfiler MeasureObjectNeighbors (Adjacent), recipe neighbor.scene2d_radius1")
    print("# paste-ready goldens")
    for l in LABELS:
        vals = ", ".join('{"%s", %r}' % (f, float(cp[f][l - 1])) for f in CP_VETTED)
        print("\t{%d, {%s}}," % (l, vals))

    pins = parse_pins(TEST_H, TABLE)
    print("")
    n = sum(len(v) for v in pins.values())
    print("# verifying %d pinned goldens against this run" % n)
    nok = nfail = nmiss = 0
    for l in sorted(pins):
        for f in sorted(pins[l]):
            want = pins[l][f]
            if f not in cp:
                print("  MISSING L%d %s: pinned %r but CP produces no such column" % (l, f, want))
                nmiss += 1
                continue
            have = float(cp[f][l - 1])
            rel = abs(have - want) / max(abs(want), 1.0)
            if rel <= RELTOL:
                print("  OK   L%d %s: cp=%r pinned=%r rel=%.3g" % (l, f, have, want, rel))
                nok += 1
            else:
                print("  FAIL L%d %s: cp=%r pinned=%r rel=%.3g" % (l, f, have, want, rel))
                nfail += 1

    # every feature CP vets must be pinned on every label, or the header quietly lost coverage
    missing = ["L%d %s" % (l, f) for l in LABELS for f in CP_VETTED if f not in pins.get(l, {})]
    for gap in missing:
        print("  UNPINNED %s: CP vets it but the header pins nothing" % gap)

    print("")
    print("# documented divergences -- NOT vetted against CP, recorded so the gap stays measured")
    reg = parse_pins(REGRESSION_H, REGRESSION_TABLE)
    print("  %2s %9s %9s %8s" % ("L", "PT_cp", "PT_nyxus", "D2_cp"))
    pt_div = 0
    for l in LABELS:
        pt_cp = float(cp["PERCENT_TOUCHING"][l - 1])
        pt_ny = reg.get(l, {}).get("PERCENT_TOUCHING", float("nan"))
        if abs(pt_cp - pt_ny) > 1e-9 * max(abs(pt_ny), 1.0):
            pt_div += 1
        print("  %2d %9.4f %9.4f %8.4f"
              % (l, pt_cp, pt_ny, float(cp["CLOSEST_NEIGHBOR2_DIST"][l - 1])))
    print("  PERCENT_TOUCHING diverges on %d/5 ROIs -> stays regression (convention gap)" % pt_div)

    print("")
    print("%d verified, %d failed, %d unproducible, %d unpinned" % (nok, nfail, nmiss, len(missing)))
    if nfail or nmiss or missing:
        print("SOME CHECKS FAILED -- do not promote")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
