"""OFFLINE CellProfiler oracle for the 2D neighbor graph/distance features
(SPEC 4, oracle=cellprofiler). Runs the real cellprofiler.modules.
MeasureObjectNeighbors module on the gtest fixture `neighborhood2d_scene_labels`
(test_data.h) and validates it against the goldens pinned in
test_neighbor_cellprofiler.h.

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

# goldens the CP run must reproduce (pinned in test_neighbor_cellprofiler.h == Nyxus)
CP_VETS = {  # feature -> {label: value}  (features where CP == Nyxus)
    "NUM_NEIGHBORS": {1: 4, 2: 1, 3: 1, 4: 1, 5: 1},
    "CLOSEST_NEIGHBOR1_DIST": {1: 2.5, 2: 2.54950975679639, 3: 2.54950975679639,
                               4: 2.5, 5: 2.54950975679639},
}
# Nyxus values for the features where CP disagrees (for the divergence record)
NYXUS_PT = {1: 100.0, 2: 66.6666666666667, 3: 66.6666666666667, 4: 50.0, 5: 33.3333333333333}
TOL = 1e-4


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
    all_ok = True

    print("=== CellProfiler MeasureObjectNeighbors (Adjacent) vs Nyxus goldens ===")
    for feat, golds in CP_VETS.items():
        for l in LABELS:
            got = cp[feat][l - 1]
            exp = golds[l]
            ok = abs(got - exp) <= TOL * max(1.0, abs(exp))
            all_ok &= ok
            print(f"  {'OK ' if ok else 'FAIL'} L{l} {feat}: CP={got!r} nyxus={exp!r}")

    print("\n=== documented divergences (NOT vetted vs CP) ===")
    print(f"  {'L':>2} {'PT_cp':>9} {'PT_nyxus':>9} {'D2_cp':>8} {'D2_nyxus':>9}")
    for l in LABELS:
        print(f"  {l:>2} {cp['PERCENT_TOUCHING'][l-1]:>9.4f} {NYXUS_PT[l]:>9.4f} "
              f"{cp['CLOSEST_NEIGHBOR2_DIST'][l-1]:>8.4f} "
              f"{'2.5495' if l==1 else '0.0000':>9}")
    pt_div = sum(1 for l in LABELS if abs(cp['PERCENT_TOUCHING'][l-1] - NYXUS_PT[l]) > TOL)
    print(f"  PERCENT_TOUCHING diverges on {pt_div}/5 ROIs -> stays regression (convention gap)")

    print(f"\n{'ALL CP-VET CHECKS PASSED' if all_ok else 'SOME CHECKS FAILED -- do not promote'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
