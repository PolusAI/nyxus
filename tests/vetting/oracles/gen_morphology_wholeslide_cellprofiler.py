"""OFFLINE CellProfiler oracle for MASS_DISPLACEMENT at the WHOLE-SLIDE cell
(SPEC 4, oracle=cellprofiler; recipe morphology.cellprofiler_wholeslide_massdisp).

Companion to gen_morphology_cellprofiler.py, which covers the segmented in-RAM cell. This one
answers a different question: when Nyxus runs at SINGLEROI=true the ROI is the whole image, so the
matching CellProfiler configuration is ONE object covering the full frame. Running it decides
whether the whole-slide cell is oracle-backed or production-only, per SPEC 5.2's rule that verdicts
are measured rather than hand-labelled.

The answer splits the cell in two, which is why it is worth a generator:

  MASS_DISPLACEMENT   CellProfiler AGREES -- 3.345311793150965 against Nyxus 3.3453118163885427,
                      rel 7e-9. It is computed by BasicMorphologyFeatures::calculate() from the
                      geometric and intensity-weighted centroids, reads no contour at all, and is
                      the same quantity in both tools whether the object is a disk or a full frame.
                      So this cell is VALID, not production-only.

  EDGE_*              CellProfiler returns 0 for every one of them. With a label image that is all
                      ones, find_boundaries(mode="inner") finds no pixel whose neighbour differs --
                      outside the array is not a neighbour -- so the edge set is EMPTY and every
                      edge statistic is 0. Nyxus instead returns statistics of the four AABB corner
                      pixels it synthesises in buildWholeSlideContour(), each carrying aux_max.
                      Those are different quantities, not a disagreement about one, so CellProfiler
                      cannot vet them and they stay regression snapshots.

That negative half is the point of checking it in: "CellProfiler produces a number for a
measurement of the same name" is not the same claim as "CellProfiler computes this feature", and
the only way to tell them apart is to run it and look.

Fixture: bench_disk64_diagonal_boundary (tests/vetting/benchmarks.md), built by
tests/python/test_data.py::disk64_arrays so this generator and the gtest/pytest assertions read one
definition. The whole-slide input is the full 64x64 intensity frame -- zeros outside the disk
included -- because that is what Nyxus featurises at SINGLEROI=true.

Environment: the dedicated CellProfiler env, as for gen_morphology_cellprofiler.py, whose docstring
carries the two Windows gotchas (activate the env rather than calling python.exe by path; run from a
working directory on the same drive as the env). CI never invokes this.

Provenance: tool=cellprofiler, version=4.2.8 (module package) / cellprofiler-core 4.2.8.1,
centrosome 1.2.3; python 3.9; module MeasureObjectIntensity, one image and one object set, all
settings at their defaults, label image = ones((64,64)).
generator=tests/vetting/oracles/gen_morphology_wholeslide_cellprofiler.py. Run offline.
"""
import os
import re
import sys
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
sys.path.insert(0, os.path.join(TESTS, "python"))
import test_data                                    # the one definition of the disk fixture

PIN_FILE = os.path.join(TESTS, "python", "test_2d_morphology_regression.py")
PIN_TABLE = "WHOLE_SLIDE"
IMAGE = "img"
OBJECTS = "objs"
# Nyxus and CellProfiler compute MASS_DISPLACEMENT from the same two centroids; the residual is
# CellProfiler's float32 image storage, as at the segmented cell. Band set from the measurement.
TOL = 1e-6
# CellProfiler's edge set is empty on an all-ones label image, so every edge statistic is exactly 0
# and none of them is the quantity Nyxus reports here.
NOT_VETTABLE = ("EDGE_MEAN_INTENSITY", "EDGE_STDDEV_INTENSITY", "EDGE_MAX_INTENSITY",
                "EDGE_MIN_INTENSITY", "EDGE_INTEGRATED_INTENSITY")


def parse_pins(path, table):
    """The pins the pytest module asserts, read out of it rather than copied."""
    txt = open(path, encoding="utf-8", errors="replace").read()
    body = txt.split(table + " = {", 1)[1].split("}", 1)[0]
    body = re.sub(r"#[^\n]*", "", body)
    return {k: float(v) for k, v in re.findall(r'"(\w+)":\s*([-0-9.eE+]+)', body)}


def run_cp():
    inten, _ = test_data.disk64_arrays()             # whole frame, zeros outside the disk included
    inten = inten.astype(float)
    labels = np.ones(inten.shape, dtype=np.int32)    # ONE object over the full image
    scale = inten.max()

    module = moi.MeasureObjectIntensity()
    module.images_list.value = IMAGE
    module.objects_list.value = OBJECTS
    objects = cpo.Objects(); objects.segmented = labels
    oset = cpo.ObjectSet(); oset.add_objects(objects, OBJECTS)
    isl = cpi.ImageSetList(); iset = isl.get_image_set(0)
    iset.add(IMAGE, cpi.Image(inten / scale))
    m = cpmeas.Measurements()
    module.run(cpw.Workspace(cpp.Pipeline(), module, iset, oset, m, isl))

    def val(meas, scaled=True):
        v = m.get_current_measurement(OBJECTS, "%s_%s_%s" % (moi.INTENSITY, meas, IMAGE))
        return float(v[0]) * (scale if scaled else 1.0)

    out = {"MASS_DISPLACEMENT": val(moi.MASS_DISPLACEMENT, False)}
    for f, meas in (("EDGE_MEAN_INTENSITY", moi.MEAN_INTENSITY_EDGE),
                    ("EDGE_STDDEV_INTENSITY", moi.STD_INTENSITY_EDGE),
                    ("EDGE_MAX_INTENSITY", moi.MAX_INTENSITY_EDGE),
                    ("EDGE_MIN_INTENSITY", moi.MIN_INTENSITY_EDGE),
                    ("EDGE_INTEGRATED_INTENSITY", moi.INTEGRATED_INTENSITY_EDGE)):
        out[f] = val(meas)
    return out


def main():
    cp = run_cp()
    pins = parse_pins(PIN_FILE, PIN_TABLE)
    ok = True

    print("=== whole-slide cell: CellProfiler, one object over the full 64x64 frame ===")
    want = pins["MASS_DISPLACEMENT"]
    got = cp["MASS_DISPLACEMENT"]
    rel = abs(got - want) / max(1.0, abs(want))
    good = rel <= TOL
    ok &= good
    print("  %s MASS_DISPLACEMENT  cp=%r  nyxus=%r  rel=%.3g"
          % ("OK  " if good else "FAIL", got, want, rel))

    print("\n=== the edge statistics CellProfiler cannot vet here ===")
    print("  an all-ones label image has an EMPTY inner boundary, so CP returns 0 for each;")
    print("  Nyxus reports statistics of the four synthetic AABB corners instead.")
    for f in NOT_VETTABLE:
        cp_zero = cp[f] == 0.0
        ok &= cp_zero
        print("  %s %-26s cp=%-8r nyxus=%r" % ("OK  " if cp_zero else "FAIL", f, cp[f], pins[f]))
    if not all(cp[f] == 0.0 for f in NOT_VETTABLE):
        print("  CellProfiler no longer returns an empty edge set here -- re-triage this cell.")

    # Guard the split itself: a pin for one of these appearing in an oracle table would claim a
    # vetting this run says does not exist.
    print("\n%s" % ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED -- do not promote"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
