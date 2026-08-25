"""OFFLINE analytic oracle for the 2D neighbour second-distance and angle features.

    python tests/vetting/oracles/gen_neighbor_analytic.py        (from the repository root)

Prints the paste-ready goldens AND re-verifies every golden pinned in test_2d_neighbor_analytic.h,
exiting non-zero on any mismatch or on any pin this generator cannot produce.

Recipe `neighbor.scene2d_radius1`: the `neighborhood2d_scene_labels` fixture from tests/test_data.h,
five ROIs, `PIXELDISTANCE=1`. Given the neighbour graph -- which CellProfiler vets independently, see
test_2d_neighbor_cellprofiler.h -- these six features are deterministic closed forms of the ROI
centroids, so an independent numpy recomputation of the documented formulas IS the oracle (the same
analytic-conformance basis as CIRCULARITY and the intensity-histogram percentiles).

Formulas recomputed here, all from the centroids:
  - direction angle = degrees(atan2(dy, dx)) mapped into [0, 360)
  - closest / second-closest by centroid distance, ties keeping ascending-label push order
  - CLOSEST_NEIGHBOR2_* = 0 when fewer than two neighbours lie within the radius
  - ANG_BW_NEIGHBORS_STDDEV is the SAMPLE standard deviation (n-1), 0 for n < 2
  - ANG_BW_NEIGHBORS_MODE is the most frequent round(angle) bucket, lowest bucket winning a tie

Agreement with Nyxus is 1.2e-16 worst case (29 of the 30 values are bit-identical; the exception is
ANG_BW_NEIGHBORS_STDDEV on label 1, one ulp), so the header asserts at the SPEC 7 exact tier.

NOT covered here: PERCENT_TOUCHING, which is not a centroid closed form. It has no promotable oracle
-- CellProfiler uses a different definition -- and is drift-pinned in test_2d_neighbor_regression.h,
with its required bounds asserted in test_2d_neighbor_invariant.h.

Provenance: tool=analytic (numpy, version printed by this script); generator=
tests/vetting/oracles/gen_neighbor_analytic.py. Run offline; CI never invokes it.
"""
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.dirname(os.path.dirname(HERE))
DATA_H = os.path.join(TESTS, "test_data.h")
TEST_H = os.path.join(TESTS, "test_2d_neighbor_analytic.h")
TABLE = "neighbor_2d_analytic_ref_vals_by_label"
SCENE_FIXTURE = "neighborhood2d_scene_labels"

ABSTOL = 1e-9          # SPEC 7 exact tier, an ABSOLUTE band; measured worst residual is 1.4e-14

RADIUS = 1  # PIXELDISTANCE in make_neighbors2d_settings()

FEATURES = ("CLOSEST_NEIGHBOR2_DIST", "CLOSEST_NEIGHBOR1_ANG", "CLOSEST_NEIGHBOR2_ANG",
            "ANG_BW_NEIGHBORS_MEAN", "ANG_BW_NEIGHBORS_STDDEV", "ANG_BW_NEIGHBORS_MODE")


def parse_scene(txt, name):
    """The {x, y, label} pixel array `name` from test_data.h, as a list of (x, y, label) triples.

    Read out of the checked-in fixture rather than transcribed here: a copy in this file would keep
    reproducing the old scene after a test_data.h edit, and the goldens it printed would silently
    stop describing what the C++ side computes.
    """
    body = txt.split(name + "[] = {", 1)[1].split("};", 1)[0]
    body = re.sub(r"//[^\n]*", "", body)
    scene = [(int(x), int(y), int(l)) for x, y, l in
             re.findall(r"\{\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\}", body)]
    if not scene:
        raise RuntimeError(f"fixture {name} not found in {os.path.basename(DATA_H)}")
    return scene


def min_sqdist(a, b):
    d = a[:, None, :] - b[None, :, :]
    return (d * d).sum(-1).min()


def direction_angle_deg(p1, p2):
    a = np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]))
    return a + 360.0 if a < 0.0 else a


def sample_std(v):
    v = np.asarray(v, float)
    if len(v) < 2:
        return 0.0
    return float(np.sqrt(((v - v.mean()) ** 2).sum() / (len(v) - 1)))


def compute(scene):
    """-> {label: {feature: value}}."""
    labels = sorted({l for _, _, l in scene})
    px = {l: np.array([(x, y) for x, y, ll in scene if ll == l], float) for l in labels}
    cen = {l: px[l].mean(axis=0) for l in labels}

    neigh = {l: [] for l in labels}
    for l1 in labels:
        for l2 in labels:
            if l1 >= l2:
                continue
            if min_sqdist(px[l1], px[l2]) <= RADIUS * RADIUS:
                neigh[l1].append(l2)
                neigh[l2].append(l1)
    for l in labels:
        neigh[l].sort()  # ascending-label push order

    out = {}
    for l in labels:
        ns = neigh[l]
        dists = [float(np.hypot(*(cen[l] - cen[n]))) for n in ns]
        order = sorted(range(len(ns)), key=lambda i: dists[i])  # stable -> ties keep push order
        angs = [float(direction_angle_deg(cen[l], cen[n])) for n in ns]

        cn1_ang = angs[order[0]] if ns else 0.0
        cn2_dist = dists[order[1]] if len(ns) > 1 else 0.0
        cn2_ang = angs[order[1]] if len(ns) > 1 else 0.0

        buckets = {}
        for a in angs:
            b = max(0, min(360, int(round(a))))
            buckets[b] = buckets.get(b, 0) + 1
        mode = 0.0
        best = 0
        for b in range(361):
            if buckets.get(b, 0) > best:
                best, mode = buckets[b], float(b)

        out[l] = {
            "CLOSEST_NEIGHBOR2_DIST": cn2_dist,
            "CLOSEST_NEIGHBOR1_ANG": cn1_ang,
            "CLOSEST_NEIGHBOR2_ANG": cn2_ang,
            "ANG_BW_NEIGHBORS_MEAN": float(np.mean(angs)) if angs else 0.0,
            "ANG_BW_NEIGHBORS_STDDEV": sample_std(angs),
            "ANG_BW_NEIGHBORS_MODE": mode,
        }
    return out


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


def parse_pins(txt, table):
    """The header's own table, keyed {label: {feature: value}}.

    Parsed out of the header rather than kept as a copy here: a validation list living in the
    generator only ever compares this script against itself, so editing the header would go
    unnoticed -- which is exactly what this file used to do.
    """
    m = re.search(re.escape(table) + r"\s*\{", txt)
    if not m:
        raise RuntimeError(f"table {table} not found in {os.path.basename(TEST_H)}")
    body = txt[m.end():].split("\n};", 1)[0]
    body = re.sub(r"//[^\n]*", "", body)          # a commented-out golden is not a pin
    pins = {}
    for label, block in _label_blocks(body):
        pins[label] = {n: float(v) for n, v in
                       re.findall(r'\{\s*"(\w+)"\s*,\s*([-0-9.eE+]+)\s*\}', block)}
    return pins


def verify(pins, got):
    """-> (verified, failed, unproducible)."""
    print("")
    n = sum(len(v) for v in pins.values())
    print(f"# verifying {n} pinned goldens against this run")
    nok = nfail = nmiss = 0
    for label in sorted(pins):
        for name in sorted(pins[label]):
            want = pins[label][name]
            have = got.get(label, {}).get(name)
            if have is None:
                print(f"  MISSING L{label} {name}: pinned {want!r} but this oracle produces no "
                      f"such value")
                nmiss += 1
                continue
            # absolute, the same band the header asserts at (SPEC 7 exact tier), so this check and
            # the gtest one cannot disagree about what "agrees" means
            err = abs(have - want)
            if err <= ABSTOL:
                print(f"  OK   L{label} {name}: oracle={have!r} pinned={want!r} abs={err:.3g}")
                nok += 1
            else:
                print(f"  FAIL L{label} {name}: oracle={have!r} pinned={want!r} abs={err:.3g}")
                nfail += 1
    return nok, nfail, nmiss


def main():
    scene = parse_scene(open(DATA_H, encoding="utf-8", errors="replace").read(), SCENE_FIXTURE)
    got = compute(scene)

    print(f"# analytic (numpy {np.__version__}), recipe neighbor.scene2d_radius1")
    print(f"# fixture {SCENE_FIXTURE}: {len(scene)} pixels read from tests/test_data.h")
    print("# paste-ready goldens")
    for label in sorted(got):
        print(f"\t{{{label}, {{")
        for name in FEATURES:
            print(f'\t\t{{"{name}", {got[label][name]!r}}},')
        print("\t}},")

    if not os.path.exists(TEST_H):
        print("")
        print(f"{os.path.basename(TEST_H)} does not exist yet - nothing to verify")
        return 0

    text = open(TEST_H, encoding="utf-8", errors="replace").read()
    pins = parse_pins(text, TABLE)

    # every feature this oracle covers must actually be pinned, on every label: a header that
    # quietly drops a row would otherwise still report all-clear
    missing = [f"L{l} {f}" for l in sorted(got) for f in FEATURES if f not in pins.get(l, {})]
    nok, nfail, nmiss = verify(pins, got)
    for gap in missing:
        print(f"  UNPINNED {gap}: this oracle produces it but the header pins nothing")

    print("")
    print(f"{nok} verified, {nfail} failed, {nmiss} unproducible, {len(missing)} unpinned")
    if nfail or nmiss or missing:
        print("SOME CHECKS FAILED -- do not promote")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
