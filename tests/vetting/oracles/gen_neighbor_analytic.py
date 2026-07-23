"""OFFLINE analytic oracle for the 2D neighbor angle/second-distance features
(SPEC 4, oracle=analytic). Independently reimplements Nyxus' documented
closed-form neighbor geometry in numpy and validates that it reproduces the
goldens pinned in test_neighbor_regression.h.

Which features this vets (6):
    CLOSEST_NEIGHBOR2_DIST
    CLOSEST_NEIGHBOR1_ANG, CLOSEST_NEIGHBOR2_ANG
    ANG_BW_NEIGHBORS_MEAN, ANG_BW_NEIGHBORS_STDDEV, ANG_BW_NEIGHBORS_MODE

Why analytic and not a reference tool: CellProfiler already vets NUM_NEIGHBORS +
CLOSEST_NEIGHBOR1_DIST (the neighbor-graph definition). Given that graph, every
angle/second-distance feature is a deterministic closed form of the ROI
centroids -- so an independent numpy recomputation of the exact documented
formula IS the oracle (the same analytic-conformance basis as CIRCULARITY /
intensity_histogram). CellProfiler's own AngleBetweenNeighbors uses a different
definition (angle subtended at an object by its two neighbors), so it is NOT a
valid oracle for Nyxus' absolute-direction angles.

Nyxus definitions (src/nyx/features/neighbors.cpp):
  - centroid = arithmetic mean of ROI pixel (x, y)
  - neighbor graph: ROIs are neighbors iff min contour-pixel L2 sqdist <= radius^2,
    radius = pixel_distance (=1 here). Per-ROI neighbor list is built lower-triangle
    and pushed to both ROIs, so it is in ASCENDING neighbor-label order.
  - closest / second-closest ranked by CENTROID Euclidean distance; ties keep the
    ascending-label push order (std::min_element returns the first minimum).
  - direction angle = atan2(y2 - y1, x2 - x1) in degrees, +360 if negative
    (image coords, y downwards).
  - ANG_BW_NEIGHBORS_MEAN/STDDEV = arithmetic mean / SAMPLE std (Moments2, n-1
    denominator) of the direction angles to ALL neighbors.
  - ANG_BW_NEIGHBORS_MODE = most frequent round(angle) integer bucket in [0,360],
    lowest bucket index wins ties.

Provenance: tool=analytic (numpy reimplementation of the documented formulas);
env=nyxus_mirp (conda), numpy 1.26/2.x; generator=tests/vetting/oracles/
gen_neighbor_analytic.py. Run offline; CI never invokes it.
"""
import numpy as np

# fixture neighborhood2d_scene_labels (tests/test_data.h): {x, y, label}
SCENE = [
    (4, 2, 3), (5, 2, 3), (4, 3, 3), (5, 3, 3),
    (2, 4, 2), (3, 4, 2), (4, 4, 1), (5, 4, 1), (6, 4, 1), (7, 4, 4), (8, 4, 4),
    (2, 5, 2), (3, 5, 2), (4, 5, 1), (5, 5, 1), (6, 5, 1), (7, 5, 4), (8, 5, 4),
    (4, 6, 1), (5, 6, 1), (6, 6, 1), (7, 6, 4), (8, 6, 4),
    (5, 7, 5), (6, 7, 5), (5, 8, 5), (6, 8, 5),
]
RADIUS = 1  # pixel_distance
TOL = 1e-9  # validation gate vs pinned goldens (gtest itself uses a loose frac bound)

# pinned goldens from test_neighbor_regression.h (the values this oracle must reproduce)
PINNED = {
    1: {"CLOSEST_NEIGHBOR2_DIST": 2.54950975679639, "CLOSEST_NEIGHBOR1_ANG": 0.0,
        "CLOSEST_NEIGHBOR2_ANG": 191.30993247402, "ANG_BW_NEIGHBORS_MEAN": 132.172516881495,
        "ANG_BW_NEIGHBORS_STDDEV": 115.230018010206, "ANG_BW_NEIGHBORS_MODE": 0.0},
    2: {"CLOSEST_NEIGHBOR2_DIST": 0.0, "CLOSEST_NEIGHBOR1_ANG": 11.3099324740202,
        "CLOSEST_NEIGHBOR2_ANG": 0.0, "ANG_BW_NEIGHBORS_MEAN": 11.3099324740202,
        "ANG_BW_NEIGHBORS_STDDEV": 0.0, "ANG_BW_NEIGHBORS_MODE": 11.0},
    3: {"CLOSEST_NEIGHBOR2_DIST": 0.0, "CLOSEST_NEIGHBOR1_ANG": 78.6900675259798,
        "CLOSEST_NEIGHBOR2_ANG": 0.0, "ANG_BW_NEIGHBORS_MEAN": 78.6900675259798,
        "ANG_BW_NEIGHBORS_STDDEV": 0.0, "ANG_BW_NEIGHBORS_MODE": 79.0},
    4: {"CLOSEST_NEIGHBOR2_DIST": 0.0, "CLOSEST_NEIGHBOR1_ANG": 180.0,
        "CLOSEST_NEIGHBOR2_ANG": 0.0, "ANG_BW_NEIGHBORS_MEAN": 180.0,
        "ANG_BW_NEIGHBORS_STDDEV": 0.0, "ANG_BW_NEIGHBORS_MODE": 180.0},
    5: {"CLOSEST_NEIGHBOR2_DIST": 0.0, "CLOSEST_NEIGHBOR1_ANG": 258.69006752598,
        "CLOSEST_NEIGHBOR2_ANG": 0.0, "ANG_BW_NEIGHBORS_MEAN": 258.69006752598,
        "ANG_BW_NEIGHBORS_STDDEV": 0.0, "ANG_BW_NEIGHBORS_MODE": 259.0},
}


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


def compute():
    labels = sorted({l for _, _, l in SCENE})
    px = {l: np.array([(x, y) for x, y, ll in SCENE if ll == l], float) for l in labels}
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
        angs = [direction_angle_deg(cen[l], cen[n]) for n in ns]

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


def main():
    got = compute()
    all_ok = True
    print("=== neighbor analytic oracle vs pinned goldens (test_neighbor_regression.h) ===")
    for l in sorted(PINNED):
        for k in sorted(PINNED[l]):
            o, p = got[l][k], PINNED[l][k]
            scale = max(1.0, abs(p), abs(o))
            ok = abs(o - p) <= TOL * scale
            all_ok &= ok
            print(f"  {'OK ' if ok else 'FAIL'} L{l} {k}: oracle={o!r} pinned={p!r}")
    print(f"\n{'ALL CHECKS PASSED' if all_ok else 'SOME CHECKS FAILED -- do not promote'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
