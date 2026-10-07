"""Derives, verifies and emits the marching-cubes case table shipped in
src/nyx/features/3d_mesh.cpp, the surface 3MESH_VOLUME is the enclosed volume of.

    python derive_marching_cubes_table.py            # derive and verify
    python derive_marching_cubes_table.py --emit     # also print the C++ table

The table is derived here rather than copied. For each of the 256 corner masks the
six cube faces are contoured with 2D marching squares and the resulting directed
segments are chained across shared cube edges into the closed loops a cube's share
of the surface has to span. An ambiguous face -- the two diagonal corners inside --
is always resolved by SEPARATING the inside corners; that rule reads only the
face's own four corners, so two cubes sharing a face cut it the same way and the
assembled surface is watertight.

A loop still admits several triangulations, and they differ in area by a few parts
in a thousand. For the 134 masks with no ambiguous face this script takes the
classic triangulation from scikit-image's marching_cubes(method='lorensen'), but
only after checking that it triangulates exactly the loops derived above -- so
the reference settles an arbitrary choice, it does not define the surface.
(MIRP calls marching_cubes with its default method, 'lewiner', and pyradiomics
carries its own 128-entry table; neither is this triangulation.) The remaining 120 masks keep the derived
triangulation, because the classic table leaves those cubes open.

scikit-image is a generation-time reference only, never a Nyxus build or CI
dependency (BSD-3-Clause; SPEC §4). Without it the derivation and every check
below still run; only the classic triangulation cannot be adopted, and --emit
refuses.

Verified here: the surface closes on random volumes exercising all 256 masks; a
lone voxel gives exactly the octahedron's 1/6 volume and sqrt(3) area; and each
solid box matches the closed form that 3d_surface.cpp's whole-volume branch uses.
"""

import argparse
import itertools
import math
import random

# corner k sits at CORNER[k], a vertex of the unit cube
CORNER = [
    (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
    (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1),
]

# edge e joins corners EDGE[e]
EDGE = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]

# the six faces, each as its four corners counter-clockwise SEEN FROM OUTSIDE
FACES = [
    (0, 3, 2, 1),   # z = 0
    (4, 5, 6, 7),   # z = 1
    (0, 1, 5, 4),   # y = 0
    (3, 7, 6, 2),   # y = 1
    (0, 4, 7, 3),   # x = 0
    (1, 2, 6, 5),   # x = 1
]

EDGE_OF_CORNERS = {frozenset(c): e for e, c in enumerate(EDGE)}


def edge_midpoint(e):
    a, b = EDGE[e]
    return tuple((CORNER[a][i] + CORNER[b][i]) / 2.0 for i in range(3))


def face_segments(face, inside):
    """Directed segments on one face, as (from_edge, to_edge) cube-edge pairs.

    A segment runs so that, with the face seen from outside the cube, the inside
    region lies on its left.
    """
    n = 4
    bits = [inside[c] for c in face]
    crossing = {}
    for i in range(n):
        a, b = face[i], face[(i + 1) % n]
        if inside[a] != inside[b]:
            crossing[i] = EDGE_OF_CORNERS[frozenset((a, b))]

    entering = [i for i in crossing if not bits[i] and bits[(i + 1) % n]]
    exiting = [i for i in crossing if bits[i] and not bits[(i + 1) % n]]
    assert len(entering) == len(exiting)

    if not entering:
        return []
    if len(entering) == 1:
        return [(crossing[exiting[0]], crossing[entering[0]])]

    # the ambiguous face: pair each exit with the entry immediately before it, which
    # wraps the contour tightly around each inside corner instead of joining them
    return [(crossing[x], crossing[min(entering, key=lambda e: (x - e) % n)])
            for x in exiting]


def loops_for_case(mask):
    """The closed loops of cube-edge indices for one corner mask."""
    inside = [(mask >> k) & 1 == 1 for k in range(8)]

    successor = {}
    for face in FACES:
        for a, b in face_segments(face, inside):
            assert a not in successor, "a cube edge is left twice"
            successor[a] = b

    loops = []
    unused = set(successor)
    while unused:
        start = min(unused)
        loop = [start]
        unused.discard(start)
        nxt = successor[start]
        while nxt != start:
            loop.append(nxt)
            assert nxt in unused, "the chain re-entered a used edge"
            unused.discard(nxt)
            nxt = successor[nxt]
        loops.append(loop)
    return loops


def fan(loops):
    """Fan-triangulate each loop. Wound opposite to the loop, so normals face out."""
    tris = []
    for loop in loops:
        for i in range(1, len(loop) - 1):
            tris.append((loop[0], loop[i + 1], loop[i]))
    return tris


def has_ambiguous_face(mask):
    inside = [(mask >> k) & 1 == 1 for k in range(8)]
    for face in FACES:
        b = [inside[c] for c in face]
        if b[0] == b[2] and b[1] == b[3] and b[0] != b[1]:
            return True
    return False


def triangulates(tris, loops):
    """True if `tris` spans exactly `loops`, orientation included."""
    want = {}
    for lp in loops:
        for i in range(len(lp)):
            he = (lp[i], lp[(i + 1) % len(lp)])
            want[he] = want.get(he, 0) + 1
    got = {}
    for t in tris:
        for i in range(3):
            he = (t[i], t[(i + 1) % 3])
            got[he] = got.get(he, 0) + 1
    # interior half-edges cancel against their opposites; the boundary must survive
    left = {}
    for he, n in got.items():
        net = n - got.get((he[1], he[0]), 0)
        if net > 0:
            left[he] = net
    return left == want


def classic_case(mask):
    """The classic triangulation of one mask, read off the reference implementation."""
    import numpy as np
    from skimage import measure

    pt2e = {tuple(round(v + 1.0, 6) for v in edge_midpoint(e)): e for e in range(12)}
    occ = np.zeros((4, 4, 4), float)
    for k, c in enumerate(CORNER):
        if (mask >> k) & 1:
            occ[1 + c[0], 1 + c[1], 1 + c[2]] = 1.0
    verts, faces, _, _ = measure.marching_cubes(occ, level=0.5, method="lorensen")
    out = []
    for f in faces:
        pts = [tuple(round(float(x), 6) for x in verts[i]) for i in f]
        if all(p in pt2e for p in pts):
            out.append(tuple(pt2e[p] for p in pts))
    return out


def build_table(use_reference=True):
    table, stats = [], {"reference": 0, "derived (ambiguous)": 0,
                        "derived (no reference)": 0, "rejected": 0, "empty": 0}
    for m in range(256):
        loops = loops_for_case(m)
        derived = fan(loops)
        if not loops:
            table.append([])
            stats["empty"] += 1
            continue
        if has_ambiguous_face(m):
            table.append(derived)
            stats["derived (ambiguous)"] += 1
            continue
        if not use_reference:
            table.append(derived)
            stats["derived (no reference)"] += 1
            continue
        ref = classic_case(m)
        # triangulates() judges against the loop direction while the table stores the
        # opposite winding, so validate the reference as given, then flip it
        for cand in (ref, [(t[0], t[2], t[1]) for t in ref]):
            if len(cand) == len(derived) and triangulates(cand, loops):
                table.append([(t[0], t[2], t[1]) for t in cand])
                stats["reference"] += 1
                break
        else:
            table.append(derived)
            stats["rejected"] += 1
    return table, stats


# ---------------------------------------------------------------------------
# verification
# ---------------------------------------------------------------------------

def mesh_of(table, occ):
    """(volume, area, closed) of the table's surface over a 3D boolean array."""
    nx, ny, nz = len(occ), len(occ[0]), len(occ[0][0])
    vol = area = 0.0
    halfedges = {}
    for x in range(nx - 1):
        for y in range(ny - 1):
            for z in range(nz - 1):
                mask = 0
                for k, c in enumerate(CORNER):
                    if occ[x + c[0]][y + c[1]][z + c[2]]:
                        mask |= 1 << k
                for tri in table[mask]:
                    p = [tuple(o + m for o, m in zip((x, y, z), edge_midpoint(e)))
                         for e in tri]
                    a, b, c3 = p
                    vol += (a[0] * (b[1] * c3[2] - b[2] * c3[1])
                            - a[1] * (b[0] * c3[2] - b[2] * c3[0])
                            + a[2] * (b[0] * c3[1] - b[1] * c3[0])) / 6.0
                    u = tuple(b[i] - a[i] for i in range(3))
                    v = tuple(c3[i] - a[i] for i in range(3))
                    cr = (u[1] * v[2] - u[2] * v[1],
                          u[2] * v[0] - u[0] * v[2],
                          u[0] * v[1] - u[1] * v[0])
                    area += 0.5 * math.sqrt(sum(q * q for q in cr))
                    for i in range(3):
                        he = (p[i], p[(i + 1) % 3])
                        halfedges[he] = halfedges.get(he, 0) + 1
    closed = all(halfedges.get((v, u), 0) == n for (u, v), n in halfedges.items())
    return vol, area, closed


def verify(table, stats):
    ok = stats["rejected"] == 0
    print("table composition:", stats)
    if stats["rejected"]:
        print("  FAIL: a classic triangulation did not span the derived loops")

    rng = random.Random(20260925)
    seen, nclosed, trials = set(), 0, 50
    for _ in range(trials):
        occ = [[[False] * 10 for _ in range(10)] for _ in range(10)]
        for p in itertools.product(range(1, 9), repeat=3):
            occ[p[0]][p[1]][p[2]] = rng.random() < 0.5
        nclosed += mesh_of(table, occ)[2]
        for cell in itertools.product(range(9), repeat=3):
            m = 0
            for k, c in enumerate(CORNER):
                if occ[cell[0] + c[0]][cell[1] + c[1]][cell[2] + c[2]]:
                    m |= 1 << k
            seen.add(m)
    print("watertight on %d/%d random volumes, exercising %d/256 masks"
          % (nclosed, trials, len(seen)))
    ok = ok and nclosed == trials and len(seen) == 256

    # a lone voxel is the octahedron whose vertices sit half a lattice step out
    occ = [[[False] * 3 for _ in range(3)] for _ in range(3)]
    occ[1][1][1] = True
    v, a, _ = mesh_of(table, occ)
    print("single voxel: volume %.17g (exact %.17g), area %.17g (exact %.17g)"
          % (v, 1 / 6, a, math.sqrt(3)))
    ok = ok and abs(v - 1 / 6) < 1e-15 and abs(a - math.sqrt(3)) < 1e-14

    # a solid box comes out bevelled, by the closed form 3d_surface.cpp's
    # whole-volume branch uses in place of meshing every voxel
    for (w, h, d) in [(1, 1, 1), (1, 4, 9), (2, 2, 2), (5, 7, 11), (8, 8, 8)]:
        occ = [[[1 <= i <= w and 1 <= j <= h and 1 <= k <= d
                 for k in range(d + 2)] for j in range(h + 2)] for i in range(w + 2)]
        v, a, _ = mesh_of(table, occ)
        e = w + h + d - 3
        fv = w * h * d - e / 2.0 - 5.0 / 6.0
        fa = 2 * (w * h + h * d + w * d) - 2 * e * (2 - math.sqrt(2)) - (6 - math.sqrt(3))
        agrees = abs(v - fv) < 1e-9 and abs(a - fa) < 1e-9
        ok = ok and agrees
        print("box %2dx%2dx%2d: volume %12.6f area %12.6f  closed form %s"
              % (w, h, d, v, a, "agrees" if agrees else "DISAGREES"))

    return ok


def emit_cpp(table):
    rows = []
    for m in range(256):
        flat = []
        for tri in table[m]:
            flat.extend(tri)
        flat.append(-1)
        flat += [-1] * (16 - len(flat))
        rows.append("\t\t{ " + ", ".join("%2d" % v for v in flat) + " },\t// %3d" % m)
    return "\n".join(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit", action="store_true",
                    help="print the C++ table body for src/nyx/features/3d_mesh.cpp")
    args = ap.parse_args()

    try:
        import skimage  # noqa: F401
        have_ref = True
    except ImportError:
        have_ref = False
        print("scikit-image not installed: deriving without the classic triangulation")
        if args.emit:
            raise SystemExit("--emit needs scikit-image to settle the 134 unambiguous masks")

    tbl, st = build_table(use_reference=have_ref)
    good = verify(tbl, st)
    if args.emit:
        print()
        print(emit_cpp(tbl))
    raise SystemExit(0 if good else 1)
