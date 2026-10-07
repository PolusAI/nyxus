"""Compares 3MESH_VOLUME's surface mesh with PyRadiomics' shape mesh (MeshVolume,
SurfaceArea) on the same masks.

    python compare_mesh_volume_pyradiomics.py

Both sides contour the binary mask with marching cubes at the 0.5 isolevel, one
voxel of padding around it, every vertex at an edge midpoint, and integrate the
enclosed volume as a sum of signed tetrahedra. They differ in the case table:
PyRadiomics (radiomics/src/cshape.c) carries the classic 128-entry table and
serves the other 128 masks by complementing the mask and flipping the volume's
sign, while Nyxus carries all 256 masks (see derive_marching_cubes_table.py).

The Nyxus side here is a replay of the MC_TRIANGLES table read from
src/nyx/features/3d_mesh.cpp, walked the way march_roi_surface() walks it and
integrated from the first vertex the way roi_mesh_volume() does. It reads the
shipped table, so it follows any change to it; it does not run the Nyxus binary.
The PyRadiomics side calls the compiled radiomics.cShape.calculate_coefficients
that shape.py calls, on a [z, y, x] mask padded by one voxel, at unit spacing.

Each mask is also run a second time on both sides, placed further from the image
origin. Over a closed surface the volume integral does not depend on that
placement, so a change there means the mesh is open. The script prints a
markdown table of both volumes, both moved volumes, both areas and whether the
Nyxus mesh is closed (every directed edge matched by its reverse).

PyRadiomics (BSD-3-Clause) is a reference only, never a Nyxus build or CI
dependency (SPEC §4).
"""

import os
import re
from collections import Counter

import numpy as np
from radiomics import cShape

HERE = os.path.dirname(os.path.abspath(__file__))
MESH_CPP = os.path.join(HERE, "..", "..", "..", "src", "nyx", "features", "3d_mesh.cpp")

# corner and edge layout of src/nyx/features/3d_mesh.cpp
CORNER = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
          (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
EDGE = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6),
        (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
EMID = np.array([[0.5 * (CORNER[a][t] + CORNER[b][t]) for t in range(3)] for a, b in EDGE])

MOVE = 7    # voxels of extra padding before the mask in the moved PyRadiomics run


def load_table():
    with open(MESH_CPP, encoding="utf-8") as f:
        src = f.read()
    body = src[src.index("MC_TRIANGLES[256][16]"):]
    rows = re.findall(r"\{([-\d,\s]+)\},\s*//\s*(\d+)", body)[:256]
    table = [[int(v) for v in r[0].split(",")] for r in rows]
    assert len(table) == 256 and all(len(t) == 16 for t in table)
    assert [int(r[1]) for r in rows] == list(range(256))
    return table


def nyxus_mesh(mask, table):
    """Triangles of the mask's surface, mask indexed [x, y, z]."""
    p = np.pad(mask.astype(bool), 1)
    tris = []
    nx, ny, nz = p.shape
    for k in range(nz - 1):
        for j in range(ny - 1):
            for i in range(nx - 1):
                m = 0
                for c, (dx, dy, dz) in enumerate(CORNER):
                    if p[i + dx, j + dy, k + dz]:
                        m |= 1 << c
                if m in (0, 255):
                    continue
                t = table[m]
                o = np.array([i, j, k], float)
                for q in range(0, 15, 3):
                    if t[q] < 0:
                        break
                    tris.append([o + EMID[t[q + u]] for u in range(3)])
    return np.array(tris)


def nyxus_volume_area(tris):
    o = tris[0, 0]
    a, b, c = tris[:, 0] - o, tris[:, 1] - o, tris[:, 2] - o
    volume = abs(np.einsum("ij,ij->i", a, np.cross(b, c)).sum()) / 6
    area = 0.5 * np.linalg.norm(np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]), axis=1).sum()
    return volume, area


def closed(tris):
    """True when the mesh is closed and consistently oriented: every directed edge
    is traversed as often in reverse. Where two sheets of the surface touch along
    an edge, four triangles share it, two in each direction."""
    uses = Counter()
    for t in tris:
        key = [tuple(np.round(2 * v).astype(int)) for v in t]
        for a, b in ((0, 1), (1, 2), (2, 0)):
            uses[(key[a], key[b])] += 1
    return all(uses.get((b, a), 0) == n for (a, b), n in uses.items())


def pyradiomics_volume_area(mask, before=1):
    zyx = np.transpose(mask, (2, 1, 0)).astype(np.int8)
    zyx = np.ascontiguousarray(np.pad(zyx, ((before, 1),) * 3))
    area, volume, _ = cShape.calculate_coefficients(zyx, np.array([1.0, 1.0, 1.0]))
    return volume, area


def ball(r):
    g = np.arange(-r, r + 1)
    x, y, z = np.meshgrid(g, g, g, indexing="ij")
    return x * x + y * y + z * z <= r * r


def cases():
    out = {"single voxel": np.ones((1, 1, 1), bool),
           "box 3x4x5": np.ones((3, 4, 5), bool),
           "ball r=6": ball(6)}
    g = np.arange(-8, 9)
    x, y, z = np.meshgrid(g, g, g, indexing="ij")
    out["ellipsoid 8/5/3"] = (x / 8.) ** 2 + (y / 5.) ** 2 + (z / 3.) ** 2 <= 1

    edge = np.zeros((2, 2, 1), bool)
    edge[0, 0, 0] = edge[1, 1, 0] = True
    out["two voxels, edge contact"] = edge
    corner = np.zeros((2, 2, 2), bool)
    corner[0, 0, 0] = corner[1, 1, 1] = True
    out["two voxels, corner contact"] = corner

    rng = np.random.default_rng(7)
    pitted = ball(10)
    shell = pitted & ~np.pad(ball(9), 1)
    pitted[shell & (rng.random(pitted.shape) < 0.15)] = False
    out["ball r=10, 15% surface pits"] = pitted

    rng = np.random.default_rng(1)
    for fill in (0.3, 0.5, 0.7):
        out[f"random {int(fill * 100)}% fill, 8^3"] = rng.random((8, 8, 8)) < fill
    return out


def main():
    table = load_table()
    head = ("mask", "voxels", "nyxus V", "nyxus V moved", "PyRadiomics V", "PyRadiomics V moved",
            "nyxus A", "PyRadiomics A", "nyxus closed")
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    for name, mask in cases().items():
        tris = nyxus_mesh(mask, table)
        nv, na = nyxus_volume_area(tris)
        nv_moved, _ = nyxus_volume_area(nyxus_mesh(np.pad(mask, ((MOVE, 0),) * 3), table))
        pv, pa = pyradiomics_volume_area(mask)
        pv_moved, _ = pyradiomics_volume_area(mask, before=1 + MOVE)
        print(f"| {name} | {int(mask.sum())} | {nv:.6f} | {nv_moved:.6f} | {pv:.6f} | {pv_moved:.6f} | "
              f"{na:.6f} | {pa:.6f} | {'yes' if closed(tris) else 'NO'} |")


if __name__ == "__main__":
    main()
