"""bench_square12_graded, the 2D fixture of the negative coarse grey depth, and MIRP's run on it.

A 12x12 ROI in a 14x14 frame at levels 100 + 3 ((7x + 11y + xy) mod 37), 36 of them over 100-208.
tests/test_main_nyxus.h builds it for the C++ tests (`make_2d_square12_graded_phantom`); this module
builds the same array for MIRP, and `check_square_in_header()` fails if the header's formula is no
longer the one below. gen_ngldm_mirp.py and gen_gldzm_mirp.py both pin MIRP on it and both load it
from here, so there is one Python copy of the fixture.

At 20 PyRadiomics-style bins (range 108, bin width 5.4) the interior bin edges fall at 127, 154 and
181, and 14 of its 144 pixels sit exactly on one: 5 at 127, 4 at 154, 5 at 181. IBSI's fixed bin
number, floor(n (x - min) / (max - min)) + 1, puts each in the bin above its edge, as MIRP does. A
division by the bin width, floor((x - min) / ((max - min) / n)) + 1, agrees at 127 and 154, and the
5 at 181 are the ones it sends one bin low: (181 - 100) / 5.4 lands just under 15 in floating point.

numpy is the only dependency at import; MIRP is imported by `mirp_fbn()` alone.
"""
import logging
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.dirname(os.path.dirname(HERE))
MAIN_H = os.path.join(TESTS, "test_main_nyxus.h")

SIDE, LO, HI = 14, 1, 12
NBINS = 20


def square12_graded():
    """-> (intensity, mask) of bench_square12_graded, as tests/test_main_nyxus.h builds it."""
    img = np.zeros((SIDE, SIDE), np.float64)
    msk = np.zeros((SIDE, SIDE), np.int32)
    for y in range(SIDE):
        for x in range(SIDE):
            if LO <= x <= HI and LO <= y <= HI:
                xx, yy = x - LO, y - LO
                img[y, x] = 100 + 3 * ((7 * xx + 11 * yy + xx * yy) % 37)
                msk[y, x] = 1
    return img, msk


def check_square_in_header():
    """The C++ fixture must be the array MIRP is fed: compare the formula's text."""
    src = open(MAIN_H, encoding="utf-8").read()
    m = re.search(r"make_2d_square12_graded_phantom.*?\{(.*?)\n    \}", src, re.S)
    if not m:
        raise RuntimeError("no make_2d_square12_graded_phantom in test_main_nyxus.h")
    body = re.sub(r"\s+", "", m.group(1))
    for frag in ("side=14", "lo=1", "hi=12", "100+3*((7*xx+11*yy+xx*yy)%37)"):
        if frag not in body:
            raise RuntimeError(f"test_main_nyxus.h's square no longer carries {frag!r}")


def mirp_fbn(family, mapping):
    """-> {Nyxus feature: (value, MIRP column)} for MIRP's `family` on the square.

    `base_discretisation_method="fixed_bin_number"`, n_bins=20, `by_slice=True`: MIRP bins the ROI
    itself, so nothing of Nyxus' binning is reproduced here. `mapping` is Nyxus feature -> MIRP column
    stem; each stem has to resolve to exactly one column at this configuration.
    """
    import mirp
    logging.disable(logging.INFO)

    img, msk = square12_graded()
    res = mirp.extract_features(
        image=img[np.newaxis], mask=msk[np.newaxis], image_spacing=(1.0, 1.0, 1.0), by_slice=True,
        base_feature_families=family, base_discretisation_method="fixed_bin_number",
        base_discretisation_n_bins=NBINS)
    row = (res[0] if isinstance(res, list) else res).iloc[0]
    out = {}
    for nyx, stem in mapping.items():
        pat = re.compile(re.escape(stem) + r"_(d1_a0\.0_)?2d" + f"_fbn_n{NBINS}$")
        cols = [c for c in row.index if pat.match(c)]
        if len(cols) != 1:
            raise RuntimeError(f"{nyx}: expected one MIRP column for {stem}, found {cols}")
        out[nyx] = (float(row[cols[0]]), cols[0])
    return out


if __name__ == "__main__":
    img, msk = square12_graded()
    roi = img[msk > 0]
    lo, hi = roi.min(), roi.max()
    on_edge = roi[(NBINS * (roi - lo)) % (hi - lo) == 0]
    on_edge = on_edge[(on_edge > lo) & (on_edge < hi)]
    print(f"{int(msk.sum())} ROI pixels, {np.unique(roi).size} levels over {lo:g}-{hi:g}; "
          f"on an interior edge at {NBINS} bins: "
          + ", ".join(f"{int(c)} at {v:g}" for v, c in zip(*np.unique(on_edge, return_counts=True))))
