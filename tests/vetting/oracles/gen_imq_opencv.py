"""OFFLINE OpenCV oracle for the image-quality focus-score features (SPEC 4, oracle=opencv).

Runs the real cv2.Laplacian on the gtest fixture `im_quality_intensity` / `im_quality_mask` and
re-verifies EVERY golden pinned in test_imq_opencv.h against that run, exiting non-zero on any
mismatch, on any pin this generator cannot produce, and on any value it produces that the header
pins nothing for. Both the fixture and the pins are parsed out of the checked-in files: a copy of
either kept here would only ever compare this script against itself.

Vets (2):
  FOCUS_SCORE        -- cv2.Laplacian(roi, CV_64F, ksize=1, BORDER_CONSTANT).var()
  LOCAL_FOCUS_SCORE  -- the same call on the tile get_local_focus_score() visits, / scale^2

FOCUS_SCORE is the Pech-Pacheco et al. (2000) "variance of the Laplacian" focus measure, which is
what `cv2.Laplacian(img, cv2.CV_64F).var()` computes. Two independent parts:

  1. The convolution. Nyxus' hand-rolled laplacian() uses the ksize=1 kernel
     [[0,1,0],[1,-4,1],[0,1,0]] and drops out-of-range taps, i.e. zero padding. That is exactly
     cv2.Laplacian(..., ksize=1, borderType=cv2.BORDER_CONSTANT); this generator asserts the two
     filtered images are equal cell for cell (max abs diff 0.0), so the convolution is proved rather
     than inferred from a matching variance.
  2. The variance. Plain population variance of the signed filtered image,
     mean((x - mean(x))^2) == numpy/cv2 .var() (ddof=0).

SCOPE OF THE CLAIM -- what these two assertions do and do not cover:
  * ksize=1 only. The ksize>1 kernel [[2,0,2],[0,-8,0],[2,0,2]] is a Nyxus/CellProfiler convention
    with no cv2.Laplacian counterpart (cv2's ksize=3 Sobel-derived Laplacian is the same stencil
    scaled differently), and Nyxus never calls it from calculate() -- not covered here.
  * LOCAL_FOCUS_SCORE covers only the tile get_local_focus_score() actually visits. It loops
    `for (y = 0; y < height - M; y += M)` with M = height/scale, so for scale=2 the condition is
    0 < h/2 -> true, 6 < 6 -> false: exactly ONE tile is visited, not scale^2 = 4, while the divisor
    stays scale^2. The generator asserts that tile count, so a change to the loop bound fails here
    instead of quietly redefining the golden.
  * The out-of-core path (get_focus_score_NT) is not covered: it convolves through a fixed 30x30
    window buffer, passes (width, height) to laplacian() in the (m_image, n_image) order, and takes
    the variance over the whole conv_buffer including the part no pixel wrote. Out of scope here.

NOT vetted here: MIN_SATURATION / MAX_SATURATION (-> gen_imq_cellprofiler.py),
POWER_SPECTRUM_SLOPE and SHARPNESS (-> test_imq_regression.h; see tests/vetting/matrix/imq.md).

Provenance of the run behind the pins: tool=opencv, version=cv2 4.13.0 (opencv-python), numpy
2.4.6, python 3.11.15, conda env nyxus_mirp. Every run prints its own installed versions, so
this line describes the run that produced the goldens rather than claiming anything about a
later one; cv2.Laplacian(src=float64 ROI, ddepth=cv2.CV_64F, ksize=1,
borderType=cv2.BORDER_CONSTANT) then ndarray.var() (ddof=0).
generator=tests/vetting/oracles/gen_imq_opencv.py. Run offline; CI never invokes it -- OpenCV is not
a Nyxus runtime dependency.
"""
import os
import re

import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.dirname(os.path.dirname(HERE))
DATA_H = os.path.join(TESTS, "test_data.h")
TEST_H = os.path.join(TESTS, "test_imq_opencv.h")
TABLE = "imq_opencv_ref_vals"

KERNEL = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], float)  # focus_score.cpp, ksize=1
SCALE = 2       # FocusScoreFeature::get_local_focus_score default
EXPECTED_TILES = 1
# The pins in the header are the TOOL's own digits printed at %.17g, so re-verifying them against a
# fresh run must ROUND-TRIP EXACTLY: a decimal literal with 17 significant digits parses back to the
# same double. This is deliberately NOT the band the C++ test asserts at (SPEC 7's exact tier, an
# absolute 1e-9). Conflating the two -- which is what this constant used to do -- would let a
# hand-edited golden drift by up to 1e-9 and still be reported as verified. A non-zero residual here
# means the tool's own output moved, and the pin has to be regenerated rather than tolerated.
ROUND_TRIP_ABSTOL = 0.0


def parse_pixels(txt, name):
    """The {x, y, value} array `name` from test_data.h, as (x, y, value) triples.

    Read out of the checked-in fixture rather than transcribed here: a copy in this file would keep
    driving cv2 with the old ROI after a test_data.h edit, and the goldens it printed would silently
    stop describing what the C++ side computes.
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


def cv_laplacian(img):
    return cv2.Laplacian(img, cv2.CV_64F, ksize=1, borderType=cv2.BORDER_CONSTANT)


def nyxus_laplacian(img):
    """Reimplementation of FocusScoreFeature::laplacian() (zero padding, ksize=1)."""
    h, w = img.shape
    out = np.zeros((h, w), float)
    for i in range(h):
        for j in range(w):
            for ik in range(3):
                for jk in range(3):
                    ii, jj = i + 1 - ik, j + 1 - jk
                    if 0 <= ii < h and 0 <= jj < w:
                        out[i, j] += img[ii, jj] * KERNEL[ik, jk]
    return out


def nyxus_tiles(img, scale):
    """The tiles get_local_focus_score() actually visits -- note the `y < height - M` bound."""
    h, w = img.shape
    m, n = h // scale, w // scale
    return [img[y:y + m, x:x + n]
            for y in range(0, h - m, m) for x in range(0, w - n, n)]


def main():
    img = build_roi()
    pins = parse_pins(TEST_H, TABLE)

    all_ok = True
    print("=== OpenCV cv2.Laplacian focus scores vs the pins in test_imq_opencv.h ===")
    # The INSTALLED versions, not the docstring's. A generator that labels every run with a
    # literal reports a provenance it did not produce, which is what #448 found in the mirp one.
    print("    cv2 %s, numpy %s, fixture %dx%d (w x h), values %s"
          % (cv2.__version__, np.__version__, img.shape[1], img.shape[0],
             sorted(set(img.ravel().astype(int).tolist()))))

    # (1) prove the convolution, not just the scalar it feeds
    conv_diff = float(np.abs(cv_laplacian(img) - nyxus_laplacian(img)).max())
    ok = conv_diff == 0.0
    all_ok &= ok
    print("  %s convolution: max|cv2 - nyxus laplacian| = %r"
          % ("OK " if ok else "FAIL", conv_diff))
    print("       raw Laplacian mean = %r (non-zero -> abs()-before-variance would understate it)"
          % float(cv_laplacian(img).mean()))

    # (2) the tiling LOCAL_FOCUS_SCORE is defined over, asserted rather than assumed
    tiles = nyxus_tiles(img, SCALE)
    ok = len(tiles) == EXPECTED_TILES
    all_ok &= ok
    print("  %s tiling: get_local_focus_score() visits %d tile(s) of %dx%d, divisor scale^2 = %d"
          % ("OK " if ok else "FAIL", len(tiles), tiles[0].shape[1], tiles[0].shape[0],
             SCALE * SCALE))

    produced = {
        "FOCUS_SCORE": float(cv_laplacian(img).var()),
        "LOCAL_FOCUS_SCORE": sum(float(cv_laplacian(t).var()) for t in tiles) / (SCALE * SCALE),
    }

    # (3) range checks: both quantities are population variances
    for name, value in sorted(produced.items()):
        ok = value >= 0.0
        all_ok &= ok
        print("  %s range: %s = %.17g >= 0 (a population variance)"
              % ("OK " if ok else "FAIL", name, value))

    # (4) every pin re-verified against this run
    for name in sorted(pins):
        if name not in produced:
            all_ok = False
            print("  FAIL %s is pinned in %s but this generator cannot produce it"
                  % (name, os.path.basename(TEST_H)))
            continue
        resid = abs(produced[name] - pins[name])
        ok = resid <= ROUND_TRIP_ABSTOL
        all_ok &= ok
        rel = resid / abs(pins[name]) if pins[name] else 0.0
        print("  %s %s: opencv=%.17g pinned=%.17g  abs=%.3g rel=%.3g"
              % ("OK " if ok else "FAIL", name, produced[name], pins[name], resid, rel))

    # (5) the reverse direction: a value this oracle produces that the header pins nothing for
    for name in sorted(set(produced) - set(pins)):
        all_ok = False
        print("  FAIL %s: opencv produces %.17g and %s pins nothing for it"
              % (name, produced[name], os.path.basename(TEST_H)))

    print("\n%s" % ("ALL OPENCV-VET CHECKS PASSED" if all_ok
                    else "SOME CHECKS FAILED -- do not promote"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
