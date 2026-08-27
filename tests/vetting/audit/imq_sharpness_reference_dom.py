"""Measures SHARPNESS against the published reference DOM implementation, and shows the six
structural reasons they differ. Backs tests/vetting/audit/imq_pydom_sharpness_vetting_report.md.

This is NOT an oracle generator: it pins nothing and `dom` is not a SPEC 4 oracle token. Its job is
to keep the refutation honest, and it has exactly two checks that can fail:

  1. The port below reproduces the SHARPNESS golden pinned in test_imq_regression.h. That is what
     makes the enumerated divergences a statement about the shipped sharpness.cpp rather than about
     a Python script. If sharpness.cpp changes and the pin moves, this fails and the report has to
     be re-derived.
  2. The reference implementation still disagrees. If a future change makes the two agree, this
     fails too -- and SHARPNESS becomes promotable, which is the outcome the report is waiting for.

Reference: Kumar, Chen, Doermann, "Sharpness estimation for document and scene images" (ICPR 2012),
as implemented in https://github.com/umang-singhal/pydom (dom/dom.py, DOM.get_sharpness, defaults
width=2, sharpness_threshold=2, edge_threshold=0.0001).

THE REFERENCE IS INVOKED, NOT VENDORED. pydom is GPL-3.0 and Nyxus is MIT, so no line of it is
copied into this repository: this script imports the installed package and calls its public API,
and every reference number below is upstream's own output. Install it into the offline audit env
only -- it is not a Nyxus dependency and CI never invokes this script:

    pip install git+https://github.com/umang-singhal/pydom.git

The `nyx_*` functions below are a port of Nyxus' own MIT-licensed sharpness.cpp and carry no
upstream code.

Provenance of the run behind this report: pydom 0.1 at upstream commit 2554af8, numpy 2.4.6,
cv2 4.13.0, python 3.11.15, conda env nyxus_mirp. Every run prints its own installed versions.
Run offline; CI never invokes it.

Usage:  python tests/vetting/audit/imq_sharpness_reference_dom.py
"""
import os
import re

import numpy as np
import cv2      # not called here: the reference uses it for its median filter, so its version is
                # part of this run's provenance and is printed rather than assumed

try:
    from dom import DOM
except ImportError as exc:      # an absent reference must name its own remedy, not traceback
    raise SystemExit(
        "the reference DOM implementation is not installed: %s\n"
        "  pip install git+https://github.com/umang-singhal/pydom.git\n"
        "It is deliberately not vendored -- pydom is GPL-3.0 and this repository is MIT." % exc)


HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.dirname(os.path.dirname(HERE))
DATA_H = os.path.join(TESTS, "test_data.h")
TEST_H = os.path.join(TESTS, "test_imq_regression.h")
TABLE = "imq_regression_ref_vals"

EPS = 1e-8                 # SharpnessFeature's EPSILON
PORT_RELTOL = 1e-14        # the port must reproduce the C++ pin, not merely resemble it


# ----------------------------------------------------------------------------- fixture and pins

def parse_pixels(txt, name):
    body = txt.split("const static NyxusPixel " + name + "[] = {", 1)[1].split("};", 1)[0]
    body = re.sub(r"//[^\n]*", "", body)
    trips = [(int(a), int(b), int(c)) for a, b, c in
             re.findall(r"\{\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\}", body)]
    if not trips:
        raise RuntimeError("fixture %s not found in %s" % (name, os.path.basename(DATA_H)))
    return trips


def build_roi():
    txt = open(DATA_H, encoding="utf-8", errors="replace").read()
    inten = parse_pixels(txt, "im_quality_intensity")
    mask = parse_pixels(txt, "im_quality_mask")
    kept = [p for p, m in zip(inten, mask) if m[2] != 0]
    x0 = min(x for x, _, _ in kept)
    y0 = min(y for _, y, _ in kept)
    w = max(x for x, _, _ in kept) - x0 + 1
    h = max(y for _, y, _ in kept) - y0 + 1
    img = np.zeros((h, w), dtype=np.int64)
    for x, y, v in kept:
        img[y - y0, x - x0] = v
    return img


def parse_pins(path, table):
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
    body = re.sub(r"//[^\n]*", "", txt[m.end():j - 1])
    return {n: float(v) for n, v in
            re.findall(r'\{\s*"(\w+)"\s*,\s*([-0-9.eE+]+)\s*\}', body)}


# --------------------------------------------------------- the shipped algorithm (sharpness.cpp)

def nyx_pad_array(a, padr, padc):
    rows, cols = a.shape
    pr, pc = rows + 2 * padr, cols + 2 * padc
    out = np.zeros((pr, pc), dtype=np.int64)
    out[padr:padr + rows, padc:padc + cols] = a
    for i in range(rows):
        for k in range(1, padc + 1):
            out[i + padr, k - 1] = a[i, 0]
            out[i + padr, pc - k] = a[i, cols - 1]
    for i in range(padr):
        for j in range(pc):
            out[i, j] = out[padr, j]
            out[padr + rows + i, j] = out[pr - padr - 1, j]
    return out


def nyx_median_blur(a, ksize=3):
    """median_blur(): pads by (rows, cols) -- not by (ksize-1)/2 -- then compacts back."""
    rows, cols = a.shape
    pad = (ksize - 1) // 2
    padded = nyx_pad_array(a, rows, cols)
    pr, pc = padded.shape
    out = np.zeros((pr, pc), float)
    for i in range(pr):
        for j in range(pc):
            win = np.sort(padded[max(0, i - pad):min(pr, i + pad + 1),
                                 max(0, j - pad):min(pc, j + pad + 1)].ravel())
            out[i, j] = win[int(np.floor(len(win) / 2))]
    flat = out.ravel().copy()
    new_cols = pc - 2 * cols
    for i in range(pr - 2 * rows):
        for j in range(new_cols):
            flat[i * new_cols + j] = flat[(i + rows) * pc + (j + cols)]
    return flat            # the erase() in remove_padding() is a no-op; the tail is left behind


def nyx_convolve_1d(v, kernel):
    n, k = len(v), len(kernel)
    pad = k // 2
    out = np.zeros(n)
    for i in range(n):
        s = 0.0
        for j in range(k):
            idx = i - pad + j
            if 0 <= idx < n:
                s += v[idx] * kernel[j]
        out[i] = s
    return out


def nyx_smooth_image(img):
    rows, cols = img.shape
    kernel = [-0.5, 0.0, 0.5]
    smoothed = img.astype(float).ravel().copy()
    smoothed_t = np.zeros(rows * cols)
    for i in range(rows):
        for j in range(cols):
            smoothed_t[j * rows + i] = img[i, j]
    for i in range(rows):
        conv = nyx_convolve_1d(smoothed[i * cols:(i + 1) * cols], kernel)
        smoothed[i * cols:i * cols + len(conv)] = conv
    for i in range(cols):
        conv = nyx_convolve_1d(smoothed_t[i * rows:(i + 1) * rows], kernel)
        smoothed_t[i * rows:i * rows + len(conv)] = conv
    for i in range(cols):                       # the transpose-back loop, verbatim
        for j in range(rows):
            p, q = j * cols + i, i * rows + j
            smoothed_t[p], smoothed_t[q] = smoothed_t[q], smoothed_t[p]
    mx = smoothed.max()                         # one max for both, taken from the row-convolved one
    return np.abs(smoothed) / (mx + EPS), np.abs(smoothed_t) / (mx + EPS)


def nyx_edges(img, thr=0.0001):
    rows, cols = img.shape
    sx, sy = nyx_smooth_image(img)
    return (sx[:rows * cols] > thr).astype(float), (sy[:rows * cols] > thr).astype(float)


def nyx_dom(Im, rows, cols):
    dx = np.zeros(rows * cols)
    dy = np.zeros(rows * cols)
    for i in range(rows):
        for j in range(cols):
            up = Im[(i - 2) * cols + j] if i >= 2 else 0.0
            dn = Im[(i + 2) * cols + j] if i < rows - 2 else 0.0
            dx[i * cols + j] = abs(up - 2 * Im[i * cols + j] + dn)
            lf = Im[i * cols + (j - 2)] if j >= 2 else 0.0
            rt = Im[i * cols + (j + 2)] if j < cols - 2 else 0.0
            dy[i * cols + j] = abs(lf - 2 * Im[i * cols + j] + rt)
    return dx, dy


def nyx_contrast(Im, rows, cols):
    cx = np.zeros(rows * cols)
    cy = np.zeros(rows * cols)
    for i in range(rows):
        for j in range(cols):
            v = Im[(i + 1) * cols + j] if i + 1 < rows else 0.0
            cx[i * cols + j] = abs(v - Im[i * cols + j])
            v = Im[i * cols + (j + 1)] if j + 1 < cols else 0.0
            cy[i * cols + j] = abs(v - Im[i * cols + j])
    return cx, cy


def nyx_sharpness(img, width=2):
    rows, cols = img.shape
    blurred = nyx_median_blur(img) / 255.0
    ex, ey = nyx_edges(img)
    dx, dy = nyx_dom(blurred, rows, cols)
    cx, cy = nyx_contrast(blurred, rows, cols)
    cx, cy = cx * ex, cy * ey
    sx = np.zeros(rows * cols)
    sy = np.zeros(rows * cols)
    for i in range(width, rows - width):
        for dom_v, con_v, out in ((dx, cx, sx), (dy, cy, sy)):
            num = np.zeros(cols)
            dn = np.zeros(cols)
            for j in range(-width, width):      # asymmetric: 2*width rows, not 2*width+1
                for k in range(cols):
                    num[k] += abs(dom_v[(i + j) * cols + k])
                    dn[k] += con_v[(i + j) * cols + k]
            for k in range(cols - width):       # the last `width` columns are never written
                out[i * cols + k] = num[k] / dn[k] if dn[k] > 1e-3 else 0.0
    rx = sx.sum() / (ex.sum() + EPS)            # SUM of sharpness, not a count above a threshold
    ry = sy.sum() / (ey.sum() + EPS)
    return float(np.sqrt(rx * rx + ry * ry)), dict(
        sum_sx=float(sx.sum()), sum_sy=float(sy.sum()),
        n_edgex=int(ex.sum()), n_edgey=int(ey.sum()), rx=float(rx), ry=float(ry))


# ------------------------------------------------- the reference (pydom, INVOKED not vendored)

# The upstream package is GPL-3.0 and this repository is MIT, so nothing below reimplements or
# copies it: every reference number is produced by calling DOM's own public methods. get_sharpness()
# is the entry point the report quotes; the intermediates come from the same object so the
# diagnostics describe that same run, and reference_sharpness() asserts the two agree before
# returning either.

WIDTH = 2                  # DOM.get_sharpness defaults, restated so the call site is readable
SHARPNESS_THRESHOLD = 2
EDGE_THRESHOLD = 0.0001


def reference_version():
    """What the installed reference calls itself -- provenance the run produced, not a literal."""
    try:
        from importlib.metadata import version
        return version("pydom")
    except Exception:
        return "unknown"


def reference_sharpness(image_u8):
    """DOM.get_sharpness on the fixture, plus the intermediates the report tabulates.

    Composed from the public API rather than reimplemented: load() -> edges() -> sharpness_matrix()
    -> sharpness_measure() is the pipeline get_sharpness() runs, so recomposing the score from the
    intermediates and comparing it to get_sharpness() checks that this call sequence IS that entry
    point. Sx/Sy are masked by the edge maps before counting because the reference does that
    immediately before aggregating -- divergence 5 in the report, and the reason the raw matrices
    count more pixels than the score reflects.
    """
    iqa = DOM()
    score = float(iqa.get_sharpness(image_u8, width=WIDTH,
                                    sharpness_threshold=SHARPNESS_THRESHOLD,
                                    edge_threshold=EDGE_THRESHOLD))

    gray, Im = iqa.load(image_u8)
    iqa.edges(gray, edge_threshold=EDGE_THRESHOLD)
    edgex, edgey = iqa.edgex, iqa.edgey
    Sx, Sy = iqa.sharpness_matrix(Im, width=WIDTH)

    n_sharpx = int(np.sum(np.multiply(Sx, edgex) >= SHARPNESS_THRESHOLD))
    n_sharpy = int(np.sum(np.multiply(Sy, edgey) >= SHARPNESS_THRESHOLD))
    n_edgex, n_edgey = int(np.sum(edgex)), int(np.sum(edgey))
    rx = n_sharpx / (n_edgex + EPS)
    ry = n_sharpy / (n_edgey + EPS)
    recomposed = float(np.sqrt(rx ** 2 + ry ** 2))
    if abs(recomposed - score) > 1e-12:
        raise RuntimeError(
            "the intermediates do not recompose get_sharpness() (%r vs %r) -- upstream's pipeline "
            "moved and the diagnostics below no longer describe the score" % (recomposed, score))

    return score, dict(n_sharpx=n_sharpx, n_sharpy=n_sharpy, n_edgex=n_edgex, n_edgey=n_edgey,
                       rx=float(rx), ry=float(ry),
                       n_sharpx_unmasked=int(np.sum(Sx >= SHARPNESS_THRESHOLD)),
                       n_sharpy_unmasked=int(np.sum(Sy >= SHARPNESS_THRESHOLD)))


def main():
    img = build_roi()
    pins = parse_pins(TEST_H, TABLE)
    if "SHARPNESS" not in pins:
        raise RuntimeError("%s pins no SHARPNESS" % os.path.basename(TEST_H))
    pinned = pins["SHARPNESS"]

    port, pd = nyx_sharpness(img)
    ref, rd = reference_sharpness(img.astype(np.uint8))

    all_ok = True
    print("=== SHARPNESS vs the reference DOM implementation ===")
    # The INSTALLED versions, not the docstring's. pydom's is here because the reference is now
    # invoked rather than vendored, so which build produced the number is a property of this env.
    print("    pydom %s, numpy %s, cv2 %s, fixture %dx%d (w x h)"
          % (reference_version(), np.__version__, cv2.__version__, img.shape[1], img.shape[0]))

    rel = abs(port - pinned) / abs(pinned)
    ok = rel <= PORT_RELTOL
    all_ok &= ok
    print("  %s the port reproduces the shipped algorithm: port=%.17g pinned=%.17g rel=%.3g"
          % ("OK " if ok else "FAIL", port, pinned, rel))

    div = abs(port - ref) / abs(ref)
    ok = div > 1e-2
    all_ok &= ok
    print("  %s reference still disagrees: nyxus=%.17g reference=%.17g  rel=%.3g (%.0f%%)"
          % ("OK " if ok else "FAIL", port, ref, div, 100 * div))

    print("\n  where the two part company")
    print("    aggregation   nyxus sums Sx/Sy (%.4f / %.4f); reference counts pixels >= 2 (%d / %d)"
          % (pd["sum_sx"], pd["sum_sy"], rd["n_sharpx"], rd["n_sharpy"]))
    print("    edge axes     nyxus edge_x=%d edge_y=%d; reference edgex=%d edgey=%d -- swapped"
          % (pd["n_edgex"], pd["n_edgey"], rd["n_edgex"], rd["n_edgey"]))
    print("    masking       nyxus never masks Sx/Sy by the edge maps before aggregating; the "
          "reference's counts drop %d->%d and %d->%d when it does"
          % (rd["n_sharpx_unmasked"], rd["n_sharpx"], rd["n_sharpy_unmasked"], rd["n_sharpy"]))
    # Read off the reference's source rather than measured: its public API exposes no hook that
    # isolates either, and neither is worth vendoring a copy to demonstrate.
    print("    Sy pass       nyxus sums Sy over ROWS like Sx; the reference sums it down COLUMNS")
    print("    normalization nyxus divides both smoothed images by the row-convolved one's max")
    print("    coverage      nyxus leaves the last width=2 columns of Sx/Sy at 0")

    print("\n%s" % ("SHARPNESS IS NOT THE REFERENCE DOM MEASURE -- keep it regression" if all_ok
                    else "SOME CHECKS FAILED -- re-derive the report"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
