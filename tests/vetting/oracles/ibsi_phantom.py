"""The IBSI digital phantom, read out of tests/test_data.h.

Every oracle generator that runs on the phantom reads it from here rather than embedding a copy, so
the C++ tests and the reference tools are fed the same pixels and cannot drift apart. numpy is the
only dependency, so this imports cleanly in each tool's own environment (the tools themselves do not
share one).
"""
import re
from pathlib import Path

import numpy as np

TEST_DATA = Path(__file__).resolve().parents[2] / "test_data.h"


def _parse_array(text, name):
    body = re.search(rf"{name}\s*\[\]\s*=\s*\{{(.*?)\}};", text, re.S)
    if body is None:
        raise LookupError("%s not found in %s" % (name, TEST_DATA))
    return [(int(x), int(y), int(v)) for x, y, v in
            re.findall(r"\{\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\}", body.group(1))]


def phantom_slices(path=TEST_DATA):
    """-> [(intensity, mask), ...] for z1..z4, as 2D int arrays. Mask is 1 inside the ROI."""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    out = []
    for z in range(1, 5):
        intensity_px = _parse_array(text, f"ibsi_phantom_z{z}_intensity")
        mask_px = _parse_array(text, f"ibsi_phantom_z{z}_mask")
        w = max(p[0] for p in intensity_px) + 1
        h = max(p[1] for p in intensity_px) + 1
        intensity, mask = np.zeros((h, w), np.int32), np.zeros((h, w), np.int32)
        for x, y, v in intensity_px:
            intensity[y, x] = v
        for x, y, v in mask_px:
            mask[y, x] = 1 if v else 0
        out.append((intensity, mask))
    return out


if __name__ == "__main__":
    for i, (intensity, mask) in enumerate(phantom_slices(), 1):
        levels = sorted(set(intensity[mask > 0].tolist()))
        print(f"z{i}: {intensity.shape[0]}x{intensity.shape[1]}, "
              f"{int(mask.sum())} masked voxels, in-mask levels {levels}")
