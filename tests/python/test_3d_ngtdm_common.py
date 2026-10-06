"""Shared fixture for the 3D NGTDM Python tests: one featurisation, and which path it took.

Which path a ROI takes -- in-RAM or oversized -- is decided inside the backend, and the only place it
says so is its verbose log ("processing oversized 3D ROI <label>"). That log goes to C++ std::cout,
which is flushed when the process exits, not when featurize_files() returns, so pytest's capture
fixtures see nothing. The featurisation therefore runs in a child interpreter, and its log is read
after the child has exited.
"""
import json
import subprocess
import sys

NGTDM_3D_FEATURES = ["3NGTDM_BUSYNESS", "3NGTDM_COARSENESS", "3NGTDM_COMPLEXITY",
                     "3NGTDM_CONTRAST", "3NGTDM_STRENGTH"]

_MARK = "NGTDM3D_VALUES="

_CHILD = """
import json, sys
import nyxus
intp, segp, ram_limit, radius = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
nyx = nyxus.Nyxus3D(["*3D_NGTDM*"], ram_limit=ram_limit, verbose=1)
nyx.set_metaparam("3ngtdm/greydepth=0")
nyx.set_metaparam("3ngtdm/radius=%d" % radius)
df = nyx.featurize_files([intp], [segp], False)
assert len(df) == 1, len(df)
print("MARK" + json.dumps({f: float(df.iloc[0][f]) for f in FEATURES}), flush=True)
""".replace("MARK", _MARK).replace("FEATURES", repr(NGTDM_3D_FEATURES))


def featurize_3d_ngtdm(intp, segp, ram_limit, radius):
    """-> ({feature: value}, True if the backend processed the ROI as oversized).

    Raw levels (3ngtdm/greydepth=0) at the given Chebyshev radius, one ROI per call -- on the in-RAM
    path. An oversized ROI does not see either setting: out-of-core 3D runs ignore set_metaparam and
    use the defaults, greydepth 0 and radius 1 (PN-140). Those coincide with what this helper sets only
    at radius 1, so an oversized run at any other radius is refused here rather than returned as if it
    had been computed at that radius.
    """
    proc = subprocess.run([sys.executable, "-c", _CHILD, intp, segp, str(ram_limit), str(radius)],
                          capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, "featurisation failed:\n%s\n%s" % (proc.stdout, proc.stderr)
    values = [ln[len(_MARK):] for ln in proc.stdout.splitlines() if ln.startswith(_MARK)]
    assert len(values) == 1, proc.stdout
    oversized = "processing oversized 3D ROI" in proc.stdout
    assert not (oversized and radius != 1), (
        "the ROI was processed out-of-core, and out-of-core 3D runs ignore set_metaparam and use the "
        "defaults (greydepth 0, radius 1), so these values are not at radius %d (PN-140)" % radius)
    return json.loads(values[0]), oversized
