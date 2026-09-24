"""The CLI's exit status.

A featurization that could not be done must tell the shell so. The 3D branches report their error
and return it; a caller that reads only stdout cannot distinguish "no rows because nothing matched"
from "no rows because the run failed", so the exit code is the contract.

This drives the nyxus binary as a subprocess because that is the only place the contract exists:
main() is not reachable from the gtest suite, and the Python binding raises instead of returning a
status. The binary is not built by the PyTest CI job, so these skip unless it is present -- set
NYXUS_CLI to point at it, or build it into one of the usual locations.
"""
import os
import pathlib
import shutil
import subprocess

import numpy as np
import pytest

tifffile = pytest.importorskip("tifffile")

REPO = pathlib.Path(__file__).resolve().parent.parent.parent


def _find_cli():
    env = os.environ.get("NYXUS_CLI")
    if env and pathlib.Path(env).exists():
        return env
    found = shutil.which("nyxus")
    if found:
        return found
    for pat in ("nyxus", "nyxus.exe"):
        for d in (REPO, REPO / "build", REPO / "build-asan", REPO / "src/nyx"):
            for hit in d.glob("**/" + pat):
                if hit.is_file():
                    return str(hit)
    return None


CLI = _find_cli()
needs_cli = pytest.mark.skipif(CLI is None, reason="the nyxus CLI is not built here; set NYXUS_CLI")


def _volume_pair(tmp_path, depth_int, depth_seg):
    """A 3D intensity/mask pair, optionally mismatched in depth so the run cannot proceed."""
    intdir, segdir = tmp_path / "int", tmp_path / "seg"
    intdir.mkdir()
    segdir.mkdir()
    Y, X = 16, 16
    inten = (1 + np.arange(depth_int * Y * X, dtype=np.uint16).reshape(depth_int, Y, X) % 4000)
    mask = np.zeros((depth_seg, Y, X), np.uint16)
    mask[:, 4:12, 4:12] = 1
    # photometric="minisblack" is what makes the leading axis Z rather than samples: tifffile
    # reads a first axis of length 3 or 4 as RGB/RGBA and writes ONE plane of that many samples
    # per pixel, whatever the axes metadata says. At depth 4 that produced a SamplesPerPixel=4
    # file, which nyxus rightly refuses as not grayscale.
    tifffile.imwrite(str(intdir / "v.ome.tif"), inten, photometric="minisblack", metadata={"axes": "ZYX"})
    tifffile.imwrite(str(segdir / "v.ome.tif"), mask, photometric="minisblack", metadata={"axes": "ZYX"})
    return intdir, segdir


def _run(intdir, segdir, outdir, extra=()):
    cmd = [
        CLI,
        "--intDir=" + str(intdir),
        "--segDir=" + str(segdir),
        "--outDir=" + str(outdir),
        "--features=*3D_ALL_INTENSITY*",
        "--filePattern=.*",
        "--outputType=separatecsv",
        "--dim=3",
        *extra,
    ]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=600)


@needs_cli
def test_cli_exits_zero_on_a_good_volume_pair_mechanics(tmp_path):
    """The control: a pair that featurizes cleanly exits 0 and writes a row.

    Without this, a test asserting non-zero on a bad pair would also pass against a binary that
    fails on everything.
    """
    intdir, segdir = _volume_pair(tmp_path, 4, 4)
    outdir = tmp_path / "out"
    outdir.mkdir()
    r = _run(intdir, segdir, outdir)
    assert r.returncode == 0, "a good pair must succeed:\n%s\n%s" % (r.stdout[-2000:], r.stderr[-2000:])
    rows = sum(
        max(0, sum(1 for ln in p.read_text().splitlines() if ln.strip()) - 1)
        for p in outdir.glob("*.csv")
    )
    assert rows > 0, "a successful run writes its feature rows"


@needs_cli
def test_cli_exits_nonzero_on_a_failed_volume_run_mechanics(tmp_path):
    """A 3D segmented run that could not featurize its pair exits non-zero and writes no row.

    What this discriminates: reporting the error on stdout and falling through to `return 0` --
    which is what main() did for the 3D branch -- tells every caller that reads the exit code, from
    a CI step to a `set -e` script to a workflow node, that a run which produced nothing succeeded.
    """
    # a mask deeper than its intensity volume: the pair cannot be read as one volume
    intdir, segdir = _volume_pair(tmp_path, 4, 6)
    outdir = tmp_path / "out"
    outdir.mkdir()
    r = _run(intdir, segdir, outdir)

    assert r.returncode != 0, (
        "a run that could not featurize its pair must fail the shell:\n%s\n%s"
        % (r.stdout[-2000:], r.stderr[-2000:])
    )
    rows = sum(
        max(0, sum(1 for ln in p.read_text().splitlines() if ln.strip()) - 1)
        for p in outdir.glob("*.csv")
    )
    assert rows == 0, "and write no row for what it could not measure"


def _slide_only(tmp_path):
    """A 2D whole-slide dataset: nyxus reads it single-ROI when the two dirs are the same."""
    d = tmp_path / "slide"
    d.mkdir()
    img = (1 + np.arange(64 * 64, dtype=np.uint16).reshape(64, 64) % 4000)
    tifffile.imwrite(str(d / "s.tif"), img)
    return d


def _run_2d(intdir, segdir, outdir, extra=()):
    cmd = [
        CLI,
        "--intDir=" + str(intdir),
        "--segDir=" + str(segdir),
        "--outDir=" + str(outdir),
        "--features=*ALL_INTENSITY*",
        "--filePattern=.*",
        "--outputType=separatecsv",
        *extra,
    ]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=600)


@needs_cli
def test_cli_wholeslide_exits_nonzero_when_the_slide_cannot_be_featurized_mechanics(tmp_path):
    """A 2D whole-slide run that refused its slide exits non-zero and writes no row.

    What this discriminates: the 2D whole-slide path carries its status through four hand-offs --
    the in-RAM pass to featurize_wholeslide, that to the per-thread rv, the per-thread rvs to the
    dataset's worst, and that to main's exit code. Each was dropped somewhere along the way: the
    pass's result was a bare statement, rv was overwritten by a blanket success, and the collected
    rvals were never read. A run that measured nothing reported success to the shell.
    """
    d = _slide_only(tmp_path)
    outdir = tmp_path / "out"
    outdir.mkdir()

    # the control: the same slide featurizes and exits 0
    ok = _run_2d(d, d, outdir)
    assert ok.returncode == 0, "the slide must featurize normally:\n%s\n%s" % (
        ok.stdout[-1500:], ok.stderr[-1500:])
    assert any(outdir.glob("*.csv")), "a successful run writes its rows"

    # and with no RAM to hold it, the slide is refused -- and that has to reach the shell
    outdir2 = tmp_path / "out2"
    outdir2.mkdir()
    bad = _run_2d(d, d, outdir2, extra=["--ramLimit=0"])
    assert bad.returncode != 0, (
        "a slide the run refused must fail the shell:\n%s\n%s" % (bad.stdout[-1500:], bad.stderr[-1500:]))
    rows = sum(
        max(0, sum(1 for ln in p.read_text().splitlines() if ln.strip()) - 1)
        for p in outdir2.glob("*.csv")
    )
    assert rows == 0, "and write no row for what it did not measure"
