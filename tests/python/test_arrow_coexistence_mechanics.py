"""nyxus and pyarrow in one process.

A nyxus wheel carries its own Arrow and pyarrow carries another, and both end up loaded in the
same interpreter whenever a caller reads nyxus' Parquet or Arrow output. Each copy registers its
filesystem factories ("file", "local", ...) in a process-wide registry when it is loaded. If the
two copies share that registry, the second registration of "file" is recorded as a conflict and
every later pyarrow.fs.LocalFileSystem() raises ArrowKeyError -- which is what any path-based read
does (pq.read_table(path), and pandas.read_parquet(path) on pandas 3.1+).

The load order is part of the contract, so each order runs in a fresh interpreter: inside this
pytest process both libraries are already loaded and the order is whatever collection chose.
"""
import subprocess
import sys
import textwrap

import numpy as np
import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

import nyxus

ARROW_ENABLED = nyxus.Nyxus(["MEAN"]).arrow_is_enabled()
needs_arrow = pytest.mark.skipif(not ARROW_ENABLED, reason="this nyxus build has no Arrow support")

# One pyarrow round trip through a local path: LocalFileSystem() resolves the "file" factory from
# the registry, and read_table(path) does the same on its own.
_PYARROW_LOCAL_IO = textwrap.dedent("""
    import os, sys, tempfile
    import pyarrow as pa, pyarrow.fs, pyarrow.parquet as pq
    pyarrow.fs.LocalFileSystem()
    path = os.path.join(tempfile.mkdtemp(), "t.parquet")
    pq.write_table(pa.table({"x": [1, 2, 3]}), path)
    assert pq.read_table(path).column("x").to_pylist() == [1, 2, 3]
""")


def _run(script):
    return subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=300)


@needs_arrow
def test_arrow_coexistence_nyxus_first_mechanics():
    """nyxus imported first, as in any script that featurizes and then reads the output."""
    r = _run("import nyxus\n" + _PYARROW_LOCAL_IO)
    assert r.returncode == 0, r.stderr


@needs_arrow
def test_arrow_coexistence_pyarrow_first_mechanics():
    """pyarrow imported first (pandas 3 imports it on `import pandas`), nyxus second."""
    r = _run("import pyarrow, pyarrow.fs\nimport nyxus\n" + _PYARROW_LOCAL_IO)
    assert r.returncode == 0, r.stderr


@needs_arrow
def test_arrow_coexistence_parquet_output_read_by_path_mechanics(tmp_path):
    """nyxus' own Parquet output, read back by pyarrow from its path in this process."""
    intens = np.zeros((32, 32), dtype=np.uint16)
    seg = np.zeros((32, 32), dtype=np.uint16)
    intens[8:24, 8:24] = np.arange(256, dtype=np.uint16).reshape(16, 16) + 1
    seg[8:24, 8:24] = 1

    nyx = nyxus.Nyxus(["MEAN"])
    path = nyx.featurize(intens, seg, output_type="parquet", output_path=str(tmp_path / "out"))

    table = pq.read_table(path)
    assert table.num_rows == 1
    assert "MEAN" in table.column_names
    assert table.column("MEAN").to_pylist()[0] == pytest.approx(128.5)
