"""The ``>>>`` examples in the docstrings of the public nyxus objects run as written.

Every docstring of the nyxus.nyxus and nyxus.functions modules that holds examples is one test
case. Its examples run with doctest, under NORMALIZE_WHITESPACE and ELLIPSIS, in a fresh working
directory that holds copies of tests/python/data/int and tests/python/data/seg - the "int" and
"seg" directories the examples name. A docstring that writes Arrow IPC or Parquet output is
skipped when the build has no Arrow support. Examples that touch the GPU guard the call with
gpu_is_available(), so they run on a CPU-only build too.

A new docstring with examples is picked up automatically.

Kind: *mechanics* per tests/vetting/SPEC.md 2 -- these assert that the documented calls run and
print what the docstrings show, and claim nothing about feature values.
"""
import doctest
import pathlib
import re
import shutil

import pytest

nyxus = pytest.importorskip("nyxus")
import nyxus.functions  # noqa: E402
import nyxus.nyxus  # noqa: E402

DATA = pathlib.Path(__file__).resolve().parent / "data"
FLAGS = doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS
ARROW_OUTPUT = re.compile(r"""output_type\s*=\s*["'](arrowipc|parquet)["']""")
ARROW_ENABLED = nyxus.Nyxus(["MEAN"]).arrow_is_enabled()


def _docstrings():
    finder = doctest.DocTestFinder()
    found = []
    for mod in (nyxus.nyxus, nyxus.functions):
        found += [t for t in finder.find(mod, mod.__name__, globs={}) if t.examples]
    return sorted(found, key=lambda t: t.name)


def _params():
    params = []
    for t in _docstrings():
        marks = []
        if any(ARROW_OUTPUT.search(ex.source) for ex in t.examples):
            marks = [pytest.mark.arrow,
                     pytest.mark.skipif(not ARROW_ENABLED, reason="this build has no Arrow support")]
        params.append(pytest.param(t, id=t.name, marks=marks))
    return params


DOCSTRINGS = _params()


def test_docstring_examples_present_mechanics():
    assert DOCSTRINGS, "no docstring of nyxus.nyxus or nyxus.functions holds a >>> example"


@pytest.mark.parametrize("test", DOCSTRINGS)
def test_docstring_examples_run_mechanics(test, tmp_path, monkeypatch):
    for d in ("int", "seg"):
        shutil.copytree(DATA / d, tmp_path / d)
    monkeypatch.chdir(tmp_path)
    test.globs = {}
    out = []
    runner = doctest.DocTestRunner(optionflags=FLAGS)
    result = runner.run(test, out=out.append, clear_globs=True)
    assert result.failed == 0, \
        f"{test.name}: {result.failed} of {result.attempted} examples failed\n" + "".join(out)
