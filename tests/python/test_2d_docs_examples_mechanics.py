"""The Python examples shown in the documentation run as written.

Every script in docs/examples/ is run here, unmodified, in a fresh directory. The pages include
those scripts with literalinclude, so what a reader copies is what this test executed. Each
script creates its own input (or copies it from tests/python/data) and ends with assertions
on its result, kept outside the part the page shows.

A new script in docs/examples/ is picked up automatically.

The scripts run on every Python the package supports, so they use only modules that Python 3.10
has. The values each page quotes live in docs/feature_extras.toml; this module reads that file
(tomllib, or tomli, which pytest installs on Python 3.10) and hands the scripts a JSON copy
through NYXUS_DOCS_EXTRAS.
"""
import json
import os
import pathlib
import re
import subprocess
import sys

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

pytest.importorskip("tifffile")

REPO = pathlib.Path(__file__).resolve().parent.parent.parent
EXAMPLES = sorted((REPO / "docs" / "examples").glob("*.py"))
NEEDS_ARROW = {"arrow_output.py"}
# Modules that Python 3.10, the oldest supported version, does not have
NOT_IN_PY310 = {"tomllib"}


@pytest.fixture(scope="module")
def extras_json(tmp_path_factory):
    extras = tomllib.loads((REPO / "docs" / "feature_extras.toml").read_text(encoding="utf-8"))
    path = tmp_path_factory.mktemp("extras") / "feature_extras.json"
    path.write_text(json.dumps(extras), encoding="utf-8")
    return path


def _params():
    return [pytest.param(s, id=s.stem, marks=[pytest.mark.arrow] if s.name in NEEDS_ARROW else [])
            for s in EXAMPLES]


def test_2d_docs_examples_present_mechanics():
    assert EXAMPLES, "docs/examples/ holds no example scripts"
    for script in EXAMPLES:
        text = script.read_text(encoding="utf-8")
        assert "# example-start" in text and "# example-end" in text, \
            f"{script.name}: the page includes the lines between '# example-start' and '# example-end'"


def test_2d_docs_examples_python310_mechanics():
    for script in EXAMPLES:
        imported = set(re.findall(r"^\s*(?:import|from)\s+(\w+)", script.read_text(encoding="utf-8"), re.M))
        assert not imported & NOT_IN_PY310, \
            f"{script.name} imports {sorted(imported & NOT_IN_PY310)}, which Python 3.10 does not have"


@pytest.mark.parametrize("script", _params())
def test_2d_docs_examples_run_mechanics(script, tmp_path, extras_json):
    env = dict(os.environ)
    env["NYXUS_DOCS_DATA"] = str(REPO / "tests" / "python" / "data")
    env["NYXUS_DOCS_EXTRAS"] = str(extras_json)
    done = subprocess.run([sys.executable, str(script)], cwd=tmp_path, env=env,
                          capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, f"{script.name} failed:\n{done.stdout}\n{done.stderr}"

README_BLOCKS = re.findall(r"```python\n(.*?)```", (REPO / "README.md").read_text(encoding="utf-8"), re.S)


def test_2d_readme_has_quick_starts_mechanics():
    assert README_BLOCKS, "README.md shows no ```python block"


@pytest.mark.parametrize("code", README_BLOCKS, ids=[f"readme_block_{i}" for i in range(len(README_BLOCKS))])
def test_2d_readme_quick_starts_run_mechanics(code, tmp_path):
    """README.md is also the PyPI page, so its Python quick starts must run as written."""
    script = tmp_path / "quick_start.py"
    script.write_text(code, encoding="utf-8")
    done = subprocess.run([sys.executable, str(script)], cwd=tmp_path,
                          capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, f"README quick start failed:\n{code}\n{done.stdout}\n{done.stderr}"
