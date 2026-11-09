"""A release's source distribution reaches PyPI only beside its full wheel grid.

pip prefers a new version's source distribution over an older wheel, so a platform whose wheel
is missing would attempt a source build. ci-utils/check_wheel_grid.py compares PyPI's file list
with the grid publish_pypi.yml builds, and the release wheels are configured so that a dropped
ALLEXTRAS part fails the build.
"""
import importlib.util
import pathlib
import re

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent.parent
VERSION = "1.2.3"
PLATFORM_TAGS = (
    "manylinux_2_28_x86_64",
    "manylinux_2_28_aarch64",
    "win_amd64",
    "macosx_10_15_x86_64",
    "macosx_11_0_arm64",
)
PYTHONS = ("cp310", "cp311", "cp312", "cp313")


def _grid():
    spec = importlib.util.spec_from_file_location("check_wheel_grid", REPO / "ci-utils" / "check_wheel_grid.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _wheel(python, tag, version=VERSION):
    return "nyxus-" + version + "-" + python + "-" + python + "-" + tag + ".whl"


def _complete(version=VERSION):
    return [_wheel(python, tag, version) for python in PYTHONS for tag in PLATFORM_TAGS]


def _jobs(workflow):
    """Split a workflow into its jobs, keyed by job id."""
    text = (REPO / ".github" / "workflows" / workflow).read_text(encoding="utf-8")
    body = text[text.index("\njobs:\n"):]
    parts = re.split(r"^  ([A-Za-z0-9_-]+):\n", body, flags=re.M)
    return dict(zip(parts[1::2], parts[2::2]))


def _wheel_jobs(workflow):
    return {name: job for name, job in _jobs(workflow).items() if "cibuildwheel --output-dir" in job}


def test_complete_grid_has_nothing_missing_mechanics():
    assert len(_complete()) == 20
    assert _grid().missing_wheels(_complete(), VERSION) == []


def test_release_without_macos_arm64_is_reported_mechanics():
    # the file set of nyxus 0.12.0: every cell but macOS arm64
    files = [name for name in _complete("0.12.0") if "arm64" not in name]
    assert len(files) == 16
    assert _grid().missing_wheels(files, "0.12.0") == [python + " macOS arm64" for python in PYTHONS]


def test_single_missing_cell_is_named_mechanics():
    # the file set of nyxus 0.11.0: 19 wheels, one Linux x86_64 cell absent
    files = [name for name in _complete() if name != _wheel("cp313", "manylinux_2_28_x86_64")]
    assert _grid().missing_wheels(files, VERSION) == ["cp313 Linux x86_64"]


def test_compound_platform_tag_covers_its_cell_mechanics():
    files = [name for name in _complete() if name != _wheel("cp312", "manylinux_2_28_x86_64")]
    files.append(_wheel("cp312", "manylinux2014_x86_64.manylinux_2_28_x86_64"))
    assert _grid().missing_wheels(files, VERSION) == []


def test_other_versions_and_sdists_cover_no_cell_mechanics():
    files = _complete("1.2.2") + ["nyxus-" + VERSION + ".tar.gz"]
    assert len(_grid().missing_wheels(files, VERSION)) == 20


def test_sdist_name_gives_the_version_mechanics():
    assert _grid().version_of_sdist("dist/nyxus-0.13.0.post2.dev5.tar.gz") == "0.13.0.post2.dev5"


def test_foreign_sdist_name_refused_mechanics():
    with pytest.raises(SystemExit):
        _grid().version_of_sdist("dist/other-1.2.3.tar.gz")


def test_incomplete_grid_exits_nonzero_and_names_the_cells_mechanics(monkeypatch, capsys):
    grid = _grid()
    files = [name for name in _complete() if "arm64" not in name]
    monkeypatch.setattr(grid, "fetch_filenames", lambda version: files)
    assert grid.main(["dist/nyxus-" + VERSION + ".tar.gz", "--attempts", "2", "--delay", "0"]) == 1
    assert "cp310 macOS arm64" in capsys.readouterr().err


def test_grid_appearing_between_attempts_passes_mechanics(monkeypatch):
    grid = _grid()
    answers = iter([[], _complete()])
    monkeypatch.setattr(grid, "fetch_filenames", lambda version: next(answers))
    assert grid.main(["dist/nyxus-" + VERSION + ".tar.gz", "--attempts", "2", "--delay", "0"]) == 0


def test_grid_check_precedes_the_sdist_upload_mechanics():
    job = _jobs("publish_pypi.yml")["sdist"]
    check = job.index("ci-utils/check_wheel_grid.py dist/*.tar.gz")
    assert check < job.index("twine upload dist/*.tar.gz")


def test_sdist_job_waits_for_every_wheel_job_mechanics():
    needs = re.search(r"^    needs: \[(.*)\]$", _jobs("publish_pypi.yml")["sdist"], re.M).group(1)
    needed = {name.strip() for name in needs.split(",")}
    assert set(_wheel_jobs("publish_pypi.yml")) <= needed


def test_grid_matches_the_wheel_jobs_pythons_mechanics():
    for name, job in _wheel_jobs("publish_pypi.yml").items():
        built = tuple(re.findall(r'"(cp3\d+)-\*"', job))
        assert built == _grid().PYTHONS, name


@pytest.mark.parametrize("workflow", ["publish_pypi.yml", "build_wheels.yml"])
def test_wheel_builds_refuse_a_dropped_extra_mechanics(workflow):
    jobs = _wheel_jobs(workflow)
    assert len(jobs) == 5
    for name, job in jobs.items():
        assert "-DREQUIRE_ALLEXTRAS=ON" in job, workflow + ": " + name
