"""The command-line reference generator catches a default or a citation that does not match the code.

docs/tools/gen_cli_reference.py --check compares every entry of docs/cli_options.toml with the
source: each single-token default must be the one literal value assigned to the target its
`default_at` key names, and every cited file and line range must exist. These tests change one
entry, or one source file in a copy of the tree, at a time and expect the check to report it, and
move code within a file and expect the check to still pass.
"""
import copy
import importlib.util
import pathlib

import pytest

pytest.importorskip("tomllib")  # the generator reads TOML with the standard library (Python 3.11+)

REPO = pathlib.Path(__file__).resolve().parent.parent.parent
_spec = importlib.util.spec_from_file_location("gen_cli_reference", REPO / "docs" / "tools" / "gen_cli_reference.py")
gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen)


def _with(options, name, **changes):
    """A copy of the option table with the entry `name` changed."""
    out = copy.deepcopy(options)
    entry = next(o for o in out if o["name"] == name)
    entry.update(changes)
    return out


def test_docs_cli_reference_table_matches_code_mechanics():
    options = gen.load_table()
    assert gen.check_names(options) == []
    assert gen.check_defaults(options) == []


@pytest.mark.parametrize("name, wrong", [
    ("--verbose", "0"),
    ("--reduceThreads", "1"),
    ("--pixelDistance", "1"),
    ("--coarseGrayDepth", "1"),
    ("--tinyval", "0.0"),
    ("--gaborkersize", "20"),
    ("--anisoy", "2"),
    ("--glcmAngles", "0,45"),
])
def test_docs_cli_reference_wrong_default_is_caught_mechanics(name, wrong):
    problems = gen.check_defaults(_with(gen.load_table(), name, default=wrong))
    assert any(p.startswith(name + ":") and "not the value assigned" in p for p in problems), problems


def test_docs_cli_reference_missing_cited_file_is_caught_mechanics():
    # --ramLimit has a descriptive default, so its evidence is checked even without a default_at line
    options = gen.load_table()
    entry = next(o for o in options if o["name"] == "--ramLimit")
    problems = gen.check_defaults(_with(options, "--ramLimit", evidence=entry["evidence"] + "; nosuchfile.cpp:12"))
    assert any("nosuchfile.cpp" in p for p in problems), problems


def test_docs_cli_reference_default_without_default_at_is_refused_mechanics():
    options = _with(gen.load_table(), "--dim")
    del next(o for o in options if o["name"] == "--dim")["default_at"]
    problems = gen.check_defaults(options)
    assert any(p.startswith("--dim:") and "no default_at" in p for p in problems), problems


def _source_copy(tmp_path, edit):
    """A copy of src/nyx with environment.cpp changed by `edit`."""
    import shutil
    dst = tmp_path / "nyx"
    shutil.copytree(gen.SRC, dst)
    env = dst / "environment.cpp"
    env.write_text(edit(env.read_text(encoding="utf-8")), encoding="utf-8")
    return dst


def test_docs_cli_reference_default_survives_moved_code_mechanics(tmp_path, monkeypatch):
    # Lines added above every default in environment.cpp move them; the check finds each by its target.
    monkeypatch.setattr(gen, "SRC", _source_copy(tmp_path, lambda t: "#include <cstdlib>\n" * 3 + t))
    assert gen.check_defaults(gen.load_table()) == []


def test_docs_cli_reference_changed_default_in_code_is_caught_mechanics(tmp_path, monkeypatch):
    monkeypatch.setattr(gen, "SRC", _source_copy(
        tmp_path, lambda t: t.replace("n_pixel_distance = 5;", "n_pixel_distance = 3;")))
    problems = gen.check_defaults(gen.load_table())
    assert any(p.startswith("--pixelDistance:") and "not the value assigned" in p for p in problems), problems


def test_docs_cli_reference_ambiguous_default_target_is_refused_mechanics(tmp_path, monkeypatch):
    # A second literal assignment of the same target leaves the default undecided.
    monkeypatch.setattr(gen, "SRC", _source_copy(
        tmp_path, lambda t: t + "\nvoid f() { n_pixel_distance = 7; }\n"))
    problems = gen.check_defaults(gen.load_table())
    assert any(p.startswith("--pixelDistance:") and "exactly one literal" in p for p in problems), problems
