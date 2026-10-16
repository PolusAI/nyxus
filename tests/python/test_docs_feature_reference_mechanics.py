"""The feature reference generator reports an empty description and an unclaimed feature class.

docs/tools/gen_feature_reference.py --check must fail with a message when a feature code has an
empty description in docs/feature_descriptions.toml, and when a feature class belongs to no family
in FAMILIES; the second case used to end in a KeyError traceback instead of the message.
"""
import importlib.util
import pathlib
import sys

import pytest

pytest.importorskip("tomllib")  # the generator reads TOML with the standard library (Python 3.11+)

REPO = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "docs" / "tools"))
_spec = importlib.util.spec_from_file_location("gen_feature_reference", REPO / "docs" / "tools" / "gen_feature_reference.py")
gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen)


def _descriptions():
    import tomllib
    with open(gen.DESCRIPTIONS, "rb") as f:
        data = tomllib.load(f)
    data.pop("family_notes", None)
    return data


def test_docs_feature_reference_descriptions_complete_mechanics():
    features, _ = gen.reg.load()
    assert gen.check_descriptions(features, _descriptions()) == []


def test_docs_feature_reference_empty_description_is_caught_mechanics():
    features, _ = gen.reg.load()
    desc = _descriptions()
    fam = next(iter(desc))
    code = next(iter(desc[fam]))
    desc[fam][code] = "  "
    problems = gen.check_descriptions(features, desc)
    assert any(p.startswith(f"{code}: no entry") for p in problems), problems


def test_docs_feature_reference_unclaimed_class_is_reported_mechanics(monkeypatch, capsys):
    real = gen.family_of
    features, _ = gen.reg.load()
    victim = features[0].feature_class
    monkeypatch.setattr(gen, "family_of", lambda f: None if f.feature_class == victim else real(f))
    assert gen.main(["--check"]) == 1
    assert "no family in FAMILIES claims class" in capsys.readouterr().err
