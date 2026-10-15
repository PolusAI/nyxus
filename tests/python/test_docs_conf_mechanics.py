"""The Sphinx configuration gives the pages' |release| substitution a value.

docs/source/index.rst quotes the release in its citation through |release|. Sphinx renders an
unset release as an empty string without a warning, so the strict docs build does not catch it;
conf.py must set it from CITATION.cff, and the page title must stay without a version number.
"""
import pathlib
import re
import runpy

REPO = pathlib.Path(__file__).resolve().parent.parent.parent
CONF = REPO / "docs" / "source" / "conf.py"


def _citation_version():
    text = (REPO / "CITATION.cff").read_text(encoding="utf-8")
    return re.search(r"^version:\s*([^\s#]+)", text, re.M).group(1)


def test_docs_conf_release_set_from_citation_mechanics():
    conf = runpy.run_path(str(CONF))
    assert conf.get("release") == _citation_version()
    assert conf.get("version") == ".".join(_citation_version().split(".")[:2])


def test_docs_conf_title_has_no_version_mechanics():
    conf = runpy.run_path(str(CONF))
    assert conf["html_title"] == "Nyxus"


def test_docs_conf_release_used_by_pages_mechanics():
    # the substitution this configuration serves is still in use; drop the release if it is not
    uses = [p for p in (REPO / "docs" / "source").rglob("*.rst") if "|release|" in p.read_text(encoding="utf-8")]
    assert uses
