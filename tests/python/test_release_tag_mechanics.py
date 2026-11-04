"""The tag release-please creates is one every release consumer reads as the release version.

release-please names the release tag from release-please-config.json: the component (the
package name) and a dash unless include-component-in-tag is false, then "v", then the version.
Three consumers read that tag: setuptools_scm through tag_regex in pyproject.toml, the publish
workflows' release test on the ref name (and the version publish_docker.yml takes from it), and
ci-utils/sync_version.py. A tag any of them misreads publishes the release under the wrong
version or not at all.
"""
import fnmatch
import importlib.util
import json
import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parent.parent.parent
VERSION = "0.12.0"


def _release_tag(version):
    config = json.loads((REPO / "release-please-config.json").read_text(encoding="utf-8"))
    package = config["packages"]["."]
    component = package.get("include-component-in-tag", config.get("include-component-in-tag", True))
    with_v = package.get("include-v-in-tag", config.get("include-v-in-tag", True))
    prefix = package["package-name"] + "-" if component else ""
    return prefix + ("v" if with_v else "") + version


def _scm_tag_regex():
    text = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    raw = re.search(r'^tag_regex\s*=\s*"(.*)"\s*$', text, re.M).group(1)
    return raw.replace("\\\\", "\\")


def _sync_version():
    spec = importlib.util.spec_from_file_location("sync_version", REPO / "ci-utils" / "sync_version.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_release_tag_matches_setuptools_scm_tag_regex_mechanics():
    match = re.match(_scm_tag_regex(), _release_tag(VERSION))
    assert match and match.group("version") == VERSION


def _workflow(name):
    return (REPO / ".github" / "workflows" / name).read_text(encoding="utf-8")


def test_release_tag_is_a_publish_workflow_release_build_mechanics():
    tag = _release_tag(VERSION)
    for name in ("publish_pypi.yml", "publish_cuda11_pypi.yml", "publish_cuda12_pypi.yml"):
        guards = re.findall(r'\[\[ "\$GITHUB_REF_NAME" == (\S+) \]\]', _workflow(name))
        assert guards, name + " has no release test on the ref name"
        for guard in guards:
            assert fnmatch.fnmatchcase(tag, guard), name + ": " + tag + " is not " + guard
    # publish_docker.yml reads a tag exactly as setuptools_scm does
    pattern = re.search(r'\[\[ ! "\$GITHUB_REF_NAME" =~ (\S+) \]\]', _workflow("publish_docker.yml")).group(1)
    assert re.search(pattern, tag).group(1) == VERSION
    for other in ("0." + VERSION, VERSION, "nyxus-v" + VERSION, "v" + VERSION + "-rc.1", "v" + VERSION + ".post1"):
        docker = re.search(pattern, other)
        scm = re.match(_scm_tag_regex(), other)
        assert (docker and docker.group(1)) == (scm and scm.group("version")), other


def test_release_tag_read_by_sync_version_mechanics():
    sync_version = _sync_version()
    version = sync_version.normalize_version(_release_tag(VERSION))
    sync_version.validate_version(version)
    assert version == VERSION


def test_component_tag_refused_by_setuptools_scm_and_sync_version_mechanics():
    # the release-please default, nyxus-v<version>, is the tag these consumers cannot read
    component_tag = "nyxus-v" + VERSION
    assert re.match(_scm_tag_regex(), component_tag) is None
    sync_version = _sync_version()
    try:
        sync_version.validate_version(sync_version.normalize_version(component_tag))
    except SystemExit:
        pass
    else:
        raise AssertionError("sync_version.py accepted " + component_tag)
