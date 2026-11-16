#!/usr/bin/env python3
"""Check that PyPI holds every wheel of a release.

The wheel jobs of publish_pypi.yml continue on error, so a release can lack wheels while its run
stays green, and pip then installs an older release on the platforms concerned. publish_pypi.yml
runs this check after the wheel jobs. Usage: check_wheel_grid.py <version>
"""
import argparse
import fnmatch
import json
import sys
import time
import urllib.error
import urllib.request

PYTHONS = ("cp310", "cp311", "cp312", "cp313")
PLATFORMS = {
    "Linux x86_64": "manylinux_*_x86_64",
    "Linux aarch64": "manylinux_*_aarch64",
    "Windows x86_64": "win_amd64",
    "macOS x86_64": "macosx_*_x86_64",
    "macOS arm64": "macosx_*_arm64",
}


def missing_wheels(filenames, version):
    """Name the Python and platform cells of the grid that no file of this version covers."""
    present = set()
    for filename in filenames:
        if not filename.endswith(".whl"):
            continue
        parts = filename[: -len(".whl")].split("-")
        if len(parts) != 5 or parts[1] != version:
            continue
        for tag in parts[4].split("."):
            for label, pattern in PLATFORMS.items():
                if fnmatch.fnmatchcase(tag, pattern):
                    present.add((parts[2], label))
    return [python + " " + label for python in PYTHONS for label in PLATFORMS if (python, label) not in present]


def fetch_filenames(version):
    """List the files PyPI holds for this version; a version PyPI does not know has none."""
    try:
        with urllib.request.urlopen(f"https://pypi.org/pypi/nyxus/{version}/json", timeout=60) as response:
            return [entry["filename"] for entry in json.load(response)["urls"]]
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return []
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description="Check that PyPI holds every wheel of a release.")
    parser.add_argument("version", help="The release's version, as setuptools-scm gives it.")
    # PyPI's JSON API can trail an upload by a few minutes.
    parser.add_argument("--attempts", type=int, default=6)
    parser.add_argument("--delay", type=float, default=30.0, help="Seconds between attempts.")
    args = parser.parse_args(argv)

    version = args.version
    if not version or any(char.isspace() for char in version):
        raise SystemExit(f"'{version}' is not a version.")
    for attempt in range(1, args.attempts + 1):
        missing = missing_wheels(fetch_filenames(version), version)
        if not missing:
            print(f"PyPI holds all {len(PYTHONS) * len(PLATFORMS)} wheels of nyxus {version}.")
            return 0
        print(f"Attempt {attempt}/{args.attempts}: nyxus {version} lacks {len(missing)} wheels.")
        if attempt < args.attempts:
            time.sleep(args.delay)
    print(f"nyxus {version} is missing these wheels on PyPI: " + ", ".join(missing), file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
