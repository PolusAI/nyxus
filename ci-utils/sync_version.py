#!/usr/bin/env python3
import argparse
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLUGIN_JSON = ROOT / "plugin.json"
VERSION_H = ROOT / "src" / "nyx" / "version.h"
VERSION_FILE = ROOT / ".version"


def read_git_tag() -> str:
    try:
        tag = subprocess.check_output(
            ["git", "describe", "--tags", "--abbrev=0"],
            cwd=ROOT,
            text=True,
        ).strip()
    except subprocess.CalledProcessError as exc:
        raise SystemExit(
            "Unable to read git tag. Create a tag or pass --version explicitly."
        ) from exc
    return tag


def normalize_version(tag: str) -> str:
    version = tag.strip()
    if version.startswith("v"):
        version = version[1:]
    return version


def validate_version(version: str) -> None:
    if not re.match(r"^\d+\.\d+\.\d+([.-][0-9A-Za-z]+)*$", version):
        raise SystemExit(f"Version '{version}' does not look like semver.")


def update_plugin_json(version: str) -> None:
    text = PLUGIN_JSON.read_text(encoding="utf-8")
    data = json.loads(text)

    new_text, count = re.subn(
        r'(\"version\"\s*:\s*\")[^\"]*(\")',
        f'"version": "{version}"',
        text,
        count=1,
    )
    if count != 1:
        raise SystemExit("Failed to update plugin.json version field.")

    container_id = data.get("containerId", "")
    if isinstance(container_id, str) and ":" in container_id:
        prefix = container_id.rsplit(":", 1)[0]
        new_container_id = f"{prefix}:{version}"
        new_text, count = re.subn(
            r'(\"containerId\"\s*:\s*\")[^\"]*(\")',
            lambda m: f'{m.group(1)}{new_container_id}{m.group(2)}',
            new_text,
            count=1,
        )
        if count != 1:
            raise SystemExit("Failed to update plugin.json containerId field.")

    PLUGIN_JSON.write_text(new_text, encoding="utf-8")


def update_version_file(version: str) -> None:
    VERSION_FILE.write_text(version + "\n", encoding="utf-8")


def update_version_h(version: str) -> None:
    text = VERSION_H.read_text(encoding="utf-8")
    new_text, count = re.subn(
        r'(#define\s+PROJECT_VER\s+\")[^\"]*(\")',
        lambda m: f'{m.group(1)}{version}{m.group(2)}',
        text,
        count=1,
    )
    if count != 1:
        raise SystemExit("Failed to update src/nyx/version.h PROJECT_VER.")
    VERSION_H.write_text(new_text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sync plugin.json and version.h with a release version."
    )
    parser.add_argument(
        "--version",
        help="Version to write (defaults to latest git tag).",
    )
    args = parser.parse_args()

    if args.version:
        version = normalize_version(args.version)
    else:
        version = normalize_version(read_git_tag())

    validate_version(version)
    update_version_file(version)
    update_plugin_json(version)
    update_version_h(version)


if __name__ == "__main__":
    main()
