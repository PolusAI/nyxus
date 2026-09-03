"""Run every family scanner's `--check`, report all of them, then fail once. Stdlib only.

    python tests/vetting/audit/run_scanners.py

The twenty scanners share one library, so a change that breaks the library should report all twenty
families rather than stopping at the first. In a shell that means accumulating a status across a
loop and exiting on it at the end, which the CI step did as

    rc=0; for s in ...; do python3 "$s" --check || rc=1; done; exit $rc

and which failed the step twice with twenty `clean` scanners logged under it and nothing on stderr.
The accumulator belongs in the process that already knows the answer: this file runs the same
declarations the per-family entry points do, through the same `scanlib.run`, and its exit status is
an ordinary Python one -- the shape every other check step in that workflow already uses.

A family that raises is caught and counted rather than allowed to abort the run, for the same
reason the loop exists: one broken declaration must not hide the other nineteen.
"""
import glob
import importlib
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import scanlib                                                      # noqa: E402


def main():
    paths = sorted(glob.glob(os.path.join(HERE, "scan_*_coverage.py")))
    if not paths:
        print("ERROR: no scan_*_coverage.py found next to this file")
        return 1

    failed = []
    for path in paths:
        name = os.path.basename(path)
        print(f"== {name}")
        try:
            fam = getattr(importlib.import_module(name[:-3]), "FAMILY", None)
            if fam is None:
                print(f"ERROR: {name} declares no FAMILY")
                failed.append(name)
                continue
            if scanlib.run(fam, ["--check"]):
                failed.append(name)
        except Exception:                                           # noqa: BLE001
            traceback.print_exc()
            failed.append(name)

    print(f"\n{len(paths) - len(failed)} of {len(paths)} families clean")
    for name in failed:
        print("FAILED:", name)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
