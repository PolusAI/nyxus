"""Regenerate ngtdm_2d_coverage.csv by scanning the 2D NGTDM tests. Stdlib only.

    python tests/vetting/audit/scan_ngtdm_coverage.py [--check]

The feature -> test mapping is read out of the test sources rather than written by hand, so the
artifact cannot drift from the tree. `--check` reports drift instead of rewriting, and also runs the
acceptance check below. The coverage rule, the scan and the rendering live in scanlib.py.

This family does NOT use scanlib's shared checks. They read `current_test` per feature: every file
that covers the feature must be named, whatever kind it is. This family reads it per KIND -- a row
is answerable for the tests of its own kind and no row for the mechanics ones -- which is stricter
and catches a class the shared reading cannot, an oracle row whose `current_test` names only a drift
guard. Keeping it here rather than pushing a fifth mode into scanlib keeps that difference visible
where the family is declared.
"""
import sys

import scanlib

SOURCES = [
    "test_2d_ngtdm_ibsi.h",
    "test_2d_ngtdm_mirp.h",
    "test_2d_ngtdm_regression.h",
    "test_2d_ngtdm_mechanics.h",
]

# All five features are deliberately asserted by both oracle files. That is not redundancy: the IBSI
# consensus values are published to three significant figures and fix the DEFINITION (rel=1e-2),
# while mirp reproduces Nyxus to 3.2e-16 and fixes the DIGITS (SPEC 7's exact tier, abs=1e-9). See
# audit/ngtdm_2d_mirp_vetting_report.md.
DUAL_ORACLE = ("asserted against both oracles by design: IBSI fixes the definition at its published "
               "3-significant-figure precision, mirp fixes the digits at 3.2e-16, with a PyRadiomics "
               "run corroborating mirp to 1.6e-16")


def disagreements(fam, cov):
    """Each row is answerable for the tests of its own kind, and no row for the mechanics ones."""
    out = []
    mechanics_files = {cov.where[fn] for fns in cov.other.values() for fn in fns}
    for r in cov.rows:
        f = r["feature"]
        claimed = {t for t in r["current_test"].split(";") if t}
        if r["status"] == "vetted":
            files = {cov.where[fn] for fn in cov.asserted.get(f, ())}
            if not cov.asserted.get(f):
                out.append(f"{f}: status=vetted but no oracle test asserts it")
            if r["oracle"] and r["oracle"] not in cov.oracles.get(f, set()):
                out.append(f"{f}: registry oracle={r['oracle']!r} but the tests asserting it are "
                           f"{sorted(cov.oracles.get(f, ())) or 'none'}")
            if not r["oracle"]:
                out.append(f"{f}: status=vetted but the row names no oracle")
        else:
            files = {cov.where[fn] for fn in cov.regression.get(f, ())}
            if r["oracle"]:
                out.append(f"{f}: status={r['status']} but the row names oracle {r['oracle']!r}")
        for stale in sorted(claimed - files):
            if stale in mechanics_files:
                out.append(f"{f}: current_test names {stale}, a mechanics file - it pins no "
                           f"reference value, so no registry row is covered by it")
            else:
                out.append(f"{f}: current_test names {stale}, which covers nothing for it at "
                           f"recipe {r['config_recipe'] or '(none)'}")
        for gap in sorted(files - claimed):
            out.append(f"{f}: {gap} covers it but current_test omits it")

    # the reverse gap: a kind of test with no row to answer for it
    for f in sorted(cov.asserted):
        if not any(r["feature"] == f and r["status"] == "vetted" for r in cov.rows):
            out.append(f"{f}: {sorted(cov.asserted[f])} assert it but no registry row is vetted")
    for f in sorted(cov.regression):
        if not any(r["feature"] == f and r["status"] == "regression" for r in cov.rows):
            out.append(f"{f}: {sorted(cov.regression[f])} pin it but no registry row is a "
                       f"regression one")
    return out


FAMILY = scanlib.Family(
    dim="2D", family="ngtdm", out="ngtdm_2d_coverage.csv",
    sources=SOURCES,
    oracle_suffix={"mirp": "mirp", "ibsi": "ibsi"},
    notes=scanlib.dual_oracle_notes({}, DUAL_ORACLE),
    # the mechanics guards are neither oracle nor regression, and this family names them in a
    # column of their own rather than folding them into the notes
    extra_column="Mechanics",
    scan_helpers=True,
    # the shared per-feature checks are replaced wholesale by the per-kind reading above; the
    # never-runs sweep is the one built-in this family keeps
    checks={"unregistered"},
    extra_problems=disagreements,
)

if __name__ == "__main__":
    sys.exit(scanlib.run(FAMILY))
