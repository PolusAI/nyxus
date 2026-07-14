"""Validate tests/vetting/oracle_coverage.csv and regenerate coverage_report.md. Stdlib only."""
import csv, os, sys, argparse

COLUMNS = ["dim","feature","family","status","oracle","agreement","config_recipe",
           "tolerance","current_test","target_test","candidate_oracle","flag","source","notes"]
ALLOWED_STATUS = {"vetted","regression","untested"}
ALLOWED_ORACLES = {"pyradiomics","radiomicsj","mirp","matlab","cellprofiler","mitk",
                   "feature2djava","wndcharm","imea","imagej","fraclac","ibsi","analytic","skimage"}

def load_registry(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))

def validate_rows(rows):
    errs = []
    if rows:
        missing = [c for c in COLUMNS if c not in rows[0]]
        if missing:
            errs.append(f"missing columns: {missing}")
            return errs
    for r in rows:
        f = r["feature"]; st = r["status"].strip(); ora = r["oracle"].strip()
        if st not in ALLOWED_STATUS:
            errs.append(f"{f}: bad status {st!r}")
        if ora and ora not in ALLOWED_ORACLES:
            errs.append(f"{f}: oracle token {ora!r} not in SPEC 4 allowed set")
        if st == "vetted" and not ora:
            errs.append(f"{f}: status=vetted but no oracle")
        if st != "vetted" and ora:
            errs.append(f"{f}: status={st} but has oracle {ora!r}")
    return errs
