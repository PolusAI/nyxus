"""Authoritative test-coverage scan: for every enum feature in features.csv, search all
test files for the exact feature token (word-boundary, dimension-aware) and record which
files reference it. Emits audit_scan.txt (pipe format, picked up by merge.py's audit_*.txt
glob) so the merge can upgrade agent-sampled 'NOT TESTED' false-negatives to their real tier.

Tier is assigned by which file references the feature:
  test_ibsi_*            -> ibsi        (external reference numbers)
  test_compat_3d_*       -> pyradiomics (pyradiomics reference)
  test_*_oracle.{py,h}   -> pyradiomics / analytic (named external / closed-form)
  test_intensity_histogram, test_feature_oracle, test_convex_hull_invariants,
  test_fractal_dim_oracle, test_gabor_truth -> analytic
  harness/loader/driver files -> skipped (not evidence of a value check)
  everything else        -> regression  (self-snapshot golden numbers)

merge.py already takes the STRONGEST tier across all audit_*.txt, so the agents' nuanced
analytic/pyradiomics/imagej calls still win where they logged them; this scan only fills gaps.
"""
import re, csv, glob
from pathlib import Path

BASE = Path(__file__).parent
# repo-relative: this script lives at <repo>/tests/vetting/audit/
TESTS = Path(__file__).resolve().parents[2]

def file_tier(b):
    if b.startswith("test_ibsi_"):        return "ibsi"
    if b.startswith("test_compat_3d_"):   return "pyradiomics"
    if b in ("test_glcm_oracle.py", "test_gldm_oracle.py",
             "test_glcm_oracle.h", "test_gldm_oracle.h"): return "pyradiomics"
    if b in ("test_fractal_dim_oracle.py", "test_feature_oracle.py",
             "test_convex_hull_invariants.py",
             "test_intensity_histogram.py", "test_intensity_histogram.h",
             "test_gabor_truth.h"):        return "analytic"
    # harness / loaders / data / driver: referencing a feature here is not a value check
    if b in ("test_all.cc", "test_feature_calculation.h", "test_main_nyxus.h",
             "test_initialization.h", "test_data.h", "test_data.py",
             "test_dsb2018_data.h", "test_tissuenet_data.py", "test_arrow.h",
             "test_arrow_file_name.h", "test_tiff_loader.h", "test_tiff_loader.py",
             "test_omezarr.h", "test_3d_nifti.h", "test_roi_blacklist.h"):
        return None
    return "regression"

# load test file texts once
texts = {}
for p in list(TESTS.glob("*.h")) + list(TESTS.glob("*.cc")) + list((TESTS / "python").glob("*.py")):
    texts[p.name] = p.read_text(errors="replace")

feats = list(csv.DictReader((BASE / "features.csv").open()))

def token(r):
    # test strings use the '3'-prefixed exposed name for 3D features
    return ("3" + r["feature"]) if r["dim"] == "3D" else r["feature"]

lines = []
for r in feats:
    tok = token(r)
    pat = re.compile(r"(?<![A-Za-z0-9_])" + re.escape(tok) + r"(?![A-Za-z0-9_])")
    for fname, txt in texts.items():
        if pat.search(txt):
            t = file_tier(fname)
            if t is None:
                continue
            lines.append(f"{tok} | {t} | {fname}")

(BASE / "audit_scan.txt").write_text("\n".join(lines) + "\n")
print(f"scan emitted {len(lines)} feature-file evidence lines -> audit_scan.txt")

# quick coverage stat
seen = {l.split("|")[0].strip() for l in lines}
print(f"distinct features with >=1 value-check test file: {len(seen)} / {len(feats)}")
