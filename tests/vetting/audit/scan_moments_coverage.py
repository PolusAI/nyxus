"""Regenerate moments_2d_coverage.csv by scanning the 2D moments tests. Stdlib only.

    python tests/vetting/audit/scan_moments_coverage.py [--check]

The feature -> test mapping is read out of the test sources rather than written by hand, so the
artifact cannot drift from the tree. `--check` reports drift instead of rewriting, and also runs the
acceptance checks. The checks and the rendering live in scanlib.py.

This family reads the tree its own way, so it overrides scanlib's collect. Nothing here names a
feature on an assertion line: every case passes a golden TABLE to the looping helper
`assert_2d_geomoment_features(...)`, so coverage is resolved table by table -- which function uses
which table, and which features that table holds. Applying the shared line rule would find nothing
at all.
"""
import os
import re
import sys

import scanlib

SOURCES = ["test_2d_moments_skimage.h", "test_2d_moments_regression.h", "test_2d_moments_common.h"]
ORACLE_SUFFIX = {"skimage": "skimage", "pyradiomics": "pyradiomics", "mirp": "mirp",
                 "ibsi": "ibsi", "analytic": "analytic"}

# Block comments only: a `#` line is a preprocessor directive in these headers, not a comment.
COMMENT = re.compile(r"//[^\n]*|/\*.*?\*/", re.S)
TABLE = re.compile(r"ref_vals_list<GeomomentGoldenValue>\s+(\w+)\s*\{(.*?)\n\};", re.S)
ENTRY = re.compile(r'\{\s*Nyxus::Feature2D::(\w+)\s*,\s*"(\w+)"')
FUNC = re.compile(r"^void\s+(test_2d_moments_\w+)\s*\(", re.M)
USES = re.compile(r"assert_2d_geomoment_features\s*\([^,]+,\s*(\w+)")

NORMALIZED = re.compile(r"^(NORM_SPAT_MOMENT|IMOM_NRM)_\d\d$")
WEIGHTED = re.compile(r"^(WEIGHTED_|WT_NORM_CTR_MOM_|IMOM_W)")


def note(feature, cov):
    if NORMALIZED.match(feature):
        return ("normalized raw moment; skimage has no native function - moments_normalized() "
                "is the central-moment quantity")
    if WEIGHTED.match(feature):
        return ("distance-to-contour weighted; dist comes from the approximate min_sqdist, "
                "measured 1.372x over the exact distance, so no tool reproduces it")
    return ""


def collect(fam, feat_re):
    """-> the five coverage maps, resolved through the golden tables rather than assertion lines."""
    covers, where, tables = {}, {}, {}
    for rel in fam.sources:
        with open(os.path.join(scanlib.TESTS, rel), encoding="utf-8", errors="replace") as fh:
            text = COMMENT.sub(lambda m: "\n" * m.group(0).count("\n"), fh.read())
        for name, body in TABLE.findall(text):
            feats = set()
            for m in ENTRY.finditer(body):
                if m.group(1) != m.group(2):
                    print(f"ERROR: {rel}: enum/name mismatch {m.group(1)} vs {m.group(2)}")
                feats.add(m.group(1))
            tables[name] = feats
        marks = [(m.start(), m.group(1)) for m in FUNC.finditer(text)]
        for i, (pos, fn) in enumerate(marks):
            block = text[pos:marks[i + 1][0] if i + 1 < len(marks) else len(text)]
            where[fn] = os.path.basename(rel)
            for tbl in USES.findall(block):
                for feat in tables.get(tbl, ()):
                    covers.setdefault(feat, set()).add(fn)

    asserted, oracles, regression, other = {}, {}, {}, {}
    for feat, fns in covers.items():
        for fn in fns:
            kind = fn.rsplit("_", 1)[-1]
            if kind in ORACLE_SUFFIX:
                asserted.setdefault(feat, set()).add(fn)
                oracles.setdefault(feat, set()).add(ORACLE_SUFFIX[kind])
            elif kind == "regression":
                regression.setdefault(feat, set()).add(fn)
            else:
                other.setdefault(feat, set()).add(fn)
    return asserted, oracles, regression, other, where


FAMILY = scanlib.Family(
    dim="2D", family="moments", out="moments_2d_coverage.csv",
    sources=SOURCES,
    oracle_suffix=ORACLE_SUFFIX,
    notes=note,
    collect_override=collect,
)

if __name__ == "__main__":
    sys.exit(scanlib.run(FAMILY))
