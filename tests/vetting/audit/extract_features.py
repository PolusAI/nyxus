"""Extract the real feature list from src/nyx/featureset.h (Feature2D, Feature3D, FeatureIMQ),
tracking the family from the // section comments. Emits features.csv: dim,family,feature."""
import re, csv
from pathlib import Path

# repo-relative: this script lives at <repo>/tests/vetting/audit/
SRC = Path(__file__).resolve().parents[3] / "src" / "nyx" / "featureset.h"
txt = SRC.read_text(errors="replace")

def enum_body(name):
    m = re.search(r"enum class %s[^{]*\{(.*?)\n\t\};" % name, txt, re.S)
    if not m:
        m = re.search(r"enum class %s[^{]*\{(.*?)\};" % name, txt, re.S)
    return m.group(1) if m else ""

# non-feature sentinels / group markers to exclude
SKIP = re.compile(r"^(_.*|.*__.*|.*_FIRST$|.*_LAST$|ALL_.*|.*_ALL_.*)$")

rows = []
for dim, ename in [("2D", "Feature2D"), ("3D", "Feature3D"), ("IMQ", "FeatureIMQ")]:
    body = enum_body(ename)
    family = "?"
    # track #if 0 preprocessor blocks: enum values inside them are compiled OUT
    # ("planned for a future PR") and are NOT real features. Respect nesting.
    skip_depth = 0     # >0 means we're inside a disabled #if 0 region
    nest = 0           # nested #if within a disabled region, to match the right #endif
    for raw in body.splitlines():
        line = raw.strip()
        # preprocessor handling first
        if re.match(r"#\s*if\b", line):
            if skip_depth:
                nest += 1
            elif re.match(r"#\s*if\s+0\b", line):
                skip_depth = 1
            continue
        if re.match(r"#\s*endif\b", line):
            if skip_depth:
                if nest:
                    nest -= 1
                else:
                    skip_depth = 0
            continue
        if re.match(r"#\s*(else|elif)\b", line):
            continue
        if skip_depth:
            continue
        cm = re.match(r"//\s*(.*)", line)
        if cm:
            fam = cm.group(1).strip().rstrip(":").strip()
            if fam:
                family = fam
            continue
        # strip trailing line comments
        line = re.sub(r"//.*$", "", line).strip().rstrip(",")
        if not line:
            continue
        # a line may hold several comma-separated enum names, some with '= value'
        for tok in line.split(","):
            tok = tok.strip()
            if not tok:
                continue
            nm = tok.split("=")[0].strip()
            if not re.match(r"^[A-Z][A-Z0-9_]*$", nm):
                continue
            if SKIP.match(nm):
                continue
            rows.append((dim, family, nm))

out = Path(__file__).parent / "features.csv"
with out.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["dim", "family", "feature"])
    w.writerows(rows)

from collections import Counter
print("total features:", len(rows))
for dim in ["2D", "3D", "IMQ"]:
    print(f"  {dim}: {sum(1 for r in rows if r[0]==dim)}")
print("families (2D):", sorted(set(r[1] for r in rows if r[0]=='2D')))
