"""Join the extracted feature list (features.csv) with the 4 agents' per-feature audits
(audit_*.txt, pipe format: FEATURE | ORACLE_TYPE | TEST_FILE | evidence) into a single
vetting report CSV. Determines per feature: tested?, vetting status, oracle(s), test file(s),
and a candidate oracle to move toward vetting."""
import csv, re, glob
from pathlib import Path
from collections import defaultdict
import sys
sys.path.insert(0, str(Path(__file__).parent))
from candidate_oracles import candidate_oracle

BASE = Path(__file__).parent
# tier per oracle token (parsed from the full oracle string, lowercased)
#   strong  = compared to an external tool's reference numbers (real V&V)
#   analytic= compared to a closed-form / hand-computed / known-invariant value
#   claimed = framework marks it "verifiable-3p" but no tool is named in the code (doc gap)
#   reg     = self-referential regression snapshot
#   cov     = only checks it computes / is finite / in-bounds
def tier(oracle):
    o = oracle.strip().lower()
    if o in ("ibsi", "pyradiomics", "skimage", "imagej", "scipy") or \
       o.startswith("other3p:matlab") or o.startswith("other3p:ibsi") or \
       o.startswith("other3p:rayryeng") or o == "matlab":
        return "strong"
    if o == "analytic":
        return "analytic"
    if o.startswith("other3p:builtin") or o == "builtin" or o.startswith("other3p"):
        return "claimed"
    if o == "regression":
        return "reg"
    if o == "coverage":
        return "cov"
    return "claimed"

RANK = {"strong": 5, "analytic": 4, "claimed": 3, "reg": 2, "cov": 1}
# pretty oracle label kept for the report
def oracle_label(oracle):
    o = oracle.strip()
    return o

def group(feature, family):
    """coarse group for pivoting."""
    h = (feature + " " + family).upper()
    if re.search(r"GLCM|GLRLM|GLSZM|GLDM|GLDZM|NGTDM|NGLDM|GABOR", h):
        return "texture"
    if re.search(r"\bIH_|HISTOGRAM|MOMENT|\bHU_|IMOM|MU[0-9]|NU[0-9]", h):
        return "moments/histogram" if re.search(r"MOMENT|HU|IMOM|MU[0-9]|NU[0-9]", h) else "intensity-histogram"
    if re.search(r"FERET|MARTIN|NASSENSTEIN|CHORD|CONVEX|SOLIDITY|ZERNIKE|EXTREMA|POLYGON|HEXAGON|"
                 r"NEIGHBOR|TOUCH|ANG_BW|RADIAL|FRAC_AT|MEAN_FRAC|GEODETIC|THICKNESS|EROSION|"
                 r"ROI_RADIUS|FRACT_DIM|CALIPER|Chords|Neighbor|convex|radius|ellipticity", h, re.I):
        return "morphology-extended"
    if re.search(r"AREA|PERIMETER|CENTROID|BBOX|EXTENT|ASPECT|ORIENTATION|ECCENTRIC|ELONGAT|AXIS|"
                 r"EULER|CIRCULARITY|COMPACTNESS|ROUNDNESS|DIAMETER|MORPHOLOG|MESH|SPHERIC|VOLUME|"
                 r"FLATNESS|LEAST_AXIS|EDGE_", h):
        return "morphology"
    if re.search(r"FOCUS|SHARP|SATURATION|POWER_SPECTRUM|BRISQUE|NOISE|BLUR|IMQ", h):
        return "image-quality"
    return "intensity"

def base(name):
    """strip a leading 3D '3' prefix + non-alnum, for matching within a dimension."""
    n = name.strip().upper()
    n = re.sub(r"^3(?=[A-Z])", "", n)          # 3GLCM_CONTRAST -> GLCMCONTRAST
    n = re.sub(r"[^A-Z0-9]", "", n)
    return n

def is3d(name):
    return bool(re.match(r"^3[A-Z]", name.strip().upper()))

# --- load features ---
feats = []
with (BASE / "features.csv").open() as f:
    for r in csv.DictReader(f):
        feats.append(r)
# key = (is-3D, base-name) so 2D COV and 3D 3COV don't collide
by_key = defaultdict(list)
for r in feats:
    by_key[(r["dim"] == "3D", base(r["feature"]))].append(r)

# --- parse agent audits ---
# feature_norm -> list of (oracle_type, test_file, evidence)
audit = defaultdict(list)
unmatched = []
for path in sorted(glob.glob(str(BASE / "audit_*.txt"))):
    for line in Path(path).read_text(errors="replace").splitlines():
        if line.strip().startswith("==="):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 3:
            continue
        feat, oracle, tf = parts[0], parts[1], parts[2]
        ev = parts[3] if len(parts) > 3 else ""
        if not re.match(r"^[A-Za-z0-9]", feat):
            continue
        k = (is3d(feat), base(feat))
        audit[k].append((oracle.strip(), tf, ev))
        if k not in by_key:
            unmatched.append((feat, oracle, tf))

# --- build report rows ---
rows = []
for r in feats:
    k = (r["dim"] == "3D", base(r["feature"]))
    hits = audit.get(k, [])
    files = sorted({tf for _, tf, _ in hits})
    if not hits:
        status = "NOT TESTED"
        oracle_used = ""
    else:
        tiers = [(tier(o), o) for o, _, _ in hits]
        bt = max((t for t, _ in tiers), key=lambda t: RANK[t])
        if bt == "strong":
            status = "VETTED"
            oracle_used = ",".join(sorted({o for t, o in tiers if t == "strong"}))
        elif bt == "analytic":
            status = "VETTED"
            oracle_used = "analytic (hand-computed / known)"
        elif bt == "claimed":
            status = "VETTED-CLAIMED (3p oracle not named in code)"
            oracle_used = ",".join(sorted({o for t, o in tiers if t == "claimed"}))
        elif bt == "reg":
            status = "REGRESSION-ONLY"
            oracle_used = "regression (self-snapshot)"
        else:
            status = "COVERAGE-ONLY"
            oracle_used = "coverage (computes/finite only)"
    cand, note = candidate_oracle(r["feature"], r["family"])
    rows.append({
        "dim": r["dim"], "group": group(r["feature"], r["family"]),
        "family": r["family"], "feature": r["feature"],
        "vetting_status": status, "oracle_used": oracle_used,
        "test_files": ";".join(files),
        "candidate_oracle": cand, "how_to_vet": note,
    })

out = BASE / "vetting_report.csv"
with out.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["dim", "group", "family", "feature", "vetting_status",
                                      "oracle_used", "test_files", "candidate_oracle", "how_to_vet"])
    w.writeheader()
    w.writerows(rows)

# --- summary ---
from collections import Counter
c = Counter(r["vetting_status"] for r in rows)
print(f"features: {len(rows)}")
for k, v in sorted(c.items(), key=lambda kv: -kv[1]):
    print(f"  {k}: {v}")
print(f"unmatched audit feature names (not in enum): {len(unmatched)}")
for u in unmatched[:25]:
    print("   ?", u)
print(f"\nreport -> {out}")
