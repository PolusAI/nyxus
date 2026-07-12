"""Generate the vetting summary: group x status pivot CSV, the NOT-TESTED / VETTED-CLAIMED /
REGRESSION lists, and a markdown report."""
import csv
from pathlib import Path
from collections import Counter, defaultdict

BASE = Path(__file__).parent
rows = list(csv.DictReader((BASE / "vetting_report.csv").open()))

STATUSES = ["VETTED", "VETTED-CLAIMED (3p oracle not named in code)", "REGRESSION-ONLY",
            "COVERAGE-ONLY", "NOT TESTED"]
SHORT = {"VETTED": "VETTED", "VETTED-CLAIMED (3p oracle not named in code)": "CLAIMED-3P",
         "REGRESSION-ONLY": "REGRESSION", "COVERAGE-ONLY": "COVERAGE", "NOT TESTED": "NOT-TESTED"}

# pivot group x status
piv = defaultdict(Counter)
for r in rows:
    piv[r["group"]][r["vetting_status"]] += 1
with (BASE / "vetting_pivot.csv").open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["group"] + [SHORT[s] for s in STATUSES] + ["total"])
    for g in sorted(piv):
        c = piv[g]
        w.writerow([g] + [c.get(s, 0) for s in STATUSES] + [sum(c.values())])
    tot = Counter()
    for c in piv.values():
        tot.update(c)
    w.writerow(["TOTAL"] + [tot.get(s, 0) for s in STATUSES] + [sum(tot.values())])

def dump(name, pred):
    sub = [r for r in rows if pred(r)]
    (BASE / name).write_text(
        "dim,group,feature,candidate_oracle,how_to_vet\n" +
        "\n".join(f'{r["dim"]},{r["group"]},{r["feature"]},"{r["candidate_oracle"]}","{r["how_to_vet"]}"'
                 for r in sub))
    return sub

not_tested = dump("gap_not_tested.csv", lambda r: r["vetting_status"] == "NOT TESTED")
claimed = dump("gap_claimed_3p.csv", lambda r: r["vetting_status"].startswith("VETTED-CLAIMED"))
regression = dump("gap_regression.csv", lambda r: r["vetting_status"] == "REGRESSION-ONLY")

print("=== group x status ===")
hdr = f'{"group":22} ' + " ".join(f"{SHORT[s]:>10}" for s in STATUSES)
print(hdr)
for g in sorted(piv):
    c = piv[g]
    print(f'{g:22} ' + " ".join(f"{c.get(s,0):>10}" for s in STATUSES))
tot = Counter()
for c in piv.values():
    tot.update(c)
print(f'{"TOTAL":22} ' + " ".join(f"{tot.get(s,0):>10}" for s in STATUSES))

print(f"\nNOT TESTED ({len(not_tested)}):")
for r in not_tested:
    print(f'   [{r["dim"]}] {r["feature"]:34} -> vet with: {r["candidate_oracle"]}')

print(f"\nsizes: not_tested={len(not_tested)} claimed_3p={len(claimed)} regression_only={len(regression)}")
