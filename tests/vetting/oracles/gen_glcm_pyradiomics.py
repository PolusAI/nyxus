"""OFFLINE golden generator for test_glcm_pyradiomics.h (SPEC 6.4).

Runs PyRadiomics on the pinned GLCM fixture under recipe glcm.pyradiomics_symmetric and prints the
golden values to paste into the oracle test. Run offline; CI never invokes it.

Provenance to record at the pinned goldens: tool=pyradiomics, version=<pin>, config=glcm.pyradiomics_symmetric.

STATUS: stub. Filled in during Wave 2 (GLCM template migration).
"""
def main():
    raise SystemExit("stub — implemented in the GLCM template wave (Phase 3 Wave 2)")

if __name__ == "__main__":
    main()
