# GLCM config matrix

Axes = the settings GLCM reads; verdicts (VALID/INVALID) are measured empirically per SPEC 5.2.

| ibsi | symmetric | binning | verdict | recipe / oracle |
|---|---|---|---|---|
| True | True | identity (levels=distinct) | VALID | glcm.ibsi_identity — ibsi/mirp |
| False | True | fixed bin count | VALID | glcm.pyradiomics_symmetric — pyradiomics |
| any | any | offset/distance = 0 | INVALID | degenerate self-cooccurrence — mechanics guard, not an oracle |

3D: same axes; the recipe MUST force symmetric + 13-direction (Nyxus 3D GT default is asymmetric/1-offset).
