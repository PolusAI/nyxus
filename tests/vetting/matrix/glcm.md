# GLCM config matrix

Axes = the settings GLCM reads; verdicts (VALID/INVALID) are measured empirically per SPEC 5.2.

| ibsi | symmetric | binning | verdict | recipe / oracle |
|---|---|---|---|---|
| True | True | identity (levels=distinct) | VALID | glcm.ibsi_identity — ibsi/mirp |
| False | True | fixed bin count | VALID | glcm.pyradiomics_symmetric — pyradiomics |
| False | False | MATLAB-style, 64 levels (default) | VALID for CLUPROM, CLUSHADE, CLUTEND, SUMVARIANCE, JVAR, VARIANCE, INFOMEAS1; the rest prod-only | glcm.analytic_default_asymmetric — analytic (IBSI definitions on the asymmetric matrix) |
| any | any | offset/distance = 0 | INVALID | degenerate self-cooccurrence — mechanics guard, not an oracle |

3D: same axes; a tool recipe MUST force symmetric + 13-direction (Nyxus 3D GT default is asymmetric/1-offset). The default asymmetric cell is vetted for the same seven features by glcm3d.analytic_default_asymmetric.
