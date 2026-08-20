# Regenerating the 2D NGTDM goldens

Concrete steps for each of the family's three tables. Everything runs offline; CI never invokes a
reference tool.

## 1. `test_2d_ngtdm_mirp.h` — the mirp oracle

**Environment.** A conda env with mirp 2.6.0:

```
conda create -n nyxus_mirp -c conda-forge python=3.11 numpy
conda activate nyxus_mirp
pip install mirp==2.6.0
```

**Run.**

```
python tests/vetting/oracles/gen_ngtdm_mirp.py        # from the repository root
```

Prints two paste-ready blocks — the four-slice means and the per-slice values keyed
`<feature>_z<slice>` — then re-verifies **both** tables already pinned in the header, reports any
value it produces that the header pins nothing for, and exits non-zero on any failure. Paste the two
blocks into `ngtdm_2d_mirp_ref_vals` and `ngtdm_2d_mirp_slice_ref_vals`, then run it again: it must
report `25 verified, 0 failed, 0 unproducible, 0 unpinned`.

**Fixture.** The four IBSI digital-phantom slices are read straight out of `tests/test_data.h` by
`tests/vetting/oracles/ibsi_phantom.py`, so the oracle and the C++ tests cannot drift onto different
inputs.

**Config mapping.**

| Nyxus | mirp |
|---|---|
| `IBSI=true` + `GREYDEPTH=128` | `base_discretisation_method="none"` — the phantom is already discrete 1..6, so both sides work on the raw levels |
| one slice per `calculate()` call, values averaged afterwards | `by_slice=True`, one `extract_features` call per slice |
| no distance setting | none — the neighbourhood is the d=1 8-neighbourhood on both sides |

**Name mapping.** Unusually direct for this framework — mirp abbreviates the family as `ngt` and
spells each feature out: `ngt_coarseness`, `ngt_contrast`, `ngt_busyness`, `ngt_complexity`,
`ngt_strength`, each suffixed `_2d` under `by_slice=True`. No meaning-vs-spelling traps here, unlike
GLSZM or NGLDM.

**Known convention differences.** None found — Nyxus, mirp and PyRadiomics agree to 3.2e-16. This
family has no entropy term, so the `fast_log10` approximation that costs 2D GLDM and 2D GLSZM their
exact tier does not arise.

### Corroborating with PyRadiomics

Not pinned anywhere — the two tools agree to 1.6e-16, so a second table would be redundancy rather
than coverage. Worth re-running when the goldens change, because it is what distinguishes "Nyxus
disagrees with a reference" from "Nyxus disagrees with one reference's convention":

```
conda activate nyxus_oracle       # pyradiomics 3.0.1, needs python <= 3.9
```

```python
from radiomics import ngtdm
import SimpleITK as sitk, numpy as np

# per phantom slice: arr = intensity[None, :, :], roi = (mask > 0)[None, :, :]
img, msk = sitk.GetImageFromArray(arr), sitk.GetImageFromArray(roi.astype(np.uint8))
img.SetSpacing((1., 1., 1.)); msk.SetSpacing((1., 1., 1.))
f = ngtdm.RadiomicsNGTDM(img, msk, binWidth=1, label=1,
                         force2D=True, force2Ddimension=0,
                         interpolator=None, resampledPixelSpacing=None)
f.enableAllFeatures()
out = f.execute()      # Coarseness, Contrast, Busyness, Complexity, Strength
```

`binWidth=1` is identity binning on this integer image, which is how PyRadiomics reaches the same
"no discretisation" point as mirp's `base_discretisation_method="none"` and Nyxus' IBSI mode.

## 2. `test_2d_ngtdm_ibsi.h` — the published consensus

Not generated: transcribed from the IBSI reference manual's digital-phantom table,
<https://ibsi.readthedocs.io/en/latest/03_Image_features.html>, dataset "dig phantom", aggregation
method "2D, averaged". Take the values exactly as published, at **three significant figures** — a
longer literal in this table is a run pasted under a published-value name, and the review check for
that is to count the digits. `rel=1e-2` follows from the three figures.

## 3. `test_2d_ngtdm_regression.h` — the default-mode drift pins

Not an oracle table: it records what Nyxus itself computes in **default** mode (`IBSI=false`) at
**`n_levels = 100`**. Nothing external reproduces those values, so they are a drift guard only.

**The grey count is part of the recipe, not an incidental.** `NGTDMFeature::n_levels` is a static,
and these pins exist only at 100:

| config | `NGTDM_CONTRAST`, four-slice mean |
|---|---|
| IBSI mode | 0.9252630 |
| default mode, `n_levels = 100` | 3169.9291 |
| default mode, `n_levels = 0` | 6634.5048 |

To re-record after a deliberate change, print the four-slice mean per feature at that config — the
same helper the tests use,
`ngtdm_2d_phantom_slice_values(feature, make_ngtdm2d_settings(false), 100)` — and paste the full
`%.17g` value. Do not round: the band is `agrees_gt`'s `rel=1e-3` default, and a pin truncated to six
digits eats a third of it before the test starts.

Re-recording these is a **deliberate act**, not a fix for a red test.

## What is deliberately not in any table

`test_2d_ngtdm_mechanics.h` holds no reference data. It asserts that IBSI mode ignores
`NGTDMFeature::n_levels` (bit-exactly, at 0 and at 100) and that the shared fixture restores the
static it borrows. Those are properties of the machinery, checked without comparing against any
reference, so the file carries `_mechanics` names and contributes no oracle token.
