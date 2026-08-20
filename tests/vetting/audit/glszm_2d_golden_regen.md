# Regenerating the 2D GLSZM goldens

Concrete steps for each of the family's three tables. Everything runs offline; CI never invokes a
reference tool.

## 1. `test_2d_glszm_mirp.h` — the mirp oracle

**Environment.** A conda env with mirp 2.6.0:

```
conda create -n nyxus_mirp -c conda-forge python=3.11 numpy
conda activate nyxus_mirp
pip install mirp==2.6.0
```

**Run.**

```
python tests/vetting/oracles/gen_glszm_mirp.py        # from the repository root
```

The generator prints two paste-ready blocks — the four-slice means and the per-slice values keyed
`<feature>_z<slice>` — and then re-verifies **both** tables already pinned in the header, exiting
non-zero on any mismatch or on any pin it cannot produce. Paste the two blocks into
`glszm_2d_mirp_ref_vals` and `glszm_2d_mirp_slice_ref_vals`, then run the generator again: it must
report `ALL CHECKS PASSED` with 80 verified.

**Fixture.** The four IBSI digital-phantom slices are read straight out of `tests/test_data.h` by
`tests/vetting/oracles/ibsi_phantom.py`, so the oracle and the C++ tests cannot drift onto different
inputs. Each slice is passed to mirp as a `(1, rows, cols)` volume with its own binary mask.

**Config mapping.**

| Nyxus | mirp |
|---|---|
| `IBSI=true` + `GREYDEPTH=128` | `base_discretisation_method="none"` — the phantom is already discrete 1..6, so both sides work on the raw levels |
| one slice per `calculate()` call, values averaged afterwards | `by_slice=True`, one `extract_features` call per slice |
| no zone-size or distance parameter | none — a zone is a maximal 8-connected set of equal-valued voxels on either side |

**Name mapping.** mirp calls the family `szm`, spells "zone size" as `zs` where Nyxus writes `SZN` /
`ZV` / `ZE`, and abbreviates the emphases as `lge` / `hge`. The full map lives in `MIRP_COLUMN` in
the generator; every column is suffixed `_2d` under `by_slice=True`.

| Nyxus | mirp | | Nyxus | mirp |
|---|---|---|---|---|
| `GLSZM_SAE` | `szm_sze_2d` | | `GLSZM_GLN` | `szm_glnu_2d` |
| `GLSZM_LAE` | `szm_lze_2d` | | `GLSZM_GLNN` | `szm_glnu_norm_2d` |
| `GLSZM_LGLZE` | `szm_lgze_2d` | | `GLSZM_SZN` | `szm_zsnu_2d` |
| `GLSZM_HGLZE` | `szm_hgze_2d` | | `GLSZM_SZNN` | `szm_zsnu_norm_2d` |
| `GLSZM_SALGLE` | `szm_szlge_2d` | | `GLSZM_ZP` | `szm_z_perc_2d` |
| `GLSZM_SAHGLE` | `szm_szhge_2d` | | `GLSZM_GLV` | `szm_gl_var_2d` |
| `GLSZM_LALGLE` | `szm_lzlge_2d` | | `GLSZM_ZV` | `szm_zs_var_2d` |
| `GLSZM_LAHGLE` | `szm_lzhge_2d` | | `GLSZM_ZE` | `szm_zs_entr_2d` |

**Known convention difference.** None on the matrix or the definitions — fifteen of the sixteen
features are bit-identical. `GLSZM_ZE` differs by 2.5e-3 because Nyxus takes its logarithm through
`Nyxus::fast_log10`, a float-precision approximation, where mirp uses a double `log2`. That is why
that one feature asserts at `rel=4e-3` and the rest at `rel=1e-9`. If the logarithm is ever switched
to `std::log`, re-run the generator and tighten the band to `rel=1e-9` with the others.

### Corroborating the mirp goldens with PyRadiomics

Not pinned anywhere — the two tools agree to 7.0e-16, so a second table would be redundancy rather
than coverage. Worth re-running when the goldens change, because it is what distinguishes "Nyxus
disagrees with a reference" from "Nyxus disagrees with one reference's convention":

```
conda create -n nyxus_oracle -c conda-forge python=3.9 pyradiomics simpleitk   # -> 3.0.1
conda activate nyxus_oracle
```

```python
from radiomics import glszm
import SimpleITK as sitk, numpy as np

# per phantom slice: arr = intensity[None, :, :], roi = (mask > 0)[None, :, :]
img, msk = sitk.GetImageFromArray(arr), sitk.GetImageFromArray(roi.astype(np.uint8))
img.SetSpacing((1., 1., 1.)); msk.SetSpacing((1., 1., 1.))
f = glszm.RadiomicsGLSZM(img, msk, binWidth=1, label=1,
                         force2D=True, force2Ddimension=0,
                         interpolator=None, resampledPixelSpacing=None)
f.enableAllFeatures()
out = f.execute()          # SmallAreaEmphasis, LargeAreaEmphasis, ... ZoneEntropy
```

`binWidth=1` is identity binning on this integer image, which is how PyRadiomics reaches the same
"no discretisation" point as mirp's `base_discretisation_method="none"` and Nyxus' IBSI mode.
PyRadiomics' names map to Nyxus' by meaning — `SmallAreaEmphasis` → `GLSZM_SAE`,
`SizeZoneNonUniformity` → `GLSZM_SZN`, `ZoneEntropy` → `GLSZM_ZE`, and the four
`{Small,Large}Area{Low,High}GrayLevelEmphasis` → `GLSZM_{SA,LA}{L,H}GLE`.

## 2. `test_2d_glszm_ibsi.h` — the published consensus

Not generated: transcribed from the IBSI reference manual's digital-phantom table,
<https://ibsi.readthedocs.io/en/latest/03_Image_features.html>, dataset "dig phantom", aggregation
method "2D, averaged". Take the values exactly as published, at **three significant figures** — a
longer literal in this table is a run pasted under a published-value name, and the review check for
that is to count the digits. `rel=1e-2` follows from the three figures.

## 3. `test_2d_glszm_regression.h` — the default-mode drift pins

Not an oracle table: it records what Nyxus itself computes in **default** mode (`IBSI=false`,
`GREYDEPTH=64`) on the same four phantom slices. Nothing external reproduces those values, so they
are a drift guard only.

To re-record after a deliberate change, print the four-slice mean per feature at that config — the
same helper the tests use, `glszm_2d_phantom_slice_values(feature, make_glszm2d_settings(false, 64))`
— and paste the full `%.17g` value. Do not round: the band is `agrees_gt`'s `rel=1e-3` default, and a
pin truncated to five digits eats a third of it before the test starts.

Re-recording these is a **deliberate act**, not a fix for a red test. A drift guard going red means
either the change was intended, in which case say so in the commit message, or it was not, in which
case the pin is the finding.
