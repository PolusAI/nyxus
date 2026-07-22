# Changelog

## [0.12.0](https://github.com/PolusAI/nyxus/compare/nyxus-v0.11.0...nyxus-v0.12.0) (2026-07-22)


### ⚠ BREAKING CHANGES

* **omezarr:** OME-Zarr support now requires z5 3.0.1 and C++20; xtensor/xsimd are no longer used.

### Features

* add 2D test coverage for touch and morphology feature families ([3f576d9](https://github.com/PolusAI/nyxus/commit/3f576d9bd54f72094a3d12a04ea72119e5152593))
* add IBSI Intensity Histogram features, per-bin HISTOGRAM output, and --mergerois ([5b517f4](https://github.com/PolusAI/nyxus/commit/5b517f473dc7fab850845cb3346eb8eee76da8e7))
* add unittest coverage to all 214 3D features, labeling tests with clear ground truth ([b89a9ac](https://github.com/PolusAI/nyxus/commit/b89a9accd7220e9ebed377bfe1596dd69f4c1118))
* Benchmark area fix ([#345](https://github.com/PolusAI/nyxus/issues/345)) ([b532f21](https://github.com/PolusAI/nyxus/commit/b532f21dd636f1176254b003f0b4eca63de1b993))
* **helpers:** add ceil_pow2 (tight next power of two) ([d0992dc](https://github.com/PolusAI/nyxus/commit/d0992dcfadec673712aa8f4fb3a9353a6a06224b))
* **hounsfield:** --preserve-hu CT/Hounsfield intensity preservation ([1b5667f](https://github.com/PolusAI/nyxus/commit/1b5667fd85d9f9abcb3bbebdd583fcbee23ab5b0))
* **hounsfield:** keep HU offset base at scanned slide min under fp-options; enable bare CLI --preserve-hu ([43f8063](https://github.com/PolusAI/nyxus/commit/43f80638c65f9682afac75a78ee4327483de1c60))
* **hounsfield:** wire --preserve-hu through the 3D NIfTI loader ([65c9ad9](https://github.com/PolusAI/nyxus/commit/65c9ad9be90f0fc2ad03757b62441d1558d2871c))
* **omezarr:** migrate to z5 3.0.1 ArrayView API, drop xtensor ([225b9b9](https://github.com/PolusAI/nyxus/commit/225b9b917103f79dc8d27ff7ce4c2390c60db3ae))
* re-organize and create complete implementation unit test coverage for 492 2D features ([#346](https://github.com/PolusAI/nyxus/issues/346)) ([011df60](https://github.com/PolusAI/nyxus/commit/011df6091c76e257b4159a324a63b0c7c0bf2d0b))
* **vetting:** check_coverage CLI (--check/--write) + drift guard ([0ad65ee](https://github.com/PolusAI/nyxus/commit/0ad65eeb72e846129630f269b46253f6f2dae607))
* **vetting:** check_coverage registry load + schema/token validation ([44d2bfa](https://github.com/PolusAI/nyxus/commit/44d2bfa834ff40126b96f87ea6ec571da36405ed))
* **vetting:** coverage stats + markdown report generation ([0f85581](https://github.com/PolusAI/nyxus/commit/0f85581e62480a807d4303ded3988b4adf3c659d))


### Bug Fixes

* **3d-shape:** correct mislabeled axis lengths + vet 7 morphology features ([0af100b](https://github.com/PolusAI/nyxus/commit/0af100bc3eb8dd17fa5796b50c22998b5548293d))
* **3d:** compute 3ROBUST_MEAN (was hardcoded 0) + vet 18 3D first-order features ([79d0974](https://github.com/PolusAI/nyxus/commit/79d097404f9b627b720dd49eb53dad43da83e985))
* audit and fix 2D feature tests 1/n ([c5cf260](https://github.com/PolusAI/nyxus/commit/c5cf260353a5cf5e118d095f72dddbf401693d0a))
* **caliper:** correct Martin/Nassenstein diameters + vet 12 features vs imea ([bc3e92a](https://github.com/PolusAI/nyxus/commit/bc3e92a6ec7235a6a42ab35613cd98a3eb092afb))
* **caliper:** float-precision hull rotation + address PR [#381](https://github.com/PolusAI/nyxus/issues/381) review ([3377183](https://github.com/PolusAI/nyxus/commit/33771833d21cd8488f54f330b48edf030b697b8a))
* categorize and name feature unit tests and expected values ([d854679](https://github.com/PolusAI/nyxus/commit/d854679ef2a67bdde682b8bf5e37a3c4b63b6d40))
* **chords:** correct max-chord angle index and all-chords histogram source ([bd9a930](https://github.com/PolusAI/nyxus/commit/bd9a93009154cacfd462bc44aa6689aac6f6794d))
* cleare state and start fresh to avoid random test failure.  ([495f05c](https://github.com/PolusAI/nyxus/commit/495f05c29101fe8fa3679784b2546c2de4f04e98))
* **convex-hull:** Pick's-theorem hull area so SOLIDITY &lt;= 1 ([0319932](https://github.com/PolusAI/nyxus/commit/03199326b71473e06805f6e482fe7c786fbedb4f))
* emit per-bin HISTOGRAM values in arrow/parquet output ([01ac9c2](https://github.com/PolusAI/nyxus/commit/01ac9c24c0d85ca88b7ba10c09fec74416ccbdb4))
* **fractal-dim:** adaptive box counting and a clean divider perimeter ([c981743](https://github.com/PolusAI/nyxus/commit/c9817432087cc9694781663cf365d123d2d38657))
* **fractal-dim:** address PR review comments ([98a4f2d](https://github.com/PolusAI/nyxus/commit/98a4f2d25fa23a4c53cfa5a4a569d2dc48093200))
* **fractal-dim:** align the ROI to the box-counting grid ([98e53ec](https://github.com/PolusAI/nyxus/commit/98e53ec77af6f8da13a462f3835193bf1552e41f))
* **fractal-dim:** mean log-log slope for box-count, Richardson D for perimeter ([af37142](https://github.com/PolusAI/nyxus/commit/af37142895b2f02f6f4ee81962494da790f7b8f3))
* **glcm:** default co-occurrence offset to 1 and exclude background ([02c5a75](https://github.com/PolusAI/nyxus/commit/02c5a75eb900c8f233574f843a5581d81274752a))
* **glcm:** normalize HOM2/ENTROPY by sum_p (2D+3D) + fully oracle-vet GLCM ([60a1ebb](https://github.com/PolusAI/nyxus/commit/60a1ebbefb1ddc51ed1f7fcbbec0529ec65890a7))
* **gldm:** exclude background from the dependence matrix ([f1f8eb4](https://github.com/PolusAI/nyxus/commit/f1f8eb4fdf99fd0c0c1ad29d44337488364371b0))
* handle multi-valued HISTOGRAM in buffer output (Python pandas/arrow) ([246f4dd](https://github.com/PolusAI/nyxus/commit/246f4dd3efaa7a36100a79cf7a7050ab1f3e125d))
* **hexagonality,lstsq:** use std::abs so gcc/Linux doesn't truncate doubles to int ([88a6f53](https://github.com/PolusAI/nyxus/commit/88a6f53f788ab25d901e6e176549a7912c3f8856))
* issue 327 single roi polus ([#330](https://github.com/PolusAI/nyxus/issues/330)) ([03eeb0c](https://github.com/PolusAI/nyxus/commit/03eeb0c7d25633483c051a04ddcd43192c833315))
* **loader:** clamp negative signed-int pixels so int16 CT doesn't wrap to a huge unsigned value ([fac1bac](https://github.com/PolusAI/nyxus/commit/fac1bac7df856c0809f04cd14c13c6c6f11d68f4))
* make imagecodecs PIP_ONLY_BINARY ([b5fa20a](https://github.com/PolusAI/nyxus/commit/b5fa20a80d0f66f7be40248f9dc33248ee94cd16))
* migrate build_and_test from macos-15-intel to macos-14 ([ac70de1](https://github.com/PolusAI/nyxus/commit/ac70de14cbc74d29a77d5e4b7a7330a8ac89657d))
* mitigate exposed bug due to incorrect reading of bx in 3d_surface ([24d18ee](https://github.com/PolusAI/nyxus/commit/24d18eeb15f334027b2ba671426397a7aba483ea))
* **moments:** correct Hu invariant h5/h6 formulas (9x bracket, stray +eta03) ([73f6ca8](https://github.com/PolusAI/nyxus/commit/73f6ca813b2c2fae7c34a7524f8a3cd604aca301))
* **moments:** use fractional centroid in 2D central moments (drop (int) truncation) ([5cbf94a](https://github.com/PolusAI/nyxus/commit/5cbf94af7d46995e4062a4056e327e107fcb4afb))
* **neighbors:** exact min distance for the touch/radius test (PERCENT_TOUCHING) ([92de774](https://github.com/PolusAI/nyxus/commit/92de774785302364eeb48170c4f3eabb9dd73a8e))
* **neighbors:** implement closest-neighbor dist/ang and cap PERCENT_TOUCHING ([4322c88](https://github.com/PolusAI/nyxus/commit/4322c88a45e88825cbfee494b6915126ae979a7f))
* remove redundant calculations and avoid flushing valid old values ([cdf48cf](https://github.com/PolusAI/nyxus/commit/cdf48cfc5ab345a1013bc7394dff0734e72c415f))
* review comments on PR 352 ([ec42edc](https://github.com/PolusAI/nyxus/commit/ec42edc4e6a400f98ad33428c0b43b4eb0115e05))
* stop featurize() mutating its shared default name lists ([372172e](https://github.com/PolusAI/nyxus/commit/372172e9045d92ef1e57936eae5a20ca05a2d5bb))
* upgrade Docker build to CUDA 12.6 and Python 3.11 ([80b548d](https://github.com/PolusAI/nyxus/commit/80b548d42322a301c1da824f7ce55b945aa8da63))
* use raw docstrings to silence invalid escape sequence SyntaxWarning ([e3bf05c](https://github.com/PolusAI/nyxus/commit/e3bf05cbe670739c7e629542342621cdfd25cf1c))


### Performance Improvements

* **omezarr:** open the z5 dataset once, not per tile ([9b36f84](https://github.com/PolusAI/nyxus/commit/9b36f84bad3d86e663005bf35e3301844894e291))


### Documentation

* add CLAUDE.md codebase and workflow guide ([e301fe6](https://github.com/PolusAI/nyxus/commit/e301fe6e9d27650e70264f383cc29c267f46f4dc))
* **caliper:** correct inaccurate/stale comments (no code change) ([cc5b4de](https://github.com/PolusAI/nyxus/commit/cc5b4dea602157d8c85bb7cbb47fda22054a0181))
* **CLAUDE:** add git workflow and tests/vetting conformance rules ([a548b57](https://github.com/PolusAI/nyxus/commit/a548b578b1e248d1c319e550b473decca8e8edaa))
* **CLAUDE:** address PR [#388](https://github.com/PolusAI/nyxus/issues/388) review — C++20; agentic git workflow ([466c038](https://github.com/PolusAI/nyxus/commit/466c038d2f47be5ea9ee000145906c4e3d379914))
* **CLAUDE:** address PR [#388](https://github.com/PolusAI/nyxus/issues/388) review — generalize paths, drop z5py pin, add no-local-paths rule ([4235e0c](https://github.com/PolusAI/nyxus/commit/4235e0c8345a87cbd5229b95950ec85587c82268))
* **test:** correct IQR/QCoD _IDX framing — discrete percentile is IBSI-conformant, not a bug ([29214ba](https://github.com/PolusAI/nyxus/commit/29214ba7ef60c3fa80a4bee66037c53cc161efe2))
* **tests:** add vetting audit baseline, fraclac oracle, and README ([e5232c0](https://github.com/PolusAI/nyxus/commit/e5232c05095d13bac0674fd47fecdeb75070c91a))
* **tests:** box-count golden is FracLac-vetted, not a self-consistency pin ([8ccc76b](https://github.com/PolusAI/nyxus/commit/8ccc76b043391593bd9c2e9f082c435901979ca8))
* **tests:** neighbor features are partly tool-vetted (CellProfiler), not analytic-only ([7910549](https://github.com/PolusAI/nyxus/commit/79105490154d4c9672207e10fcc534df36598497))
* **tests:** oracle tool local-setup research (TOOLS.md) ([65e0b79](https://github.com/PolusAI/nyxus/commit/65e0b790d5df5f498d0068e1de65889f9de283ad))
* **tests:** spec for the oracle-vetting test framework ([f44ff25](https://github.com/PolusAI/nyxus/commit/f44ff251e25e3009c0a6a71c4e1acd49f9f86320))
* **vetting:** add skimage to the SPEC 4 oracle-token set ([3ee1dfd](https://github.com/PolusAI/nyxus/commit/3ee1dfd851ee07c785de5283a6b43ff207cbab62))
* **vetting:** correct split to 16 ibsi + 4 analytic; file IQR_IDX flooring bug ([726b082](https://github.com/PolusAI/nyxus/commit/726b0826c8672b1b7a57cef16b8308e87abaccab))
* **vetting:** design for IBSI-vetted IH dispersion/index tests ([07b1b75](https://github.com/PolusAI/nyxus/commit/07b1b75f2a5476b20f65836fc09fcde7648287f2))
* **vetting:** fix stale token count in plan example (16 ibsi + 4 analytic) ([3255fef](https://github.com/PolusAI/nyxus/commit/3255fef34af9d54e846ad9ae2d617e27de72ba08))
* **vetting:** implementation plan for IBSI-vetted IH dispersion/index tests ([4dfb1ff](https://github.com/PolusAI/nyxus/commit/4dfb1ff9d3b2f98e26379c228e65dd2ec0e08642))
* **vetting:** migration map + seeded oracle_coverage.csv (Phase 1, no code changes) ([77f3d07](https://github.com/PolusAI/nyxus/commit/77f3d07abf1377fcf9248d237deba5de06e1005c))
* **vetting:** record reconciled IBSI IH phantom config + index base ([99df244](https://github.com/PolusAI/nyxus/commit/99df244c1f49977d631561c339bdb5e05d5979a6))
* **vetting:** record the config scope of the gldm promotions (7 are mode-invariant) ([0780912](https://github.com/PolusAI/nyxus/commit/0780912413b5d03290b33cf2bf27ee838c0cbb65))
* **vetting:** record the config scope of the glrlm promotions ([285bc39](https://github.com/PolusAI/nyxus/commit/285bc39be3e7fc9125514a28094e417f98d65802))
* **vetting:** record the config scope of the glszm promotions ([c4bb8b1](https://github.com/PolusAI/nyxus/commit/c4bb8b1f1dafe9c1e81c0be254c7e7016db5936c))
* **vetting:** resolve missing oracles via deep-dive research (Phase 2) ([2fa0515](https://github.com/PolusAI/nyxus/commit/2fa051573417375f533d751bff14482934e9288c))
* **vetting:** resolve the Phase-2 reconciliation decisions (SPEC 6) ([8b45ddd](https://github.com/PolusAI/nyxus/commit/8b45ddd704994b85792166be0282fb905b475003))
* **vetting:** retract unproven 3D NGLDM "ibsi" promotions; document MIRP disagreement ([b924f8e](https://github.com/PolusAI/nyxus/commit/b924f8e488b8bfb08a020ee5754cc148cb18672d))
* **vetting:** scaffold matrix/ and oracles/gen_ conventions (GLCM example) ([61fdc66](https://github.com/PolusAI/nyxus/commit/61fdc668adae748d47990bbd52cc9b229e2f7a35))
* **vetting:** seed config_recipes.md ([b64c82b](https://github.com/PolusAI/nyxus/commit/b64c82b2eb1e9c18392be4e4b525bf940c70cf5d))


### Code Refactoring

* address PR [#351](https://github.com/PolusAI/nyxus/issues/351) review nits ([c623405](https://github.com/PolusAI/nyxus/commit/c6234053f966ee58ce32f2c98a69f8b229706e78))
* **tests:** route fractal fixture through the shared masked-loader helper ([050e672](https://github.com/PolusAI/nyxus/commit/050e67250b5d04b6668aaa79d4337c6247214c49))


### Build System

* **arrow:** require Boost for Arrow instead of Z5 ([43a584f](https://github.com/PolusAI/nyxus/commit/43a584f6ac4cadb963c8fcc02f6197b5aae6724e))


### Continuous Integration

* **release-please:** use manifest config so 0.x breaking = minor, and pin 0.12.0 ([a500bd9](https://github.com/PolusAI/nyxus/commit/a500bd913e94de68db3b6a608f8d4ed66b5f10d5))
