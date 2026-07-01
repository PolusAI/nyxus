# Changelog

## [1.0.0](https://github.com/PolusAI/nyxus/compare/0.11.0...v1.0.0) (2026-07-01)


### ⚠ BREAKING CHANGES

* **omezarr:** OME-Zarr support now requires z5 3.0.1 and C++20; xtensor/xsimd are no longer used.

### Features

* add 2D test coverage for touch and morphology feature families ([3f576d9](https://github.com/PolusAI/nyxus/commit/3f576d9bd54f72094a3d12a04ea72119e5152593))
* add IBSI Intensity Histogram features, per-bin HISTOGRAM output, and --mergerois ([5b517f4](https://github.com/PolusAI/nyxus/commit/5b517f473dc7fab850845cb3346eb8eee76da8e7))
* add unittest coverage to all 214 3D features, labeling tests with clear ground truth ([b89a9ac](https://github.com/PolusAI/nyxus/commit/b89a9accd7220e9ebed377bfe1596dd69f4c1118))
* Benchmark area fix ([#345](https://github.com/PolusAI/nyxus/issues/345)) ([b532f21](https://github.com/PolusAI/nyxus/commit/b532f21dd636f1176254b003f0b4eca63de1b993))
* **omezarr:** migrate to z5 3.0.1 ArrayView API, drop xtensor ([225b9b9](https://github.com/PolusAI/nyxus/commit/225b9b917103f79dc8d27ff7ce4c2390c60db3ae))
* re-organize and create complete implementation unit test coverage for 492 2D features ([#346](https://github.com/PolusAI/nyxus/issues/346)) ([011df60](https://github.com/PolusAI/nyxus/commit/011df6091c76e257b4159a324a63b0c7c0bf2d0b))


### Bug Fixes

* audit and fix 2D feature tests 1/n ([c5cf260](https://github.com/PolusAI/nyxus/commit/c5cf260353a5cf5e118d095f72dddbf401693d0a))
* categorize and name feature unit tests and expected values ([d854679](https://github.com/PolusAI/nyxus/commit/d854679ef2a67bdde682b8bf5e37a3c4b63b6d40))
* cleare state and start fresh to avoid random test failure.  ([495f05c](https://github.com/PolusAI/nyxus/commit/495f05c29101fe8fa3679784b2546c2de4f04e98))
* emit per-bin HISTOGRAM values in arrow/parquet output ([01ac9c2](https://github.com/PolusAI/nyxus/commit/01ac9c24c0d85ca88b7ba10c09fec74416ccbdb4))
* handle multi-valued HISTOGRAM in buffer output (Python pandas/arrow) ([246f4dd](https://github.com/PolusAI/nyxus/commit/246f4dd3efaa7a36100a79cf7a7050ab1f3e125d))
* issue 327 single roi polus ([#330](https://github.com/PolusAI/nyxus/issues/330)) ([03eeb0c](https://github.com/PolusAI/nyxus/commit/03eeb0c7d25633483c051a04ddcd43192c833315))
* make imagecodecs PIP_ONLY_BINARY ([b5fa20a](https://github.com/PolusAI/nyxus/commit/b5fa20a80d0f66f7be40248f9dc33248ee94cd16))
* migrate build_and_test from macos-15-intel to macos-14 ([ac70de1](https://github.com/PolusAI/nyxus/commit/ac70de14cbc74d29a77d5e4b7a7330a8ac89657d))
* mitigate exposed bug due to incorrect reading of bx in 3d_surface ([24d18ee](https://github.com/PolusAI/nyxus/commit/24d18eeb15f334027b2ba671426397a7aba483ea))
* remove redundant calculations and avoid flushing valid old values ([cdf48cf](https://github.com/PolusAI/nyxus/commit/cdf48cfc5ab345a1013bc7394dff0734e72c415f))
* review comments on PR 352 ([ec42edc](https://github.com/PolusAI/nyxus/commit/ec42edc4e6a400f98ad33428c0b43b4eb0115e05))
* stop featurize() mutating its shared default name lists ([372172e](https://github.com/PolusAI/nyxus/commit/372172e9045d92ef1e57936eae5a20ca05a2d5bb))
* use raw docstrings to silence invalid escape sequence SyntaxWarning ([e3bf05c](https://github.com/PolusAI/nyxus/commit/e3bf05cbe670739c7e629542342621cdfd25cf1c))
