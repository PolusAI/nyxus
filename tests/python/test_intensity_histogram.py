"""
Unit tests for the IBSI Intensity Histogram (IH_*) feature family and the
--mergerois (merge-all-labels) option exposed through the Python API.

A tiny, fully hand-computable single-ROI image is used so the IH values can be
checked against an exact ground truth (the same case as the C++ unit test):

    intensities {1, 1, 3, 5, 7}, one ROI, N = 3 bins over [1, 7] (binWidth = 2)
    freq = [2, 1, 2], probabilities = [0.4, 0.2, 0.4]
      mean = 4.0   variance = 3.2   median = 4.0   uniformity = 0.36
      entropy = 1.5219280949   skewness = 0   min/max/range = 1 / 7 / 6
      min/max idx (1-based) = 1 / 3   num bins = 3   bin size = 2
"""
import math
import numpy as np
import nyxus

# One 2-D image (axis 0 = image index): a single 1x5 ROI, all label 1.
IH_INT = np.array([[[1, 1, 3, 5, 7]]], dtype=np.int32)
IH_SEG = np.array([[[1, 1, 1, 1, 1]]], dtype=np.int32)


class TestIntensityHistogram:

    def test_ih_requires_ibsi(self):
        # With IBSI off the IH family is stripped; MEAN keeps the request non-empty.
        nyx = nyxus.Nyxus(features=["*ALL_IH*", "MEAN"], ibsi=False, coarse_gray_depth=3)
        df = nyx.featurize(IH_INT, IH_SEG)
        ih_cols = [c for c in df.columns if c.startswith("IH_")]
        assert len(ih_cols) == 0
        assert "MEAN" in df.columns

    def test_ih_enabled_with_ibsi(self):
        # With IBSI on, *ALL_IH* emits all 46 IH_* columns.
        nyx = nyxus.Nyxus(features=["*ALL_IH*"], ibsi=True, coarse_gray_depth=3)
        df = nyx.featurize(IH_INT, IH_SEG)
        ih_cols = [c for c in df.columns if c.startswith("IH_")]
        assert len(ih_cols) == 46

    def test_ih_values_integer_domain(self):
        nyx = nyxus.Nyxus(features=["*ALL_IH*"], ibsi=True, coarse_gray_depth=3)
        df = nyx.featurize(IH_INT, IH_SEG)
        assert df.shape[0] == 1
        row = df.iloc[0]

        assert math.isclose(row["IH_NUM_BINS"], 3.0, rel_tol=1e-6)
        assert math.isclose(row["IH_BIN_SIZE"], 2.0, rel_tol=1e-6)
        assert math.isclose(row["IH_MINIMUM_VAL"], 1.0, rel_tol=1e-6)
        assert math.isclose(row["IH_MAXIMUM_VAL"], 7.0, rel_tol=1e-6)
        assert math.isclose(row["IH_RANGE_VAL"], 6.0, rel_tol=1e-6)
        assert math.isclose(row["IH_MEAN_VAL"], 4.0, rel_tol=1e-6)
        assert math.isclose(row["IH_MEDIAN_VAL"], 4.0, rel_tol=1e-6)
        assert math.isclose(row["IH_VARIANCE_VAL"], 3.2, rel_tol=1e-6)
        assert math.isclose(row["IH_UNIFORMITY_VAL"], 0.36, rel_tol=1e-6)
        assert math.isclose(row["IH_ENTROPY_VAL"], 1.5219280949, rel_tol=1e-5)
        assert abs(row["IH_SKEWNESS_VAL"]) < 1e-9
        # 1-based bin indices
        assert math.isclose(row["IH_MINIMUM_IDX"], 1.0, rel_tol=1e-6)
        assert math.isclose(row["IH_MAXIMUM_IDX"], 3.0, rel_tol=1e-6)
        assert math.isclose(row["IH_MEAN_IDX"], 2.0, rel_tol=1e-6)
        # gradients
        assert math.isclose(row["IH_MAX_GRADIENT"], 1.0, rel_tol=1e-6)
        assert math.isclose(row["IH_MIN_GRADIENT"], -1.0, rel_tol=1e-6)

    def test_ih_index_features_within_bins(self):
        N = 3
        nyx = nyxus.Nyxus(features=["*ALL_IH*"], ibsi=True, coarse_gray_depth=N)
        df = nyx.featurize(IH_INT, IH_SEG)
        row = df.iloc[0]
        for col in ["IH_MINIMUM_IDX", "IH_MAXIMUM_IDX", "IH_MEDIAN_IDX",
                    "IH_P10_IDX", "IH_P90_IDX", "IH_MODE_IDX"]:
            assert 1.0 <= row[col] <= float(N)


class TestMergeRois:

    # Two ROIs (labels 1 and 2) on one image.
    INT = np.array([[[1, 1, 3, 5, 7, 2, 4, 6]]], dtype=np.int32)
    SEG = np.array([[[1, 1, 1, 1, 1, 2, 2, 2]]], dtype=np.int32)

    def test_per_label_default(self):
        nyx = nyxus.Nyxus(features=["MEAN"])
        df = nyx.featurize(self.INT, self.SEG)
        assert df.shape[0] == 2   # one row per label

    def test_mergerois_collapses_to_one_roi(self):
        nyx = nyxus.Nyxus(features=["MEAN"], mergerois=True)
        df = nyx.featurize(self.INT, self.SEG)
        assert df.shape[0] == 1   # all foreground merged into a single ROI

    def test_mergerois_excludes_background(self):
        # background (label 0) must stay excluded after merging
        seg_with_bg = np.array([[[1, 1, 0, 0, 2, 2, 2, 2]]], dtype=np.int32)
        intens = np.array([[[5, 5, 9, 9, 5, 5, 5, 5]]], dtype=np.int32)
        nyx = nyxus.Nyxus(features=["MEAN"], mergerois=True)
        df = nyx.featurize(intens, seg_with_bg)
        assert df.shape[0] == 1
        # the two background pixels (value 9) are excluded -> mean over the 6 fg pixels = 5
        assert math.isclose(df.iloc[0]["MEAN"], 5.0, rel_tol=1e-6)
