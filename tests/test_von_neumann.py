"""Tests for von Neumann Ratio Test."""

import numpy as np
import pandas as pd
import pytest

import pyhomogeneity as hg


class TestVonNeumannTest:
    """Tests for von Neumann ratio homogeneity test."""

    def test_basic_functionality_with_sample_data(self, sample_data):
        """Test basic functionality with known sample data."""
        res = hg.von_neumann_test(sample_data, sim=None)
        assert res.h is None  # No hypothesis without Monte Carlo
        assert res.cp is not None  # Should detect a change point
        assert res.p is None
        assert res.VN > 0  # VN statistic is positive
        assert res.avg is None  # Von Neumann doesn't compute segment means

    def test_simple_data(self, simple_data):
        """Test with simple data."""
        res = hg.von_neumann_test(simple_data, sim=None)
        assert res.cp is not None
        assert res.VN > 0
        assert res.avg is None

    def test_homogeneous_data(self, homogeneous_data):
        """Test with homogeneous data (should have VN ratio near 2.0)."""
        res = hg.von_neumann_test(homogeneous_data, sim=None)
        assert res.VN > 0
        # For random normal data, VN ratio should be close to 2.0
        # Allow wide range since it varies with randomness
        assert 0.5 < res.VN < 4.0

    def test_data_with_changepoint(self, data_with_changepoint):
        """Test with data containing clear change point."""
        res = hg.von_neumann_test(data_with_changepoint, sim=None)
        assert res.cp is not None
        assert res.VN > 0
        # Note: von Neumann is a global test; change point detection is heuristic
        # So we just verify a change point is returned, not its exact location

    def test_empty_data_raises_error(self):
        """Test that empty data raises ValueError."""
        with pytest.raises(ValueError, match="No valid data remaining"):
            hg.von_neumann_test([])

    def test_single_value_raises_error(self):
        """Test that single value raises ValueError."""
        with pytest.raises(ValueError, match="at least 2 samples"):
            hg.von_neumann_test([1.0])

    def test_constant_data_raises_error(self):
        """Test that constant data raises ValueError (zero variance)."""
        with pytest.raises(ValueError, match="standard deviation is zero"):
            hg.von_neumann_test([5.0, 5.0, 5.0, 5.0])

    def test_invalid_alpha_raises_error(self, simple_data):
        """Test that invalid alpha raises ValueError."""
        with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
            hg.von_neumann_test(simple_data, alpha=2.0)

    def test_invalid_alpha_zero_raises_error(self, simple_data):
        """Test that alpha=0 raises ValueError."""
        with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
            hg.von_neumann_test(simple_data, alpha=0)

    def test_invalid_alpha_one_raises_error(self, simple_data):
        """Test that alpha=1 raises ValueError."""
        with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
            hg.von_neumann_test(simple_data, alpha=1.0)

    def test_invalid_alpha_negative_raises_error(self, simple_data):
        """Test that negative alpha raises ValueError."""
        with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
            hg.von_neumann_test(simple_data, alpha=-0.1)

    def test_with_monte_carlo_sim(self, simple_data):
        """Test with Monte Carlo simulation."""
        res = hg.von_neumann_test(simple_data, sim=1000)
        assert res.p is not None
        assert 0 <= res.p <= 1
        assert res.h is not None  # Should have hypothesis result
        assert isinstance(res.h, bool)

    def test_with_monte_carlo_homogeneous(self, homogeneous_data):
        """Test Monte Carlo with homogeneous data."""
        res = hg.von_neumann_test(homogeneous_data, sim=1000, alpha=0.05)
        assert res.p is not None
        assert res.h is not None
        # Homogeneous data should likely not be rejected (but not guaranteed)

    def test_two_point_data(self):
        """Test with minimum valid data (2 points)."""
        data = np.array([1.0, 2.0])
        res = hg.von_neumann_test(data, sim=None)
        assert res.cp is not None
        assert res.VN > 0

    def test_three_point_data(self):
        """Test with three points."""
        data = np.array([1.0, 2.0, 3.0])
        res = hg.von_neumann_test(data, sim=None)
        assert res.cp is not None
        assert res.VN > 0

    def test_negative_values(self):
        """Test with negative values."""
        data = np.array([-10.0, -5.0, -3.0, 5.0, 10.0, 15.0])
        res = hg.von_neumann_test(data, sim=None)
        assert res.cp is not None
        assert res.VN > 0

    def test_with_list_input(self):
        """Test with Python list input."""
        data = [1.0, 2.0, 3.0, 10.0, 11.0, 12.0]
        res = hg.von_neumann_test(data, sim=None)
        assert res.cp is not None
        assert res.VN > 0

    def test_with_pandas_series(self):
        """Test with pandas Series input."""
        data = pd.Series([1.0, 2.0, 3.0, 10.0, 11.0, 12.0])
        res = hg.von_neumann_test(data, sim=None)
        assert res.cp is not None
        assert res.VN > 0

    def test_with_nan_values(self):
        """Test that NaN values are handled correctly."""
        data = np.array([1.0, 2.0, np.nan, 4.0, 5.0, np.nan, 7.0, 8.0])
        res = hg.von_neumann_test(data, sim=None)
        assert res.cp is not None
        assert res.VN > 0
        # Should process 6 valid values

    def test_all_nan_raises_error(self):
        """Test that all NaN values raises ValueError."""
        data = np.array([np.nan, np.nan, np.nan])
        with pytest.raises(ValueError, match="No valid data remaining"):
            hg.von_neumann_test(data)

    def test_sim_zero_skips_monte_carlo(self, simple_data):
        """Test that sim=0 skips Monte Carlo (like sim=None)."""
        res = hg.von_neumann_test(simple_data, sim=0)
        assert res.p is None
        assert res.h is None

    def test_named_tuple_fields(self, simple_data):
        """Test that result has all expected named tuple fields."""
        res = hg.von_neumann_test(simple_data, sim=None)
        assert hasattr(res, "h")
        assert hasattr(res, "cp")
        assert hasattr(res, "p")
        assert hasattr(res, "VN")
        assert hasattr(res, "avg")

    def test_different_alpha_values(self, simple_data):
        """Test with different alpha values."""
        res1 = hg.von_neumann_test(simple_data, alpha=0.01, sim=1000)
        res2 = hg.von_neumann_test(simple_data, alpha=0.10, sim=1000)

        # Both should have valid p-values
        assert res1.p is not None
        assert res2.p is not None

        # Same p-value for same data
        assert res1.p == res2.p

    def test_large_dataset(self):
        """Test with larger dataset."""
        np.random.seed(123)
        data = np.concatenate([np.random.normal(50, 10, 500), np.random.normal(100, 10, 500)])
        res = hg.von_neumann_test(data, sim=500)

        assert res.cp is not None
        assert res.VN > 0
        assert res.p is not None
        # von Neumann change point detection is heuristic; just verify it runs

    def test_monotonic_increasing(self):
        """Test with monotonically increasing data."""
        data = np.arange(1.0, 101.0)
        res = hg.von_neumann_test(data, sim=None)
        assert res.VN > 0
        # Monotonic data should have low VN ratio (high autocorrelation)
        assert res.VN < 2.0

    def test_alternating_pattern(self):
        """Test with alternating pattern."""
        data = np.array([1.0, 10.0, 1.0, 10.0, 1.0, 10.0, 1.0, 10.0])
        res = hg.von_neumann_test(data, sim=None)
        assert res.VN > 0
        # Alternating pattern should have high VN ratio
        assert res.VN > 2.0

    def test_result_immutability(self, simple_data):
        """Test that result is an immutable named tuple."""
        res = hg.von_neumann_test(simple_data, sim=None)
        with pytest.raises(AttributeError):
            res.h = True  # Should not be able to modify
