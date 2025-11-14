"""Tests for Pettitt's homogeneity test."""

import numpy as np
import pytest

import pyhomogeneity as hg


class TestPettittTest:
    """Tests for Pettitt's homogeneity test."""

    def test_basic_functionality_with_sample_data(self, sample_data):
        """Test basic functionality with known sample data."""
        res = hg.pettitt_test(sample_data)
        assert res.cp == 298
        assert res.U == 2716.0
        assert res.avg.mu1 == 157.87285223367698
        assert res.avg.mu2 == 120.93548387096774

    def test_simple_data(self, simple_data):
        """Test with simple data."""
        res = hg.pettitt_test(simple_data, sim=None)
        assert res.cp is not None
        assert res.U > 0
        assert res.avg.mu1 < res.avg.mu2

    def test_returns_named_tuple(self, simple_data):
        """Test that result is accessible as named tuple."""
        res = hg.pettitt_test(simple_data, sim=None)
        # Access by attribute
        assert hasattr(res, "h")
        assert hasattr(res, "cp")
        assert hasattr(res, "p")
        assert hasattr(res, "U")
        assert hasattr(res, "avg")

    def test_empty_data_raises_error(self):
        """Test that empty data raises ValueError."""
        with pytest.raises(ValueError, match="No valid data remaining"):
            hg.pettitt_test([])

    def test_single_value_raises_error(self):
        """Test that single value raises ValueError."""
        with pytest.raises(ValueError, match="at least 2 samples"):
            hg.pettitt_test([1.0])

    def test_all_nan_raises_error(self):
        """Test that all NaN values raises ValueError."""
        with pytest.raises(ValueError, match="No valid data remaining"):
            hg.pettitt_test([np.nan, np.nan, np.nan])

    def test_constant_data_works(self):
        """Test that constant data works (no change point but valid)."""
        res = hg.pettitt_test([5.0, 5.0, 5.0, 5.0], sim=None)
        assert res.U == 0.0  # No variation in ranks

    def test_invalid_alpha_raises_error(self, simple_data):
        """Test that invalid alpha raises ValueError."""
        with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
            hg.pettitt_test(simple_data, alpha=1.5)

        with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
            hg.pettitt_test(simple_data, alpha=-0.1)

    def test_invalid_sim_raises_error(self, simple_data):
        """Test that invalid sim value raises ValueError."""
        with pytest.raises(ValueError, match="Number of simulations must be positive"):
            hg.pettitt_test(simple_data, sim=-100)

    def test_list_input(self):
        """Test that list input works."""
        res = hg.pettitt_test([1, 2, 3, 10, 11, 12], sim=None)
        assert res.cp is not None

    def test_with_monte_carlo_sim(self, simple_data):
        """Test with Monte Carlo simulation for p-value."""
        res = hg.pettitt_test(simple_data, sim=1000)
        assert res.p is not None
        assert 0 <= res.p <= 1
        assert res.h is not None

    def test_hypothesis_result_type(self, simple_data):
        """Test hypothesis result is boolean when sim is provided."""
        res = hg.pettitt_test(simple_data, sim=100)
        assert isinstance(res.h, (bool, np.bool_))
