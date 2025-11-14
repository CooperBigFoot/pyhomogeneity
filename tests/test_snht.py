"""Tests for Standard Normal Homogeneity Test."""

import pytest

import pyhomogeneity as hg


class TestSNHTTest:
    """Tests for Standard Normal Homogeneity Test."""

    def test_basic_functionality_with_sample_data(self, sample_data):
        """Test basic functionality with known sample data."""
        res = hg.snht_test(sample_data, sim=None)
        assert res.h is None
        assert res.cp == 298
        assert res.p is None
        assert res.T == 2.4426594259172947
        assert res.avg.mu1 == 157.87285223367698
        assert res.avg.mu2 == 120.93548387096774

    def test_simple_data(self, simple_data):
        """Test with simple data."""
        res = hg.snht_test(simple_data, sim=None)
        assert res.cp is not None
        assert res.T > 0

    def test_empty_data_raises_error(self):
        """Test that empty data raises ValueError."""
        with pytest.raises(ValueError, match="No valid data remaining"):
            hg.snht_test([])

    def test_single_value_raises_error(self):
        """Test that single value raises ValueError."""
        with pytest.raises(ValueError, match="at least 2 samples"):
            hg.snht_test([1.0])

    def test_constant_data_raises_error(self):
        """Test that constant data raises ValueError (zero variance)."""
        with pytest.raises(ValueError, match="standard deviation is zero"):
            hg.snht_test([5.0, 5.0, 5.0, 5.0])

    def test_invalid_alpha_raises_error(self, simple_data):
        """Test that invalid alpha raises ValueError."""
        with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
            hg.snht_test(simple_data, alpha=2.0)

    def test_with_monte_carlo_sim(self, simple_data):
        """Test with Monte Carlo simulation."""
        res = hg.snht_test(simple_data, sim=1000)
        assert res.p is not None
        assert 0 <= res.p <= 1
