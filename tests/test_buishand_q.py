"""Tests for Buishand's Q statistic test."""

import pytest

import pyhomogeneity as hg


class TestBuishandQTest:
    """Tests for Buishand's Q statistic test."""

    def test_basic_functionality_with_sample_data(self, sample_data):
        """Test basic functionality with known sample data."""
        res = hg.buishand_q_test(sample_data, sim=None)
        assert res.h is None
        assert res.cp == 298
        assert res.p is None
        assert res.Q == 0.5955457285563376
        assert res.avg.mu1 == 157.87285223367698
        assert res.avg.mu2 == 120.93548387096774

    def test_simple_data(self, simple_data):
        """Test with simple data."""
        res = hg.buishand_q_test(simple_data, sim=None)
        assert res.cp is not None
        assert res.Q > 0

    def test_empty_data_raises_error(self):
        """Test that empty data raises ValueError."""
        with pytest.raises(ValueError, match="No valid data remaining"):
            hg.buishand_q_test([])

    def test_constant_data_raises_error(self):
        """Test that constant data raises ValueError."""
        with pytest.raises(ValueError, match="standard deviation is zero"):
            hg.buishand_q_test([5.0, 5.0, 5.0, 5.0])

    def test_invalid_alpha_raises_error(self, simple_data):
        """Test that invalid alpha raises ValueError."""
        with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
            hg.buishand_q_test(simple_data, alpha=0.0)
