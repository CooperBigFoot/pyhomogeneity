"""Tests for Buishand's likelihood ratio test."""

import pytest

import pyhomogeneity as hg


class TestBuishandLikelihoodRatioTest:
    """Tests for Buishand's likelihood ratio test."""

    def test_basic_functionality_with_sample_data(self, sample_data):
        """Test basic functionality with known sample data."""
        res = hg.buishand_likelihood_ratio_test(sample_data, sim=None)
        assert res.h is None
        assert res.cp == 298
        assert res.p is None
        assert res.V == 0.08330290132452312
        assert res.avg.mu1 == 157.87285223367698
        assert res.avg.mu2 == 120.93548387096774

    def test_simple_data(self, simple_data):
        """Test with simple data."""
        res = hg.buishand_likelihood_ratio_test(simple_data, sim=None)
        assert res.cp is not None
        assert res.V > 0

    def test_empty_data_raises_error(self):
        """Test that empty data raises ValueError."""
        with pytest.raises(ValueError, match="No valid data remaining"):
            hg.buishand_likelihood_ratio_test([])

    def test_constant_data_raises_error(self):
        """Test that constant data raises ValueError."""
        with pytest.raises(ValueError, match="standard deviation is zero"):
            hg.buishand_likelihood_ratio_test([5.0, 5.0, 5.0, 5.0])
