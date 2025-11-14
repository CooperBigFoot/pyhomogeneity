"""Tests for Buishand's range test."""

import pytest

import pyhomogeneity as hg


class TestBuishandRangeTest:
    """Tests for Buishand's range test."""

    def test_basic_functionality_with_sample_data(self, sample_data):
        """Test basic functionality with known sample data."""
        res = hg.buishand_range_test(sample_data, sim=None)
        assert res.h is None
        assert res.cp == 298
        assert res.p is None
        assert res.R == 0.9893156056266303
        assert res.avg.mu1 == 157.87285223367698
        assert res.avg.mu2 == 120.93548387096774

    def test_simple_data(self, simple_data):
        """Test with simple data."""
        res = hg.buishand_range_test(simple_data, sim=None)
        assert res.cp is not None
        assert res.R > 0

    def test_empty_data_raises_error(self):
        """Test that empty data raises ValueError."""
        with pytest.raises(ValueError, match="No valid data remaining"):
            hg.buishand_range_test([])

    def test_constant_data_raises_error(self):
        """Test that constant data raises ValueError."""
        with pytest.raises(ValueError, match="standard deviation is zero"):
            hg.buishand_range_test([5.0, 5.0, 5.0, 5.0])
