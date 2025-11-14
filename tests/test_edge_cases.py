"""Tests for edge cases and boundary conditions."""

import numpy as np

import pyhomogeneity as hg


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_two_point_data(self):
        """Test with minimum valid data (2 points)."""
        data = np.array([1.0, 2.0])
        res = hg.pettitt_test(data, sim=None)
        assert res.cp == 1

    def test_large_dataset(self):
        """Test with large dataset."""
        np.random.seed(42)
        large_data = np.random.normal(100, 15, 10000)
        res = hg.pettitt_test(large_data, sim=100)
        assert res.cp is not None
        assert res.p is not None

    def test_negative_values(self):
        """Test with negative values."""
        data = np.array([-10.0, -5.0, -3.0, 5.0, 10.0, 15.0])
        res = hg.pettitt_test(data, sim=None)
        assert res.cp is not None

    def test_mixed_signs(self):
        """Test with mixed positive and negative values."""
        data = np.array([-5.0, -2.0, 0.0, 2.0, 5.0])
        res = hg.snht_test(data, sim=None)
        assert res.cp is not None

    def test_very_small_values(self):
        """Test with very small values."""
        data = np.array([1e-10, 2e-10, 3e-10, 1e-9, 2e-9, 3e-9])
        res = hg.pettitt_test(data, sim=None)
        assert res.cp is not None

    def test_very_large_values(self):
        """Test with very large values."""
        data = np.array([1e10, 2e10, 3e10, 1e11, 2e11, 3e11])
        res = hg.pettitt_test(data, sim=None)
        assert res.cp is not None
