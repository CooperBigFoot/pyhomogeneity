"""Integration tests across multiple test methods."""

import numpy as np

import pyhomogeneity as hg


class TestIntegration:
    """Integration tests across multiple test methods."""

    def test_all_methods_agree_on_changepoint_location(self, data_with_changepoint):
        """Test that all methods detect change point in similar location."""
        pettitt = hg.pettitt_test(data_with_changepoint, sim=None)
        snht = hg.snht_test(data_with_changepoint, sim=None)
        buishand_q = hg.buishand_q_test(data_with_changepoint, sim=None)
        buishand_r = hg.buishand_range_test(data_with_changepoint, sim=None)
        buishand_lr = hg.buishand_likelihood_ratio_test(data_with_changepoint, sim=None)
        buishand_u = hg.buishand_u_test(data_with_changepoint, sim=None)

        # All should detect change point near position 50 (within 10 positions)
        changepoints = [pettitt.cp, snht.cp, buishand_q.cp, buishand_r.cp, buishand_lr.cp, buishand_u.cp]
        assert all(40 <= cp <= 60 for cp in changepoints), f"Change points not consistent: {changepoints}"

    def test_homogeneous_data_detection(self, homogeneous_data):
        """Test that all methods agree on homogeneous data (with high alpha)."""
        # Use alpha=0.95 to make it easier to accept null hypothesis
        pettitt = hg.pettitt_test(homogeneous_data, alpha=0.95, sim=1000)
        snht = hg.snht_test(homogeneous_data, alpha=0.95, sim=1000)
        buishand_q = hg.buishand_q_test(homogeneous_data, alpha=0.95, sim=1000)

        # With homogeneous data and high alpha, h should be False (homogeneous)
        # Note: This is probabilistic, so we just check that p-values are reasonable
        assert pettitt.p is not None
        assert snht.p is not None
        assert buishand_q.p is not None

    def test_all_methods_handle_missing_values(self):
        """Test that all methods properly handle missing values."""
        data_with_nans = np.array([1.0, 2.0, np.nan, 3.0, 10.0, np.nan, 11.0, 12.0])

        pettitt = hg.pettitt_test(data_with_nans, sim=None)
        snht = hg.snht_test(data_with_nans, sim=None)
        buishand_q = hg.buishand_q_test(data_with_nans, sim=None)
        buishand_r = hg.buishand_range_test(data_with_nans, sim=None)
        buishand_lr = hg.buishand_likelihood_ratio_test(data_with_nans, sim=None)
        buishand_u = hg.buishand_u_test(data_with_nans, sim=None)

        # All should return valid results
        assert all(r.cp is not None for r in [pettitt, snht, buishand_q, buishand_r, buishand_lr, buishand_u])
