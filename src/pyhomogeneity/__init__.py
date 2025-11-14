"""
pyHomogeneity: A Python package for homogeneity tests of time series data.

This package provides various statistical tests for detecting change points
in time series data, useful for climate, hydrological, and environmental analyses.
"""

from .buishand_lr import buishand_likelihood_ratio_test
from .buishand_q import buishand_q_test
from .buishand_range import buishand_range_test
from .buishand_u import buishand_u_test
from .pettitt import pettitt_test
from .snht import snht_test
from .von_neumann import von_neumann_test

__all__ = [
    "pettitt_test",
    "snht_test",
    "buishand_q_test",
    "buishand_range_test",
    "buishand_likelihood_ratio_test",
    "buishand_u_test",
    "von_neumann_test",
]

from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions
