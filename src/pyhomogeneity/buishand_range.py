"""
Buishand's range test for change point detection.

Reference: Buishand, T. A. (1982). Some methods for testing the homogeneity of rainfall records.
Journal of Hydrology, 58(1-2), 11-27.
"""

from __future__ import annotations

from collections import namedtuple
from typing import Any

import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from ._utils import (
    calculate_means,
    monte_carlo_p_value,
    preprocess,
    remove_missing_values,
    validate_alpha,
    validate_variance,
)


def _buishand_range_statistic_jax(x, alpha: float = 0.05):
    """JAX-native Buishand range statistic calculation for Monte Carlo simulations.

    Args:
        x: Input data array (JAX array)
        alpha: Significance level (not used in calculation)

    Returns:
        R statistic normalized by sqrt(n) (JAX scalar)
    """
    n = len(x)
    std = jnp.std(x)

    k = jnp.arange(1, n + 1)
    S = jnp.cumsum(x) - k * jnp.mean(x)

    S_std = S / std
    R = (S_std.max() - S_std.min()) / jnp.sqrt(n)

    return R


def _buishand_range_statistic(x: NDArray[np.floating], alpha: float = 0.05) -> tuple[float, int]:
    """Calculate Buishand's range statistic for change point detection.

    Args:
        x: Input data array
        alpha: Significance level (not used in calculation)

    Returns:
        Tuple of (R statistic normalized by sqrt(n), change point location)

    Raises:
        ValueError: If data has less than 2 samples or zero variance
    """
    n = len(x)
    if n < 2:
        raise ValueError(f"Buishand range test requires at least 2 samples, got {n}")

    std = x.std()
    validate_variance(std, "Buishand range test")

    k = np.arange(1, n + 1)
    S = x.cumsum() - k * x.mean()

    S_std = S / std  # should use sample std -> x.std()
    R = (S_std.max() - S_std.min()) / np.sqrt(n)

    return float(R), int(abs(S).argmax() + 1)


def buishand_range_test(x: Any, alpha: float = 0.05, sim: int | None = 20000) -> Any:
    """
    This function checks homogeneity test using Buishand's range method proposed in T. A. Buishand (1982).

    Args:
        x: a vector (list, numpy array or pandas series) data
        alpha: significance level (default 0.05)
        sim: No. of monte carlo simulation for p-value calculation (default 20000)

    Returns:
        Named tuple with fields:
            h: True (if data is nonhomogeneous) or False (if data is homogeneous)
            cp: probable change-point location index
            p: p-value of the significance test
            R: Buishand's Q Statistics range divided by square root of sample size [R/sqrt(n)]
            avg: mean values at before and after change-point

    Examples:
        >>> import pyhomogeneity as hg
        >>> x = np.random.rand(1000)
        >>> h, cp, p, R, mu = hg.buishand_range_test(x, 0.05)

    Raises:
        ValueError: If input data is invalid or has insufficient samples
    """
    validate_alpha(alpha)

    x_array, c, idx = preprocess(x)
    x_clean, n, idx = remove_missing_values(x_array, idx, method="skip")

    stat, loc = _buishand_range_statistic(x_clean)

    if sim:
        p = monte_carlo_p_value(_buishand_range_statistic, stat, n, sim)
        h = alpha > p
    else:
        p = None
        h = None

    mu = calculate_means(x_clean, loc)

    result = namedtuple("Buishand_Range_Test", ["h", "cp", "p", "R", "avg"])
    return result(h, idx[loc - 1], p, stat, mu)
