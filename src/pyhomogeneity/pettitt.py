"""
Pettitt's test for change point detection.

Reference: Pettitt, A. N. (1979). A non-parametric approach to the change-point problem.
Applied Statistics, 28(2), 126-135.
"""

from __future__ import annotations

from collections import namedtuple
from typing import Any

import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray
from scipy.stats import rankdata

from ._utils import calculate_means, monte_carlo_p_value, preprocess, remove_missing_values, validate_alpha


def _rankdata_jax(x):
    """JAX-compatible ranking function (simple, no tie handling).

    Note: For Monte Carlo simulations with continuous random data,
    ties are extremely rare, so this simplified version is acceptable.
    """
    sorted_indices = jnp.argsort(x)
    ranks = jnp.empty_like(sorted_indices, dtype=jnp.float32)
    ranks = ranks.at[sorted_indices].set(jnp.arange(1, len(x) + 1, dtype=jnp.float32))
    return ranks


def _pettitt_statistic_jax(x):
    """JAX-native Pettitt statistic calculation for Monte Carlo simulations.

    Args:
        x: Input data array (JAX array)

    Returns:
        Max absolute U statistic (JAX scalar)
    """
    n = len(x)
    r = _rankdata_jax(x)

    k = jnp.arange(n - 1)
    s = jnp.cumsum(r)[:-1]

    U = 2 * s - (k + 1) * (n + 1)

    return jnp.max(jnp.abs(U))


def _pettitt_statistic(x: NDArray[np.floating]) -> tuple[float, int]:
    """Calculate Pettitt's U statistic for change point detection.

    Args:
        x: Input data array

    Returns:
        Tuple of (max absolute U statistic, change point location)

    Raises:
        ValueError: If data has less than 2 samples
    """
    n = len(x)
    if n < 2:
        raise ValueError(f"Pettitt test requires at least 2 samples, got {n}")

    # Use scipy's rankdata for proper tie handling
    r = rankdata(x)

    k = np.arange(n - 1)
    s = r.cumsum()[:-1]

    U = 2 * s - (k + 1) * (n + 1)

    return float(abs(U).max()), int(abs(U).argmax() + 1)


def pettitt_test(x: Any, alpha: float = 0.05, sim: int = 20000) -> Any:
    """
    This function checks homogeneity test using A. N. Pettitt's (1979) method.

    Args:
        x: a vector (list, numpy array or pandas series) data
        alpha: significance level (default 0.05)
        sim: No. of monte carlo simulation for p-value calculation (default 20000)

    Returns:
        Named tuple with fields:
            h: True (if data is nonhomogeneous) or False (if data is homogeneous)
            cp: probable change-point location index
            p: p-value of the significance test
            U: Maximum of absolute Pettitt's U Statistics
            avg: mean values at before and after change-point

    Examples:
        >>> import pyhomogeneity as hg
        >>> x = np.random.rand(1000)
        >>> h, cp, p, U, mu = hg.pettitt_test(x, 0.05)

    Raises:
        ValueError: If input data is invalid or has insufficient samples
    """
    validate_alpha(alpha)

    x_array, c, idx = preprocess(x)
    x_clean, n, idx = remove_missing_values(x_array, idx, method="skip")

    stat, loc = _pettitt_statistic(x_clean)

    if sim:
        p = monte_carlo_p_value(_pettitt_statistic, stat, n, sim)
        h = alpha > p
    else:
        # Use analytical formula when sim is None/0
        p = 2 * np.exp((-6 * stat**2) / (n**3 + n**2))
        h = alpha > p

    mu = calculate_means(x_clean, loc)

    result = namedtuple("Pettitt_Test", ["h", "cp", "p", "U", "avg"])
    return result(h, idx[loc - 1], p, stat, mu)
