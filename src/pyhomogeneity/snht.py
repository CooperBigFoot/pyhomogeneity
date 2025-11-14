"""
Standard Normal Homogeneity Test (SNHT).

Reference: Alexandersson, H. (1986). A homogeneity test applied to precipitation data.
Journal of Climatology, 6(6), 661-675.
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


def _snht_statistic_jax(x):
    """JAX-native SNHT statistic calculation for Monte Carlo simulations.

    Args:
        x: Input data array (JAX array)

    Returns:
        Max T statistic (JAX scalar)
    """
    n = len(x)
    std = jnp.std(x, ddof=1)

    k = jnp.arange(1, n)
    s = jnp.cumsum(x)[:-1]
    rs = jnp.cumsum(x[::-1])[::-1][1:]

    x_mean = jnp.mean(x)
    z1 = ((s - k * x_mean) / std) / k
    z2 = ((rs - k[::-1] * x_mean) / std) / (n - k)
    T = k * z1**2 + (n - k) * z2**2

    return jnp.max(T)


def _snht_statistic(x: NDArray[np.floating]) -> tuple[float, int]:
    """Calculate Standard Normal Homogeneity Test (SNHT) statistic.

    Args:
        x: Input data array

    Returns:
        Tuple of (max T statistic, change point location)

    Raises:
        ValueError: If data has less than 2 samples or zero variance
    """
    n = len(x)
    if n < 2:
        raise ValueError(f"SNHT test requires at least 2 samples, got {n}")

    std = x.std(ddof=1)
    validate_variance(std, "SNHT test")

    k = np.arange(1, n)
    s = x.cumsum()[:-1]
    rs = x[::-1].cumsum()[::-1][1:]

    z1 = ((s - k * x.mean()) / std) / k
    z2 = ((rs - k[::-1] * x.mean()) / std) / (n - k)
    T = k * z1**2 + (n - k) * z2**2

    return float(T.max()), int(T.argmax() + 1)


def snht_test(x: Any, alpha: float = 0.05, sim: int | None = 20000) -> Any:
    """
    This function checks homogeneity test using H. Alexandersson (1986) method.

    Args:
        x: a vector (list, numpy array or pandas series) data
        alpha: significance level (default 0.05)
        sim: No. of monte carlo simulation for p-value calculation (default 20000)

    Returns:
        Named tuple with fields:
            h: True (if data is nonhomogeneous) or False (if data is homogeneous)
            cp: probable change-point location index
            p: p-value of the significance test
            T: Maximum of SNHT T Statistics
            avg: mean values at before and after change-point

    Examples:
        >>> import pyhomogeneity as hg
        >>> x = np.random.rand(1000)
        >>> h, cp, p, T, mu = hg.snht_test(x, 0.05)

    Raises:
        ValueError: If input data is invalid or has insufficient samples
    """
    validate_alpha(alpha)

    x_array, c, idx = preprocess(x)
    x_clean, n, idx = remove_missing_values(x_array, idx, method="skip")

    stat, loc = _snht_statistic(x_clean)

    if sim:
        p = monte_carlo_p_value(_snht_statistic, stat, n, sim)
        h = alpha > p
    else:
        p = None
        h = None

    mu = calculate_means(x_clean, loc)

    result = namedtuple("SNHT_Test", ["h", "cp", "p", "T", "avg"])
    return result(h, idx[loc - 1], p, stat, mu)
