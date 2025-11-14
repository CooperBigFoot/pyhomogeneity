"""
Buishand's likelihood ratio test for change point detection.

Reference: Buishand, T. A. (1984). Tests for detecting a shift in the mean of hydrological time series.
Journal of Hydrology, 73(1-2), 51-69.
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


def _buishand_lr_statistic_jax(x, alpha: float = 0.05):
    """JAX-native Buishand likelihood ratio statistic calculation for Monte Carlo simulations.

    Args:
        x: Input data array (JAX array)
        alpha: Significance level (not used in calculation)

    Returns:
        V statistic (JAX scalar)
    """
    n = len(x)
    std = jnp.std(x)

    k = jnp.arange(1, n + 1)
    S = jnp.cumsum(x) - k * jnp.mean(x)

    V = S[:-1] / (std * jnp.sqrt(k[:-1] * (n - k[:-1])))

    return jnp.max(jnp.abs(V))


def _buishand_lr_statistic(x: NDArray[np.floating], alpha: float = 0.05) -> tuple[float, int]:
    """Calculate Buishand's likelihood ratio statistic for change point detection.

    Args:
        x: Input data array
        alpha: Significance level (not used in calculation)

    Returns:
        Tuple of (V statistic, change point location)

    Raises:
        ValueError: If data has less than 2 samples or zero variance
    """
    n = len(x)
    if n < 2:
        raise ValueError(f"Buishand likelihood ratio test requires at least 2 samples, got {n}")

    std = x.std()
    validate_variance(std, "Buishand likelihood ratio test")

    k = np.arange(1, n + 1)
    S = x.cumsum() - k * x.mean()

    V = S[:-1] / (std * (k[:-1] * (n - k[:-1])) ** 0.5)

    return float(abs(V).max()), int(abs(S).argmax() + 1)


def buishand_likelihood_ratio_test(x: Any, alpha: float = 0.05, sim: int | None = 20000) -> Any:
    """
    This function checks homogeneity test using Buishand's likelihood ratio method proposed in T. A. Buishand (1984).

    Args:
        x: a vector (list, numpy array or pandas series) data
        alpha: significance level (default 0.05)
        sim: No. of monte carlo simulation for p-value calculation (default 20000)

    Returns:
        Named tuple with fields:
            h: True (if data is nonhomogeneous) or False (if data is homogeneous)
            cp: probable change-point location index
            p: p-value of the significance test
            V: Maximum of absolute Buishand's weighted adjusted partial sum S
            avg: mean values at before and after change-point

    Examples:
        >>> import pyhomogeneity as hg
        >>> x = np.random.rand(1000)
        >>> h, cp, p, V, mu = hg.buishand_likelihood_ratio_test(x, 0.05)

    Raises:
        ValueError: If input data is invalid or has insufficient samples
    """
    validate_alpha(alpha)

    x_array, c, idx = preprocess(x)
    x_clean, n, idx = remove_missing_values(x_array, idx, method="skip")

    stat, loc = _buishand_lr_statistic(x_clean)

    if sim:
        p = monte_carlo_p_value(_buishand_lr_statistic, stat, n, sim)
        h = alpha > p
    else:
        p = None
        h = None

    mu = calculate_means(x_clean, loc)

    result = namedtuple("Buishand_Likelihood_Ratio_Test", ["h", "cp", "p", "V", "avg"])
    return result(h, idx[loc - 1], p, stat, mu)
