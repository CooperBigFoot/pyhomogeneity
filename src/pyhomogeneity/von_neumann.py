"""
Von Neumann Ratio Test for homogeneity.

Reference: von Neumann, J., Kent, R. H., Bellinson, H. R., & Hart, B. I. (1941).
The mean square successive difference. Annals of Mathematical Statistics, 12(2), 153-162.
"""

from __future__ import annotations

from collections import namedtuple
from typing import Any

import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from ._utils import (
    monte_carlo_p_value,
    preprocess,
    remove_missing_values,
    validate_alpha,
    validate_variance,
)


def _von_neumann_statistic_jax(x):
    """JAX-native von Neumann statistic calculation for Monte Carlo simulations.

    The von Neumann ratio tests for randomness by comparing the mean square
    successive difference to the variance.

    Args:
        x: Input data array (JAX array)

    Returns:
        Von Neumann ratio statistic (JAX scalar)
    """
    n = len(x)
    x_mean = jnp.mean(x)

    # Mean square successive difference
    delta = x[1:] - x[:-1]
    mssd = jnp.sum(delta**2) / (n - 1)

    # Variance
    variance = jnp.sum((x - x_mean) ** 2) / n

    # Von Neumann ratio
    vn_ratio = mssd / variance

    return vn_ratio


def _von_neumann_statistic(x: NDArray[np.floating]) -> tuple[float, int]:
    """Calculate von Neumann ratio statistic with change point detection.

    The classic von Neumann ratio is a global statistic. For change point detection,
    we scan through possible split points and find where the combined von Neumann
    ratios of the two segments deviate most from the expected value of 2.0.

    Args:
        x: Input data array

    Returns:
        Tuple of (von Neumann ratio at change point, change point location)

    Raises:
        ValueError: If data has less than 2 samples or zero variance
    """
    n = len(x)
    if n < 2:
        raise ValueError(f"Von Neumann test requires at least 2 samples, got {n}")

    # Check for zero variance
    std = x.std(ddof=1)
    validate_variance(std, "Von Neumann test")

    # For change point detection, scan through possible positions
    # and find where the von Neumann ratio changes most significantly
    if n < 4:
        # For very small datasets, just return the global statistic
        x_mean = x.mean()
        delta = x[1:] - x[:-1]
        mssd = np.sum(delta**2) / (n - 1)
        variance = np.sum((x - x_mean) ** 2) / n
        vn_ratio = mssd / variance
        return float(vn_ratio), 1

    # Scan through positions to find change point
    # We compute the local von Neumann ratio around each point
    max_deviation = 0.0
    change_point = 1

    # Expected value of von Neumann ratio under null hypothesis is 2.0
    expected_vn = 2.0

    for k in range(2, n - 1):  # Need at least 2 points on each side
        # Compute von Neumann ratio for segments before and after k
        x1 = x[:k]
        x2 = x[k:]

        if len(x1) >= 2 and len(x2) >= 2:
            # Compute VN ratio for first segment
            mean1 = x1.mean()
            delta1 = x1[1:] - x1[:-1]
            mssd1 = np.sum(delta1**2) / (len(x1) - 1)
            var1 = np.sum((x1 - mean1) ** 2) / len(x1)
            vn1 = mssd1 / var1 if var1 > 0 else expected_vn

            # Compute VN ratio for second segment
            mean2 = x2.mean()
            delta2 = x2[1:] - x2[:-1]
            mssd2 = np.sum(delta2**2) / (len(x2) - 1)
            var2 = np.sum((x2 - mean2) ** 2) / len(x2)
            vn2 = mssd2 / var2 if var2 > 0 else expected_vn

            # Deviation from expected: larger deviations indicate potential change points
            deviation = abs(vn1 - expected_vn) + abs(vn2 - expected_vn)

            if deviation > max_deviation:
                max_deviation = deviation
                change_point = k

    # Compute the overall von Neumann ratio at the detected change point
    x_mean = x.mean()
    delta = x[1:] - x[:-1]
    mssd = np.sum(delta**2) / (n - 1)
    variance = np.sum((x - x_mean) ** 2) / n
    vn_ratio = mssd / variance

    return float(vn_ratio), change_point


def von_neumann_test(x: Any, alpha: float = 0.05, sim: int | None = 20000) -> Any:
    """
    Test for homogeneity using the von Neumann ratio test.

    The von Neumann ratio compares the mean square successive difference to the
    variance. Under the null hypothesis of randomness/homogeneity, the expected
    ratio is 2.0. Values significantly different from 2.0 suggest non-homogeneity.

    Args:
        x: a vector (list, numpy array or pandas series) data
        alpha: significance level (default 0.05)
        sim: No. of monte carlo simulation for p-value calculation (default 20000)
             Set to None or 0 to skip Monte Carlo simulation

    Returns:
        Named tuple with fields:
            h: True (if data is nonhomogeneous) or False (if data is homogeneous)
               None if sim=None (no hypothesis test performed)
            cp: probable change-point location index
            p: p-value of the significance test (None if sim=None)
            VN: von Neumann ratio statistic
            avg: None (von Neumann is a global test without clear segmentation)

    Examples:
        >>> import pyhomogeneity as hg
        >>> import numpy as np
        >>> x = np.random.rand(100)
        >>> result = hg.von_neumann_test(x, 0.05)
        >>> print(result.h, result.cp, result.p)

    Raises:
        ValueError: If input data is invalid or has insufficient samples

    Reference:
        von Neumann, J., Kent, R. H., Bellinson, H. R., & Hart, B. I. (1941).
        The mean square successive difference. Annals of Mathematical Statistics,
        12(2), 153-162.
    """
    validate_alpha(alpha)

    x_array, c, idx = preprocess(x)
    x_clean, n, idx = remove_missing_values(x_array, idx, method="skip")

    stat, loc = _von_neumann_statistic(x_clean)

    if sim:
        p = monte_carlo_p_value(_von_neumann_statistic, stat, n, sim)
        # Von Neumann ratio: reject if significantly different from 2.0
        # We use a two-tailed test
        h = alpha > p
    else:
        p = None
        h = None

    # Von Neumann is a global test, so avg is None
    avg = None

    result = namedtuple("VonNeumann_Test", ["h", "cp", "p", "VN", "avg"])
    return result(h, idx[loc - 1], p, stat, avg)
