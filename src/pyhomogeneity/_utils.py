"""
Shared utility functions for homogeneity tests.

This module contains common preprocessing, validation, and statistical computation
functions used across all homogeneity test methods.
"""

from __future__ import annotations

from collections import namedtuple
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray


def preprocess(x: Any) -> tuple[NDArray[np.floating], int, NDArray[Any]]:
    """Preprocess input data and extract indices.

    Args:
        x: Input data (list, numpy array, or pandas series)

    Returns:
        Tuple of (data array, number of columns, index array)

    Raises:
        ValueError: If input has more than 2 dimensions
    """
    try:
        idx = x.index.date.astype("str") if x.index.dtype != "int64" else np.asarray(range(1, len(x) + 1))
    except AttributeError:
        idx = np.asarray(range(1, len(x) + 1))

    x = np.asarray(x, dtype=float)
    dim = x.ndim

    if dim == 1:
        c = 1

    elif dim == 2:
        (n, c) = x.shape

        if c == 1:
            dim = 1
            x = x.flatten()

    else:
        raise ValueError(f"Input data must be 1D or 2D array, got {dim}D array. Please check your dataset.")

    return x, c, idx


def remove_missing_values(
    x: NDArray[np.floating], idx: NDArray[Any], method: str = "skip"
) -> tuple[NDArray[np.floating], int, NDArray[Any]]:
    """Remove missing values from data.

    Args:
        x: Input data array
        idx: Index array
        method: Method for handling missing values (default: 'skip')

    Returns:
        Tuple of (cleaned data array, number of samples, cleaned index array)

    Raises:
        ValueError: If data is empty after removing missing values
    """
    if method.lower() == "skip":
        if x.ndim == 1:
            mask = ~np.isnan(x)
            idx = idx[mask]
            x = x[mask]

        else:
            mask = ~np.isnan(x).any(axis=1)
            idx = idx[mask]
            x = x[mask]

    n = len(x)

    if n == 0:
        raise ValueError("No valid data remaining after removing missing values")

    return x, n, idx


def calculate_means(x: NDArray[np.floating], loc: int) -> Any:
    """Calculate mean before and after change point.

    Args:
        x: Input data array
        loc: Change point location (1-indexed)

    Returns:
        Named tuple with mu1 (mean before) and mu2 (mean after)

    Raises:
        ValueError: If loc is out of bounds
    """
    if loc < 1 or loc >= len(x):
        raise ValueError(f"Change point location must be between 1 and {len(x) - 1}, got {loc}")

    mu = namedtuple("mean", ["mu1", "mu2"])
    mu1 = float(x[:loc].mean())
    mu2 = float(x[loc:].mean())

    return mu(mu1, mu2)


def monte_carlo_p_value(
    func: Callable[[NDArray[np.floating]], tuple[float, int]], stat: float, n: int, sim: int
) -> float:
    """Calculate p-value using Monte Carlo simulation with JAX acceleration.

    Args:
        func: Statistical test function that returns (statistic, location)
        stat: Observed test statistic
        n: Sample size
        sim: Number of Monte Carlo simulations

    Returns:
        Estimated p-value

    Raises:
        ValueError: If sim is not positive or n is less than 2
    """
    if sim <= 0:
        raise ValueError(f"Number of simulations must be positive, got {sim}")
    if n < 2:
        raise ValueError(f"Sample size must be at least 2, got {n}")

    # Generate random data using JAX for better performance
    key = jax.random.PRNGKey(0)
    rand_data = jax.random.normal(key, shape=(sim, n))

    # Check if the function has a JAX-optimized version
    func_name = func.__name__
    jax_func_name = f"{func_name}_jax"

    # Try to get the JAX version from the same module
    import sys

    func_module = sys.modules[func.__module__]
    jax_func = getattr(func_module, jax_func_name, None)

    if jax_func is not None:
        # Use the JAX-optimized path with vmap
        batched_stat = jax.vmap(jax_func)
        stats = batched_stat(rand_data)
    else:
        # Fallback to numpy conversion (slower but compatible)
        def stat_only(x):
            x_np = np.asarray(x)
            stat_val, _ = func(x_np)
            return stat_val

        batched_stat = jax.vmap(stat_only)
        stats = batched_stat(rand_data)

    # Calculate p-value
    p_val = float(jnp.sum(stats > stat) / sim)

    return p_val


def validate_alpha(alpha: float) -> None:
    """Validate significance level alpha.

    Args:
        alpha: Significance level

    Raises:
        ValueError: If alpha is not in (0, 1)
    """
    if not 0 < alpha < 1:
        raise ValueError(f"Significance level alpha must be between 0 and 1, got {alpha}")


def validate_sample_size(n: int, min_samples: int = 2, test_name: str = "Test") -> None:
    """Validate sample size.

    Args:
        n: Sample size
        min_samples: Minimum required samples
        test_name: Name of the test for error message

    Raises:
        ValueError: If sample size is insufficient
    """
    if n < min_samples:
        raise ValueError(f"{test_name} requires at least {min_samples} samples, got {n}")


def validate_variance(std: float, test_name: str = "Test") -> None:
    """Validate that data has non-zero variance.

    Args:
        std: Standard deviation
        test_name: Name of the test for error message

    Raises:
        ValueError: If standard deviation is zero
    """
    if std == 0:
        raise ValueError(f"{test_name} requires non-constant data (standard deviation is zero)")
