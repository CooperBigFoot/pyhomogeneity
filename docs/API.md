# API Documentation

## Overview

This library provides seven statistical tests for detecting change points in time series data:

1. Pettitt's test
2. Standard Normal Homogeneity Test (SNHT)
3. Buishand's Q test
4. Buishand's Range test
5. Buishand's Likelihood Ratio test
6. Buishand's U test
7. von Neumann Ratio test

All functions follow the same interface pattern.

---

## Common Interface

All test functions accept the following parameters:

### Parameters

- **x** : `list`, `numpy.ndarray`, or `pandas.Series`
  - Input time series data
  - Missing values (NaN) are automatically removed

- **alpha** : `float`, default=0.05
  - Significance level for hypothesis testing
  - Must be between 0 and 1

- **sim** : `int` or `None`, default=20000
  - Number of Monte Carlo simulations for p-value calculation
  - Set to `None` or `0` to skip Monte Carlo simulation
  - When disabled, some tests will return `None` for `h` and `p` values

### Return Value

All functions return a named tuple with the following fields:

- **h** : `bool` or `None`
  - Hypothesis test result
  - `True`: Data is non-homogeneous (change point detected)
  - `False`: Data is homogeneous (no change point)
  - `None`: When `sim=None` (no hypothesis test performed)

- **cp** : `int`
  - Index of probable change point location
  - Refers to position in original data (before NaN removal)

- **p** : `float` or `None`
  - P-value from Monte Carlo simulation
  - `None`: When `sim=None`

- **statistic** : `float`
  - Test statistic value
  - Field name varies by test: `U`, `T`, `Q`, `R`, or `V`

- **avg** : `namedtuple(mu1, mu2)`
  - Mean values before (`mu1`) and after (`mu2`) the change point

---

## Functions

### pettitt_test

```python
from pyhomogeneity import pettitt_test

result = pettitt_test(x, alpha=0.05, sim=20000)
```

**Return fields:**
- `h`, `cp`, `p`, `U`, `avg`

**Reference:** Pettitt, A. N. (1979). A non-parametric approach to the change-point problem. Applied Statistics, 28(2), 126-135.

**Notes:**
- Non-parametric test based on Mann-Whitney statistic
- Uses ranking of data values
- Slower than other tests when `sim > 0`

---

### snht_test

```python
from pyhomogeneity import snht_test

result = snht_test(x, alpha=0.05, sim=20000)
```

**Return fields:**
- `h`, `cp`, `p`, `T`, `avg`

**Reference:** Alexandersson, H. (1986). A homogeneity test applied to precipitation data. Journal of Climatology, 6(6), 661-675.

**Notes:**
- Assumes normal distribution
- Requires non-constant data (will raise `ValueError` if std=0)
- Analytical formula not available; requires `sim > 0` for p-value

---

### buishand_q_test

```python
from pyhomogeneity import buishand_q_test

result = buishand_q_test(x, alpha=0.05, sim=20000)
```

**Return fields:**
- `h`, `cp`, `p`, `Q`, `avg`

**Reference:** Buishand, T. A. (1982). Some methods for testing the homogeneity of rainfall records. Journal of Hydrology, 58(1-2), 11-27.

**Notes:**
- Based on cumulative deviations from mean
- Statistic normalized by `sqrt(n)`
- Requires non-constant data

---

### buishand_range_test

```python
from pyhomogeneity import buishand_range_test

result = buishand_range_test(x, alpha=0.05, sim=20000)
```

**Return fields:**
- `h`, `cp`, `p`, `R`, `avg`

**Reference:** Buishand, T. A. (1982). Some methods for testing the homogeneity of rainfall records. Journal of Hydrology, 58(1-2), 11-27.

**Notes:**
- Based on range of cumulative deviations
- Statistic normalized by `sqrt(n)`
- Requires non-constant data

---

### buishand_likelihood_ratio_test

```python
from pyhomogeneity import buishand_likelihood_ratio_test

result = buishand_likelihood_ratio_test(x, alpha=0.05, sim=20000)
```

**Return fields:**
- `h`, `cp`, `p`, `V`, `avg`

**Reference:** Buishand, T. A. (1984). Tests for detecting a shift in the mean of hydrological time series. Journal of Hydrology, 73(1-2), 51-69.

**Notes:**
- Likelihood ratio approach
- Weighted by variance at each potential change point
- Requires non-constant data

---

### buishand_u_test

```python
from pyhomogeneity import buishand_u_test

result = buishand_u_test(x, alpha=0.05, sim=20000)
```

**Return fields:**
- `h`, `cp`, `p`, `U`, `avg`

**Reference:** Buishand, T. A. (1984). Tests for detecting a shift in the mean of hydrological time series. Journal of Hydrology, 73(1-2), 51-69.

**Notes:**
- Based on squared cumulative deviations
- Normalized by sample size
- Requires non-constant data

---

### von_neumann_test

```python
from pyhomogeneity import von_neumann_test

result = von_neumann_test(x, alpha=0.05, sim=20000)
```

**Return fields:**
- `h`, `cp`, `p`, `VN`, `avg` (note: `avg` is always `None`)

**Reference:** von Neumann, J., Kent, R. H., Bellinson, H. R., & Hart, B. I. (1941). The mean square successive difference. Annals of Mathematical Statistics, 12(2), 153-162.

**Notes:**
- Tests for randomness using the von Neumann ratio
- Compares mean square successive difference to variance
- Expected ratio under null hypothesis is 2.0
- Change point detection via scanning local deviations
- `avg` field returns `None` (global test without clear segmentation)
- Requires non-constant data

---

## Usage Examples

### Basic Usage

```python
import numpy as np
from pyhomogeneity import pettitt_test

# Generate sample data
data = np.concatenate([
    np.random.normal(50, 5, 100),  # First segment
    np.random.normal(100, 5, 100)  # Second segment with shift
])

# Run test
result = pettitt_test(data, alpha=0.05)

print(f"Change point detected: {result.h}")
print(f"Change point location: {result.cp}")
print(f"P-value: {result.p}")
print(f"Mean before: {result.avg.mu1}")
print(f"Mean after: {result.avg.mu2}")
```

### Without Monte Carlo Simulation

```python
# Fast computation without p-value
result = snht_test(data, alpha=0.05, sim=None)

print(f"Test statistic: {result.T}")
print(f"Change point: {result.cp}")
# result.h and result.p will be None
```

### With Pandas Series

```python
import pandas as pd

# Create time series
dates = pd.date_range('2000-01-01', periods=200, freq='M')
ts = pd.Series(data, index=dates)

# Run test
result = buishand_range_test(ts)

# Change point returned as date string
print(f"Change detected at: {result.cp}")
```

### Handling Missing Data

```python
# Data with missing values
data_with_nan = np.array([1, 2, np.nan, 4, 5, np.nan, 7, 8])

# NaN values automatically removed
result = snht_test(data_with_nan)
# Processes: [1, 2, 4, 5, 7, 8]
```

---

## Error Handling

All functions raise `ValueError` in the following cases:

1. **Insufficient data**: Less than 2 samples after removing NaN values
2. **Invalid alpha**: Not in range (0, 1)
3. **Invalid sim**: Negative value
4. **Zero variance**: Constant data (for SNHT and Buishand tests)
5. **All NaN**: No valid data after removing missing values

### Example

```python
try:
    result = snht_test([5, 5, 5, 5], alpha=0.05)
except ValueError as e:
    print(e)  # "SNHT test requires non-constant data (standard deviation is zero)"
```

---

## Performance Considerations

### Monte Carlo Simulation

The computational cost is dominated by Monte Carlo simulations when `sim > 0`:

- **Default (sim=20000)**: Accurate p-values, slower execution
- **Reduced (sim=1000)**: Faster, less accurate p-values
- **Disabled (sim=None)**: Very fast, no p-value

### Test Performance (10,000 data points, sim=20000)

| Test | Approximate Time |
|------|------------------|
| Buishand Q | ~0.7 seconds |
| Buishand Range | ~0.7 seconds |
| Buishand U | ~0.8 seconds |
| Buishand LR | ~0.8 seconds |
| von Neumann | ~0.8 seconds |
| SNHT | ~1.3 seconds |
| Pettitt | ~17 seconds* |

*Pettitt test is slower due to ranking operations

### Recommendations

- For exploratory analysis: Use `sim=1000` or `sim=None`
- For publication: Use `sim=20000` (default)
- For large datasets (n > 10,000): Consider reducing `sim`
- Buishand tests are fastest for routine analysis

---

## Implementation Details

### Backend

- Core statistics: NumPy
- Ranking (Pettitt): scipy.stats.rankdata
- Monte Carlo acceleration: JAX (optional, automatic)

### JAX Acceleration

Monte Carlo simulations are automatically accelerated using JAX when available. This provides ~5x speedup for SNHT and Buishand tests with no API changes.

JAX acceleration is **not used** for Pettitt test due to ranking operations.

---

## Type Signatures

```python
def pettitt_test(
    x: list | np.ndarray | pd.Series,
    alpha: float = 0.05,
    sim: int = 20000
) -> namedtuple  # Fields: h, cp, p, U, avg

def snht_test(
    x: list | np.ndarray | pd.Series,
    alpha: float = 0.05,
    sim: int | None = 20000
) -> namedtuple  # Fields: h, cp, p, T, avg

def buishand_q_test(
    x: list | np.ndarray | pd.Series,
    alpha: float = 0.05,
    sim: int | None = 20000
) -> namedtuple  # Fields: h, cp, p, Q, avg

def buishand_range_test(
    x: list | np.ndarray | pd.Series,
    alpha: float = 0.05,
    sim: int | None = 20000
) -> namedtuple  # Fields: h, cp, p, R, avg

def buishand_likelihood_ratio_test(
    x: list | np.ndarray | pd.Series,
    alpha: float = 0.05,
    sim: int | None = 20000
) -> namedtuple  # Fields: h, cp, p, V, avg

def buishand_u_test(
    x: list | np.ndarray | pd.Series,
    alpha: float = 0.05,
    sim: int | None = 20000
) -> namedtuple  # Fields: h, cp, p, U, avg

def von_neumann_test(
    x: list | np.ndarray | pd.Series,
    alpha: float = 0.05,
    sim: int | None = 20000
) -> namedtuple  # Fields: h, cp, p, VN, avg (avg is always None)
```
