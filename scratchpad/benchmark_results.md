# pyHomogeneity Benchmark Results

**Generated:** 2025-11-14 15:47:44

## System Information

- **Python Version:** 3.12.11
- **Platform:** macOS-15.6-arm64-arm-64bit
- **Processor:** arm
- **Machine:** arm64

## Benchmark Configuration

- **Data Sizes:** 100, 1,000, 10,000 points
- **Iterations:** 10 runs per configuration
- **MC Simulations:** 20,000 (when enabled)
- **Alpha Level:** 0.05

## Results: With Monte Carlo Simulations

| Function | 100 pts | 1,000 pts | 10,000 pts |
|----------|---------|-----------|------------|
| Pettitt | 234.30 ms | 2.320 s | 27.404 s |
| SNHT | 38.23 ms | 191.75 ms | 1.300 s |
| Buishand Q | 20.99 ms | 102.43 ms | 710.74 ms |
| Buishand Range | 12.77 ms | 99.86 ms | 713.84 ms |
| Buishand LR | 17.55 ms | 108.21 ms | 818.84 ms |
| Buishand U | 13.76 ms | 107.61 ms | 749.97 ms |

## Results: Without Monte Carlo Simulations

| Function | 100 pts | 1,000 pts | 10,000 pts |
|----------|---------|-----------|------------|
| Pettitt | 142.1 µs | 265.5 µs | 1.25 ms |
| SNHT | 120.0 µs | 210.7 µs | 707.3 µs |
| Buishand Q | 98.6 µs | 161.1 µs | 742.5 µs |
| Buishand Range | 84.4 µs | 149.1 µs | 605.0 µs |
| Buishand LR | 105.9 µs | 132.3 µs | 552.7 µs |
| Buishand U | 94.2 µs | 106.4 µs | 522.9 µs |

## Key Observations

### Performance Rankings (10,000 points with MC)

- **Fastest:** Buishand Q (710.74 ms)
- **Slowest:** Pettitt (27.404 s)
- **Ratio:** 38.56x

### Monte Carlo Simulation Impact

Average speedup when disabling MC simulations (10,000 points):

- **Pettitt:** 21871.9x faster
- **SNHT:** 1838.1x faster
- **Buishand Q:** 957.3x faster
- **Buishand Range:** 1180.0x faster
- **Buishand LR:** 1481.5x faster
- **Buishand U:** 1434.3x faster

### Scaling Analysis

Time increase from 100 to 10,000 points (with MC):

- **Pettitt:** 117.0x
- **SNHT:** 34.0x
- **Buishand Q:** 33.9x
- **Buishand Range:** 55.9x
- **Buishand LR:** 46.6x
- **Buishand U:** 54.5x

---

*Benchmark completed successfully. Use these baseline measurements to compare against post-refactoring performance.*
