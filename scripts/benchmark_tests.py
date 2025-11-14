#!/usr/bin/env python3
"""Benchmark script for pyHomogeneity statistical tests.

This script benchmarks all 6 statistical test functions across different
data sizes and configurations to establish a performance baseline before refactoring.
"""

import logging
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# Add parent directory to path to import pyhomogeneity
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pyhomogeneity.buishand_lr import buishand_likelihood_ratio_test
from pyhomogeneity.buishand_q import buishand_q_test
from pyhomogeneity.buishand_range import buishand_range_test
from pyhomogeneity.buishand_u import buishand_u_test
from pyhomogeneity.pettitt import pettitt_test
from pyhomogeneity.snht import snht_test

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def generate_test_data(size: int, with_changepoint: bool = True) -> np.ndarray:
    """Generate synthetic test data with or without a change point.

    Args:
        size: Number of data points
        with_changepoint: If True, create data with changepoint at midpoint

    Returns:
        NumPy array of test data
    """
    np.random.seed(42)  # For reproducibility

    if with_changepoint:
        # First half: mean=50, std=5
        # Second half: mean=100, std=5
        half = size // 2
        first_half = np.random.normal(50, 5, half)
        second_half = np.random.normal(100, 5, size - half)
        return np.concatenate([first_half, second_half])
    else:
        # Homogeneous data: mean=100, std=10
        return np.random.normal(100, 10, size)


def benchmark_function(
    func: callable,
    data: np.ndarray,
    iterations: int = 10,
    with_sim: bool = True,
) -> dict[str, float]:
    """Benchmark a single function with given data.

    Args:
        func: Test function to benchmark
        data: Input data array
        iterations: Number of iterations to average over
        with_sim: Whether to use Monte Carlo simulations

    Returns:
        Dictionary with timing statistics
    """
    sim_value = 20000 if with_sim else None
    times = []

    for _ in range(iterations):
        start = time.perf_counter()
        _ = func(data, alpha=0.05, sim=sim_value)
        end = time.perf_counter()
        times.append(end - start)

    return {
        "mean": np.mean(times),
        "std": np.std(times),
        "min": np.min(times),
        "max": np.max(times),
    }


def run_benchmarks() -> dict[str, dict]:
    """Run comprehensive benchmarks across all functions and configurations.

    Returns:
        Nested dictionary of benchmark results
    """
    # Test configurations
    test_functions = {
        "Pettitt": pettitt_test,
        "SNHT": snht_test,
        "Buishand Q": buishand_q_test,
        "Buishand Range": buishand_range_test,
        "Buishand LR": buishand_likelihood_ratio_test,
        "Buishand U": buishand_u_test,
    }

    data_sizes = [100, 1000, 10000]
    iterations = 10

    results = {}
    total_tests = len(test_functions) * len(data_sizes) * 2  # *2 for with/without sim
    current_test = 0

    logger.info("=" * 80)
    logger.info("Starting pyHomogeneity Benchmark Suite")
    logger.info("=" * 80)
    logger.info(f"Functions to test: {len(test_functions)}")
    logger.info(f"Data sizes: {data_sizes}")
    logger.info(f"Iterations per test: {iterations}")
    logger.info(f"Total benchmark runs: {total_tests}")
    logger.info("=" * 80)

    for func_name, func in test_functions.items():
        logger.info(f"\nBenchmarking {func_name}...")
        results[func_name] = {}

        for size in data_sizes:
            logger.info(f"  Data size: {size}")
            data = generate_test_data(size, with_changepoint=True)

            # Benchmark with Monte Carlo simulations
            current_test += 1
            logger.info(f"    [{current_test}/{total_tests}] With MC simulations...")
            with_sim_results = benchmark_function(
                func, data, iterations=iterations, with_sim=True
            )

            # Benchmark without Monte Carlo simulations
            current_test += 1
            logger.info(f"    [{current_test}/{total_tests}] Without MC simulations...")
            without_sim_results = benchmark_function(
                func, data, iterations=iterations, with_sim=False
            )

            results[func_name][size] = {
                "with_sim": with_sim_results,
                "without_sim": without_sim_results,
            }

    logger.info("\n" + "=" * 80)
    logger.info("Benchmark Complete!")
    logger.info("=" * 80)

    return results


def format_time(seconds: float) -> str:
    """Format time in seconds to human-readable string.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted string with appropriate unit
    """
    if seconds < 0.001:
        return f"{seconds * 1_000_000:.1f} µs"
    elif seconds < 1.0:
        return f"{seconds * 1000:.2f} ms"
    else:
        return f"{seconds:.3f} s"


def write_results_to_markdown(results: dict[str, dict], output_path: Path) -> None:
    """Write benchmark results to a markdown file.

    Args:
        results: Benchmark results dictionary
        output_path: Path to output markdown file
    """
    with open(output_path, "w") as f:
        # Header
        f.write("# pyHomogeneity Benchmark Results\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # System info
        f.write("## System Information\n\n")
        f.write(f"- **Python Version:** {platform.python_version()}\n")
        f.write(f"- **Platform:** {platform.platform()}\n")
        f.write(f"- **Processor:** {platform.processor()}\n")
        f.write(f"- **Machine:** {platform.machine()}\n\n")

        # Configuration
        f.write("## Benchmark Configuration\n\n")
        f.write("- **Data Sizes:** 100, 1,000, 10,000 points\n")
        f.write("- **Iterations:** 10 runs per configuration\n")
        f.write("- **MC Simulations:** 20,000 (when enabled)\n")
        f.write("- **Alpha Level:** 0.05\n\n")

        # Results table - With Monte Carlo
        f.write("## Results: With Monte Carlo Simulations\n\n")
        f.write("| Function | 100 pts | 1,000 pts | 10,000 pts |\n")
        f.write("|----------|---------|-----------|------------|\n")

        for func_name, size_results in results.items():
            row = [func_name]
            for size in [100, 1000, 10000]:
                mean_time = size_results[size]["with_sim"]["mean"]
                row.append(format_time(mean_time))
            f.write("| " + " | ".join(row) + " |\n")

        f.write("\n")

        # Results table - Without Monte Carlo
        f.write("## Results: Without Monte Carlo Simulations\n\n")
        f.write("| Function | 100 pts | 1,000 pts | 10,000 pts |\n")
        f.write("|----------|---------|-----------|------------|\n")

        for func_name, size_results in results.items():
            row = [func_name]
            for size in [100, 1000, 10000]:
                mean_time = size_results[size]["without_sim"]["mean"]
                row.append(format_time(mean_time))
            f.write("| " + " | ".join(row) + " |\n")

        f.write("\n")

        # Analysis section
        f.write("## Key Observations\n\n")

        # Find fastest/slowest for large dataset with MC
        large_with_mc = {
            name: data[10000]["with_sim"]["mean"] for name, data in results.items()
        }
        fastest = min(large_with_mc, key=large_with_mc.get)
        slowest = max(large_with_mc, key=large_with_mc.get)

        f.write(f"### Performance Rankings (10,000 points with MC)\n\n")
        f.write(f"- **Fastest:** {fastest} ({format_time(large_with_mc[fastest])})\n")
        f.write(f"- **Slowest:** {slowest} ({format_time(large_with_mc[slowest])})\n")
        f.write(
            f"- **Ratio:** {large_with_mc[slowest] / large_with_mc[fastest]:.2f}x\n\n"
        )

        # Monte Carlo impact
        f.write("### Monte Carlo Simulation Impact\n\n")
        f.write("Average speedup when disabling MC simulations (10,000 points):\n\n")
        for func_name in results.keys():
            with_mc = results[func_name][10000]["with_sim"]["mean"]
            without_mc = results[func_name][10000]["without_sim"]["mean"]
            speedup = with_mc / without_mc
            f.write(f"- **{func_name}:** {speedup:.1f}x faster\n")

        f.write("\n")

        # Scaling analysis
        f.write("### Scaling Analysis\n\n")
        f.write("Time increase from 100 to 10,000 points (with MC):\n\n")
        for func_name in results.keys():
            time_100 = results[func_name][100]["with_sim"]["mean"]
            time_10000 = results[func_name][10000]["with_sim"]["mean"]
            ratio = time_10000 / time_100
            f.write(f"- **{func_name}:** {ratio:.1f}x\n")

        f.write("\n---\n\n")
        f.write("*Benchmark completed successfully. Use these baseline measurements ")
        f.write("to compare against post-refactoring performance.*\n")

    logger.info(f"\nResults written to: {output_path}")


def main() -> None:
    """Main entry point for benchmark script."""
    # Run benchmarks
    results = run_benchmarks()

    # Write results to markdown
    output_path = Path(__file__).parent.parent / "scratchpad" / "benchmark_results.md"
    write_results_to_markdown(results, output_path)

    logger.info("\n✓ Benchmarking complete!")
    logger.info(f"✓ Results saved to: {output_path.relative_to(Path.cwd())}")


if __name__ == "__main__":
    main()
