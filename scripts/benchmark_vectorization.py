"""
In credit risk, a missed payment from 30 days ago is vastly
more dangerous than one from 5 years ago.
We want to weight loans by exponential decay across 3 different
half-lives: λ ∈ [30,90,365  days]:
"""

import time

import numpy as np


# -------------------------------------------------------------
# 1. Recency Decay Weights
# -------------------------------------------------------------
def naive_time_decay(days: np.ndarray, lambdas: np.ndarray) -> np.ndarray:
    """Compute decay weights using standard nested Python loops."""
    # Step 1: Allocate output array
    out = np.zeros((len(days), len(lambdas)))

    # Step 2: Iterate through every loan and every lambda one-by-one
    for i in range(len(days)):
        for j in range(len(lambdas)):
            out[i, j] = np.exp(-abs(days[i]) / lambdas[j])
    return out


def vectorized_time_decay(days: np.ndarray, lambdas: np.ndarray) -> np.ndarray:
    """Compute decay weights using pure NumPy broadcasting."""
    # Step 1: Broadcast (N, 1) against (1, L) -> (N, L)
    return np.exp(-np.abs(days[:, None]) / lambdas[None, :])


def run_benchmark_1():
    print("\n--- Benchmark 1: Recency Decay (100,000 loans x 3 decay windows) ---")
    np.random.seed(42)
    days = np.random.uniform(-1500, -1, size=100_000)
    lambdas = np.array([30.0, 90.0, 365.0])

    # Time naive loop
    t0 = time.perf_counter()
    res_naive = naive_time_decay(days, lambdas)
    t_naive = time.perf_counter() - t0

    # Time vectorized broadcasting
    t0 = time.perf_counter()
    res_vec = vectorized_time_decay(days, lambdas)
    t_vec = time.perf_counter() - t0

    # Verify correctness
    assert np.allclose(res_naive, res_vec), "Results do not match!"

    print(f"Naive Loop Time:      {t_naive:.4f} seconds")
    print(f"Vectorized Time:      {t_vec:.4f} seconds")
    print(f"Speedup:              {t_naive / t_vec:.1f}x faster!")


# -------------------------------------------------------------
# 2. Benchmark Distance Matrix
# -------------------------------------------------------------
def naive_benchmark_distance(credits: np.ndarray, benchmarks: np.ndarray) -> np.ndarray:
    out = np.zeros((len(credits), len(benchmarks)))
    for i in range(len(credits)):
        for j in range(len(benchmarks)):
            out[i, j] = (credits[i] - benchmarks[j]) / benchmarks[j]
    return out


def vectorized_benchmark_distance(credits: np.ndarray, benchmarks: np.ndarray) -> np.ndarray:
    return (credits[:, None] - benchmarks[None, :]) / benchmarks[None, :]


def run_benchmark_2():
    print("\n--- Benchmark 2: Relative Credit Gap (100,000 credits x 5 tiers) ---")
    credits = np.random.uniform(10_000, 1_000_000, size=100_000)
    benchmarks = np.array([25_000.0, 50_000.0, 100_000.0, 250_000.0, 500_000.0])

    t0 = time.perf_counter()
    res_naive = naive_benchmark_distance(credits, benchmarks)
    t_naive = time.perf_counter() - t0

    t0 = time.perf_counter()
    res_vec = vectorized_benchmark_distance(credits, benchmarks)
    t_vec = time.perf_counter() - t0

    assert np.allclose(res_naive, res_vec), "Benchmark 2 mismatch!"
    print(f"Naive Loop Time:      {t_naive:.4f} seconds")
    print(f"Vectorized Time:      {t_vec:.4f} seconds")
    print(f"Speedup:              {t_naive / t_vec:.1f}x faster!")


# -------------------------------------------------------------
# 3. Matrix Standardization (Z-score)
# -------------------------------------------------------------
def naive_standardize(X: np.ndarray) -> np.ndarray:
    n_rows, n_cols = X.shape
    out = np.zeros_like(X)
    means = np.zeros(n_cols)
    stds = np.zeros(n_cols)

    # Compute means per column
    for j in range(n_cols):
        col_sum = 0.0
        for i in range(n_rows):
            col_sum += X[i, j]
        means[j] = col_sum / n_rows

    # Compute std per column
    for j in range(n_cols):
        var_sum = 0.0
        for i in range(n_rows):
            var_sum += (X[i, j] - means[j]) ** 2
        stds[j] = np.sqrt(var_sum / n_rows)

    # Standardize cell by cell
    for i in range(n_rows):
        for j in range(n_cols):
            out[i, j] = (X[i, j] - means[j]) / stds[j]
    return out


def vectorized_standardize(X: np.ndarray) -> np.ndarray:
    means = np.mean(X, axis=0)  # shape (10,)
    stds = np.std(X, axis=0)  # shape (10,)
    return (X - means[None, :]) / stds[None, :]


def run_benchmark_3():
    print("\n--- Benchmark 3: Matrix Standardization (100,000 rows x 10 features) ---")
    X = np.random.normal(loc=50.0, scale=15.0, size=(100_000, 10))

    t0 = time.perf_counter()
    res_naive = naive_standardize(X)
    t_naive = time.perf_counter() - t0

    t0 = time.perf_counter()
    res_vec = vectorized_standardize(X)
    t_vec = time.perf_counter() - t0

    assert np.allclose(res_naive, res_vec), "Benchmark 3 mismatch!"
    print(f"Naive Loop Time:      {t_naive:.4f} seconds")
    print(f"Vectorized Time:      {t_vec:.4f} seconds")
    print(f"Speedup:              {t_naive / t_vec:.1f}x faster!")


if __name__ == "__main__":
    run_benchmark_1()
    run_benchmark_2()
    run_benchmark_3()
