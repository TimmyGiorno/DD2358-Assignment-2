import numpy as np
import time


def dgemm_numpy_loops(A, B, C):
    N = A.shape[0]
    for i in range(N):
        for j in range(N):
            for k in range(N):
                C[i, j] += A[i, k] * B[k, j]
    return C


def dgemm_numpy_builtin(A, B, C):
    C += np.matmul(A, B)
    return C


def measure_time(func, A, B, C, trials=5):
    """ Measure the execution time of a function """
    times = []
    for _ in range(trials):
        C_copy = C.copy()
        start = time.perf_counter()
        func(A, B, C_copy)
        end = time.perf_counter()
        times.append(end - start)
    return np.mean(times), np.std(times)


def calculate_flops(N, time):
    """Calculate FLOPS"""
    flops = 2 * N**3  # Each element involves 2 operations (multiplication and addition) per element
    return flops / time


# Test different matrix sizes
sizes = [10, 50, 100]
results = {}

# Theoretical peak (a 3.0 GHz CPU with AVX2 support, 16 FLOPS per cycle)
theoretical_peak = 3.0e9 * 16  # 48 GFLOPS

for N in sizes:
    A = np.random.rand(N, N)
    B = np.random.rand(N, N)
    C = np.zeros((N, N))

    # Measure the triple loop implementation
    mean_loops, std_loops = measure_time(dgemm_numpy_loops, A, B, C)

    # Measure the built-in implementation
    mean_builtin, std_builtin = measure_time(dgemm_numpy_builtin, A, B, C)

    results[N] = {
        'loops': (mean_loops, std_loops),
        'builtin': (mean_builtin, std_builtin)
    }

    t_loops = results[N]['loops'][0]
    flops_loops = calculate_flops(N, t_loops)
    t_builtin = results[N]['builtin'][0]
    flops_builtin = calculate_flops(N, t_builtin)

    print(f"N={N}")
    print(f"  Loops: Mean={results[N]['loops'][0]:.6f}s, Std={results[N]['loops'][1]:.6f}")
    print(f"  Built-in: Mean={results[N]['builtin'][0]:.6f}s, Std={results[N]['builtin'][1]:.6f}")
    print(f"  Triple Loop FLOPS: {flops_loops / 1e9:.2f} GFLOPS")
    print(f"  Built-in Implementation FLOPS: {flops_builtin / 1e9:.2f} GFLOPS")
    print(f"  Theoretical Peak: {theoretical_peak / 1e9:.2f} GFLOPS")


import matplotlib.pyplot as plt

sizes = list(results.keys())
times_loops = [results[N]['loops'][0] for N in sizes]
times_builtin = [results[N]['builtin'][0] for N in sizes]

plt.figure(figsize=(10, 6))
plt.plot(sizes, times_loops, 'o-', label='Triple Loop (NumPy Arrays)')
plt.plot(sizes, times_builtin, 's-', label='NumPy matmul (BLAS)')
plt.xlabel('Matrix Size (N)')
plt.ylabel('Time (s)')
plt.yscale('log')
plt.title('DGEMM Performance Comparison')
plt.legend()
plt.grid(True)
plt.show()