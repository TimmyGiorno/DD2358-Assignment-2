import numpy as np
import time
import matplotlib.pyplot as plt


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


def dgemm_python_lists(A, B, C):
    N = len(A)
    for i in range(N):
        for j in range(N):
            for k in range(N):
                C[i][j] += A[i][k] * B[k][j]
    return C


def measure_time(func, A, B, C, trials=5):
    times = []
    for _ in range(trials):
        C_copy = [row[:] for row in C] if isinstance(C, list) else C.copy()
        start = time.perf_counter()
        func(A, B, C_copy)
        end = time.perf_counter()
        times.append(end - start)
    return np.mean(times), np.std(times)


def calculate_flops(N, time):
    """Calculate FLOPS"""
    flops = 2 * N ** 3  # Each element involves 2 operations (multiplication and addition) per element
    return flops / time


# Test different matrix sizes
sizes = [10, 50, 100]
results = {}

theoretical_peak = 3.0e9 * 16  # 48 GFLOPS

for N in sizes:
    A_np = np.random.rand(N, N)
    B_np = np.random.rand(N, N)
    C_np = np.zeros((N, N))

    A_list = A_np.tolist()
    B_list = B_np.tolist()
    C_list = [[0.0] * N for _ in range(N)]

    mean_loops, std_loops = measure_time(dgemm_numpy_loops, A_np, B_np, C_np)
    mean_builtin, std_builtin = measure_time(dgemm_numpy_builtin, A_np, B_np, C_np)
    mean_lists, std_lists = measure_time(dgemm_python_lists, A_list, B_list, C_list)

    results[N] = {
        'loops': (mean_loops, std_loops),
        'builtin': (mean_builtin, std_builtin),
        'lists': (mean_lists, std_lists)
    }

    t_loops = results[N]['loops'][0]
    flops_loops = calculate_flops(N, t_loops)
    t_builtin = results[N]['builtin'][0]
    flops_builtin = calculate_flops(N, t_builtin)
    t_lists = results[N]['lists'][0]
    flops_lists = calculate_flops(N, t_lists)

    print(f"N={N}")
    print(f"  Loops: Mean={results[N]['loops'][0]:.6f}s, Std={results[N]['loops'][1]:.6f}")
    print(f"  Built-in: Mean={results[N]['builtin'][0]:.6f}s, Std={results[N]['builtin'][1]:.6f}")
    print(f"  Python Lists: Mean={results[N]['lists'][0]:.6f}s, Std={results[N]['lists'][1]:.6f}")
    print(f"  Triple Loop FLOPS: {flops_loops / 1e9:.2f} GFLOPS")
    print(f"  Built-in Implementation FLOPS: {flops_builtin / 1e9:.2f} GFLOPS")
    print(f"  Python Lists FLOPS: {flops_lists / 1e9:.2f} GFLOPS")
    print(f"  Theoretical Peak: {theoretical_peak / 1e9:.2f} GFLOPS\n")

sizes = list(results.keys())
times_loops = [results[N]['loops'][0] for N in sizes]
times_builtin = [results[N]['builtin'][0] for N in sizes]
times_lists = [results[N]['lists'][0] for N in sizes]

plt.figure(figsize=(10, 6))
plt.plot(sizes, times_loops, 'o-', label='Triple Loop (NumPy Arrays)')
plt.plot(sizes, times_builtin, 's-', label='NumPy matmul (BLAS)')
plt.plot(sizes, times_lists, 'd-', label='Triple Loop (Python Lists)')
plt.xlabel('Matrix Size (N)')
plt.ylabel('Time (s)')
plt.yscale('log')
plt.title('DGEMM Performance Comparison')
plt.legend()
plt.grid(True)
plt.show()