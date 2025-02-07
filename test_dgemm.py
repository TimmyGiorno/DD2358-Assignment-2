import pytest
import dgemm
import numpy as np


def test_dgemm_numpy_loops():
    N = 10
    A = np.random.rand(N, N)
    B = np.random.rand(N, N)
    C_initial = np.random.rand(N, N)

    C_manual = C_initial.copy()
    dgemm.dgemm_numpy_loops(A, B, C_manual)

    C_numpy = C_initial.copy()
    C_numpy += np.matmul(A, B)

    np.testing.assert_allclose(C_manual, C_numpy, rtol=1e-5)


def test_dgemm_numpy_builtin():
    N = 10
    A = np.random.rand(N, N)
    B = np.random.rand(N, N)
    C = np.zeros((N, N))
    expected = A @ B
    result = dgemm.dgemm_numpy_builtin(A, B, C)
    np.testing.assert_allclose(result, expected)