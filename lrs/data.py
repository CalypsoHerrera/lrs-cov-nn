"""generates a low rank and a sparse matrix and save them.
"""

import math
import numpy as np
import random
import scipy.io as sio


def sparsity(A, tolerance=0.01):
    """Returns ~% of zeros."""
    cnt = A.size
    non_zeros = np.count_nonzero(np.abs(A) > tolerance)
    return (cnt-non_zeros) / cnt


def spars_matrix(given_sparsity, N):
    S = np.zeros((N, N))
    current_sparsity = 1.0
    while (current_sparsity > given_sparsity):
        A = np.zeros((N, N))
        j = random.randrange(1, N)
        i = random.randrange(0, j)
        b = random.uniform(-1, 1)
        a = random.uniform(abs(b), 1)
        A[i, i] = A[j, j] = a
        A[i, j] = A[j, i] = b
        S += A
        current_sparsity = sparsity(S)
    return S


def lr_matrix(N, K):
    A = np.random.randn(N, K)
    L = A @ A.transpose((1, 0))
    return L


def filled_matrix(m_tril):
    """
    Given the number of elements of ltriangular matrix, returns N,
    where full_size=N*N. Resolution of the equation N(N+1)= len(m_tril)"""
    # N = {(n*(n+1)/2): n for n in range(1, 1000)}[len(m_tril)]
    N = math.ceil(-1+math.sqrt(1+2*len(m_tril)))
    res = np.zeros((N, N))
    i, j = np.tril_indices(N)
    res[i, j] = m_tril
    res[j, i] = m_tril
    return res


def generate_matrices(N, given_rank, sparsity):
    L0 = lr_matrix(N, given_rank)
    S0 = spars_matrix(sparsity, N)
    mat = {'L0': L0, 'S0': S0, 'N': N,
           'given_rank': given_rank, 'sparsity': sparsity}
    s_str = int(sparsity * 100)
    path = 'data/synthetic/input/'
    path_file = '%sN%s_r%s_s%s.mat' % (path, N, given_rank, s_str)
    sio.savemat(path_file, mat)
    return L0, S0


if __name__ == '__main__':
    write_matrice(N=10, given_rank=2, sparsity=0.95)
