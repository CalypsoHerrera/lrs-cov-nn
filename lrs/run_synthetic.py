""" Generates a low rank and a sparse matrix and runs the algo over the sum of them.
"""
import scipy.io as sio
from lrs import data
from lrs import tools


if __name__ == '__main__':
  arg = { 'path': 'data/synthetic/output/',
          'N': 10,
          'forced_rank': 2,
          'given_rank': 2,
          'sparsity': 0.95,
          'rank_tolerance': 0.01,
          'eps_nn': 1e-6,
          'it_nn': 4000,
          'use_previous_weights': False,
          'evaluation_only': False,}
  L0, S0 = data.generate_matrices(arg['N'], arg['given_rank'], arg['sparsity'])
  Sigma = L0 + S0
  L, S, metrics = tools.eval_decomposition(Sigma, arg)
  tools.plot_decomposition(Sigma, L, S, arg['path'], vmin=-0.5, vmax=0.5)
