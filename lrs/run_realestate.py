""" Runs the algo over the real-estate input matrix and generates images.
"""
from lrs import tools
import scipy.io as sio

if __name__ == '__main__':
  mat = sio.loadmat("data/real_estate/input/Sigma_real_estate_perm.mat")
  sigma = mat['mm_tri']
  arg = { 'path': "data/real_estate/output/",
          'N': 44,
          'forced_rank': 3,
          'rank_tolerance': 0.01,
          'eps_nn': 1e-6,
          'it_nn': 4000,
          'use_previous_weights': False,
          'evaluation_only': False, }
  L, S, metrics = tools.eval_decomposition(sigma, arg)
  tools.plot_decomposition(sigma, L, S, arg['path'], vmin=-0.5, vmax=0.5)
