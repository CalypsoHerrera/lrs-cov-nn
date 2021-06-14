""" Runs the algo over a portfolio of 500 stocks within the S&P500 input matrix and generates images.
"""
import numpy as np
from lrs import tools

if __name__ == '__main__':
  Sigma = np.loadtxt('data/sp500/input/sp500.csv', delimiter=',')
  arg = { 'path': 'data/sp500/output/',
          'N': 500,
          'forced_rank': 3,
          'rank_tolerance': 0.01,
          'eps_nn': 1e-6,
          'it_nn': 50,
          'use_previous_weights': False,
          'evaluation_only': False}
  L, S, metrics = tools.eval_decomposition(Sigma, arg)
  tools.plot_decomposition(Sigma, L, S, arg['path'], vmin=-0.2, vmax=0.9)
