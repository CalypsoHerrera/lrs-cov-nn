""" some tools in order to plot the images, save the matrix and evaluate the
decomposition.
"""
import time
import numpy as np
import matplotlib.pyplot as plt

from lrs import data
from lrs import algo


def plot(matrix, vmin, vmax, output_file):
    plt.imshow(matrix, cmap='Spectral', vmin=vmin, vmax=vmax)
    plt.axis('off')
    plt.colorbar()
    plt.tight_layout(pad=0, w_pad=0, h_pad=0)
    plt.savefig(output_file)
    plt.close()

def plot_decomposition(Sigma, L, S, path, vmin, vmax):
    plot(Sigma, vmin, vmax, '%sSigma.pdf' % path)
    plot(L, vmin, vmax, '%sL.pdf' % path)
    plot(S, vmin, vmax, '%sS.pdf' % path)

def save_decoposition(Sigma, L, S, path):
    np.savetxt('%sSigma.csv' % path, Sigma, delimiter=",")
    np.savetxt('%sL.csv' % path, L, delimiter=",")
    np.savetxt('%sS.csv' % path, S, delimiter=",")

def eval_decomposition(Sigma, arg, L0=None, S0=None):
    before = time.time()
    L, S = algo.algo(Sigma, arg)
    after = time.time()
    save_decoposition(Sigma, L, S, arg['path'])

    metrics = {
           'rank_of_L': np.linalg.matrix_rank(L, arg['rank_tolerance']),
           'sparsity_of_S': round(data.sparsity(S),2),
           'norm_of_S': round(np.linalg.norm(S, 1),2),
           'duration_in_seconds': round(after - before,2),
           }
    if S0 is not None:
        metrics['norm_of_S_minus_S0'] = round(np.linalg.norm(S-S0),2)
        metrics['norm_of_L_minus_L0'] = round(np.linalg.norm(L-L0),2)

    np.save("%smetrics.npy" % arg['path'], metrics)
    print("The matrices and metrics of the decomposition are save in", arg['path'], "\nMetrics:", metrics)
    return L, S, metrics
