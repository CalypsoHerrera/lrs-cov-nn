""" Low-Rank plus Sparse Decomposition of Covariance Matrices using
Neural Network Parametrization
Michel Baes, Calypso Herrera, Ariel Neufeld, Pierre Ruyssen

https://arxiv.org/abs/1908.00461
"""


import numpy as np
import torch
from lrs import data

def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def algo(sigma, arg):
    i, j = np.tril_indices(arg['N'])
    sigma_tril = torch.FloatTensor(sigma[i, j])
    L = torch.zeros([arg['N'],  arg['forced_rank']])
    S = torch.zeros([arg['N'],  arg['forced_rank']])
    input_size = int(arg['N'] * (arg['N'] + 1) / 2)
    output_size = arg['N'] * arg['forced_rank']
    H1, H2, H3, H4 = 1000, 200, 200, 200
    learning_rate = 1e-4
    model = torch.nn.Sequential(torch.nn.Linear(input_size, H1), torch.nn.ReLU(),
                                torch.nn.Linear(H1, H2), torch.nn.ReLU(),
                                torch.nn.Linear(H2, H3), torch.nn.ReLU(),
                                torch.nn.Linear(H3, H4), torch.nn.ReLU(),
                                torch.nn.Linear(H4, output_size))
    trained = False
    t_list, loss_list = [], []
    weights_file = arg['path']+'weights.csv'
    if arg['use_previous_weights'] and os.path.exists(weights_file):
        model.load_state_dict(torch.load(weights_file))
        trained = True
        t_list = [0]
    else:
        trained = False
        t_list = [t for t in range(arg['it_nn'])]
        model.apply(init_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    last_values = []
    it, diff_loss, average_lookback_size = 0, 1, 10
    while diff_loss > arg['eps_nn'] and it <= arg['it_nn']:
        it += 1
        if it > average_lookback_size:
            last_values.pop(0)
        sigma_tril.requires_grad_(True)
        M = model(sigma_tril)
        M = M.reshape((arg['N'], arg['forced_rank']))
        L = M @ M.transpose(-1, -2)
        sigma_tril = sigma_tril.detach()
        sigma = torch.FloatTensor(data.filled_matrix(sigma_tril.numpy()))
        S = sigma - L
        if arg['use_previous_weights'] and arg['evaluation_only'] and os.path.exists(weights_file):
            break
        loss = abs(S).sum(dim=(0, 1))
        optimizer.zero_grad()
        loss.backward()
        if it > average_lookback_size:
            diff_loss = abs(loss.item() - np.mean(last_values)) / \
                np.mean(last_values)
        last_values.append(loss.item())
        loss_list.append(loss.item())
        # print(it, loss.item())
        optimizer.step()

    torch.save(model.state_dict(), weights_file)
    L_out = L.detach().numpy()
    S_out = S.detach().numpy()
    return L_out, S_out
