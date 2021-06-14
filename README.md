## Low Rank plus Sparse Decomposition of Covariance Matrices using Neural Network Parametrization

This repository is the official implementation of the paper
[Low Rank plus Sparse Decomposition of Covariance Matrices using Neural Network Parametrization](https://arxiv.org/abs/1908.00461).


## Installation

Clone this git repo and cd into it.
```sh
git clone https://github.com/CalypsoHerrera/lrs-cov-nn.git
cd lrs-cov-nn
```
Then create a virtual environment. Use equivalent command for non debian based systems.
```sh
sudo apt-get install python3-venv
python3 -m venv py3
source py3/bin/activate
pip3 install --no-cache-dir -e .
pip install --upgrade pip
```


## Updating requirements

When contributing to project, if adding dependencies, please update the
`requirements.txt` file.

```sh
pip freeze > requirements.txt
```


## Running the algorithm

### on a synthetic dataset

Generation of a low rank and of a sparse matrix. It is possible to choose the size of the matrix as well as the rank of the low rank matrix L and the sparsity of the sparse matrix S. Then the algorithm is run over the sum of them, Sigma = L+S.

```sh
python3 lrs/run_synthetic.py

```

### on a five hundred S&P500 stocks portfolio

Running the algo over a five hundred S&P500 stocks returns covariance matrix and generates images.

```sh
python3 lrs/run_sp500.py

```

### on real estate return

Running the algo over a reale sate returns covariance matrix and generates images.

```sh
python3 lrs/run_realestate.py

```



## License

This code can be used in accordance with the LICENSE.

Citation
--------

If you use this library for your publications, please cite our paper:
[Low Rank plus Sparse Decomposition of Covariance Matrices using Neural Network Parametrization](https://arxiv.org/abs/1908.00461)
```
@article{Baes2021Lowrank,
author    = {Baes, Michel and Herrera, Calypso and Neufeld, Ariel and Ruyssen, Pierre},
title     = {Low-Rank plus Sparse Decomposition of Covariance Matrices using Neural Network Parametrization
},
journal   = {CoRR},
volume    = {abs/2104.13669},
year      = {2021},
url       = {https://arxiv.org/abs/1908.00461}}
```

Last Page Update: **14/06/2021**
