# Automatic Bayesian Density Analysis

This repository contains the code and the supplementary material of the paper 

_Antonio Vergari, Alejandro Molina, Robert Peharz, Zoubin Ghahramani, Kristian Kersting and Isabel Valera_  
"[**Automatic Bayesian Density Analysis**]()"

In Proceedings of the Thirty-third AAAI Conference on Artificial Intelligence (AAAI'19)


## Overview

**ABDA** is a hierarchical probabilistic model taking into account both the uncertainties around random variable (RV) interactions and their (parametric) likelihood models.

**ABDA** allows to deal with __*data heterogeneity*__ (of statistical data types and likelihood models) and enables __*efficient probabilistic inference*__ in those domains through a __*latent variable structure*__ via sum-product networks.

## Requirements

### Python packages

The code relies on the following `python3.4+` libs:

  * [numpy (1.10+)](https://www.numpy.org/),
  * [scipy (0.17+)](https://www.scipy.org/),
  * [sklearn (0.17+)](https://scikit-learn.org/stable/),
  * [pandas (0.15+)](https://pandas.pydata.org/),
  * [numba (0.27+)](https://numba.pydata.org/),
  * [matplotlib (2.0+)](https://matplotlib.org/) and [seaborn (0.6+)](https://seaborn.pydata.org/) for plots and visualizations,
  * tentative probabilistic programming integration is provided via [pymc3 (3.4+)](http://docs.pymc.io/index.html) and [pyro (0.2+)](http://pyro.ai/) 

### Numerical libraries (optional)
For optimal `theano` (for pymc) and `numpy` performances one can exploit the
`blas` and `lapack` libs. CUDA is required to exploit the GPU for inference.
To properly install them please refer to
[this page](http://deeplearning.net/software/theano/install.html)
The lib versions used on a Ubuntu 14.04 installation are:

```
liblapack3 3.5.0-2
libopenblas-dev 0.2.8-6
cuda 7+
```

## Data

The `data` folder contains both the synthetically generated datasets (refer to Appendix C) in `data\synth` and the real-world UCI datasets in `data\real` (refer to Table 1).

Synthetic datasets are organized in `data/synth/<N>/<D>/<SEED>` subfolders, regarding the number of samples `N` and dimensions `D` and `Seed` employed to generate them.

Each folder in the format `real/xxxPP` contains the missing data masks (in the `miss` subfolder, both for 10 and 50 percent of training data) for dataset `xxx` and the original data, as translated in the MATLAB format of [1] both for the `transductive` and `inductive` scenarios (refer to the "Experiment" section of the paper).

[1] _Valera, Isabel, and Zoubin Ghahramani_ 

"Automatic discovery of the statistical types of variables in a dataset." 

ICML 2017.

## Usage

[ipython (3.2+)](https://ipython.org/) has been used to launch all scripts.
The following commands will assume the use of `ipython` (`python3` interpreter is fine as well) and being in
the repo main directory:

```
cd abda
```

To run ABDA, use the `bin/abda.py` script. The most important parameters to be specified are

```
ipython3 -- bin/abda.py <path-to-data> -o <output-path> --min-inst-slice <N> --col-split-threshold <T> --seed <SEED> --type-param-map <prior-dict-name> --param-init <init-scheme> --param-weight-init <init-scheme> --save-model  --n-iters <I> --burn-in <B> --ll-history <LL> --plot-iter <PP> --save-samples <SS> --omega-prior <prior-scheme> --omega-unif-prior <prior-val> --leaf-omega-unif-prior <prior-val>
```
where:

  - `--min-inst-slice` specifies the minimum number of instance `N` before stopping learning the LV structure via SPN structure learning
  - `--col-split-threshold` determines the threshold `T` to consider two RVs independent via the RDC during structure learning
  - `--seed` sets the random number generator seed
  - `--type-param-map` selects the name of a map containing the likelihood models priors (see `abda.py`)
  - `--param-init` specifies how to initialize the prior hyperparameters (`default` means MLE, when possible)
  - `--param-weight-init` specifies how to initialize the likelihood dictionary weights (defaults to sparse `uniform`)
  - `--save-model` saves the learned model as `pickle` object
  - `--n-iters` is the total number of Gibbs sampling iterations `I`
  - `--burn-in` tells how many samples `B` to discard 
  - `--ll-history` when to save log-likelihood evaluations
  - `--plot-iter` whether to plot partial fitting for all leaf distributions
  - `--save-samples` whether to serialize drawn samples
  - `--omega-prior` specifies how to initialize the sum node weights (defaults to `uniform`)
  - `--omega-unif-prior` specifies the value for sum node weights, if uniform (defaults to 10)
  - `--leaf-omega-unif-prior` specifies the value for likelihood dictionary weights if uniform (defaults to 0.1)

For instance, to perform inference with ABDA on the `wine` dataset, run:

```
ipython3 -- bin/abda.py data/real/winePP  -o exp/wine-output --min-inst-slice 500 --col-split-threshold 0.1 --seed 17 --type-param-map wider-prior-2 --param-init default --param-weight-init uniform --save-model  --n-iters 5 --burn-in 2 --ll-history 200 --plot-iter 0 --save-samples 1  --omega-prior uniform --omega-unif-prior 10 --leaf-omega-unif-prior 0.1
```

	
For default values please refer to the documentation with

```
ipython3 -- bin/abda.py --help
```

and for the hyperparameters employed in the experiments see the paper and the supplementary material in the `supplementary` folder.





	 

