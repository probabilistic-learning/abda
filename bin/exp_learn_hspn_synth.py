import os
import itertools
import functools
import operator
import datetime

import matplotlib
matplotlib.use('Agg')

import numpy as np

Ns = [2000, 5000, 10000]
Ds = [4, 8, 16]

MIN_INST_SLICES = [100, 200]
COL_SPLIT_THRESHOLD = [.3]

PYTHON_INT = 'ipython3 --'
GEN_BIN = 'bin/spstd_model_ha1.py'

OUTPUT_DIR = 'exp/synth/hspn/'
VERBOSITY_LEVEL = ' -v 0 '

# MISS_PERC = [.2]
# BETA_ROWS = (2, 5)
# BETA_COLS = (4, 5)
# TVT_SPLITS = [.7, .1, .2]

LEAF_TYPES = ['pm']
TYPE_PARAM_MAP = ['spicky-prior-1']
PARAM_INIT = ['default']
PARAM_WEIGHT_INIT = 'uniform'
SEEDS = [1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888, 9999, 1337]
EXP_ID = 'real-pos-count-cat'
FLAGS = ' --save-model '
LL_HIST = 1
SAVE_SAMPLES = 1
PLOT_HIST = 200
PERF_HIST = 1
N_ITERS = [2000]
BURN_IN = 10000
OMEGA_PRIOR = ['uniform']
OMEGA_UNIF_PRIOR = [10]
LEAF_OMEGA_UNIF_PRIOR = [0.1]
DATA_DIR = 'data/synth'

RAND_SEED = 17
rand_gen = np.random.RandomState(RAND_SEED)

if __name__ == '__main__':

    config_lists = [Ns, Ds, MIN_INST_SLICES, COL_SPLIT_THRESHOLD, SEEDS,
                    TYPE_PARAM_MAP, LEAF_TYPES, PARAM_INIT,
                    N_ITERS, OMEGA_PRIOR, OMEGA_UNIF_PRIOR, LEAF_OMEGA_UNIF_PRIOR]
    n_configs = functools.reduce(operator.mul, map(len, config_lists), 1)
    configs = itertools.product(*config_lists)

    # date_string = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    date_string = '20180502-202917'

    for i, (N, D, m_p, threshold, seed, prior,
            leaf_type, param_init, n_iters, omega_prior,
            omega_unif_prior, leaf_omega_unif_prior) in enumerate(configs):

        print('#\n# {}/{}'.format(i + 1, n_configs))

        data_path = os.path.join(DATA_DIR, '*', EXP_ID, str(N), str(D), str(seed))

        CMD = '{} {} {}'.format(PYTHON_INT, GEN_BIN, data_path)

        out_path = os.path.join(OUTPUT_DIR, date_string, EXP_ID)
        CMD += ' -o {}'.format(out_path)

        exp_name = os.path.join(str(N), str(D), str(seed), str(m_p),
                                str(threshold), prior, omega_prior)
        CMD += ' --exp-id {}'.format(exp_name)

        CMD += ' --min-inst-slice {}'.format(m_p)

        CMD += ' --col-split-threshold {}'.format(threshold)

        CMD += ' --seed {}'.format(RAND_SEED)

        CMD += ' --leaf-type {}'.format(leaf_type)

        CMD += ' --type-param-map {}'.format(prior)

        CMD += ' --param-init {}'.format(param_init)

        CMD += ' --param-weight-init {}'.format(PARAM_WEIGHT_INIT)

        CMD += FLAGS

        CMD += ' --n-iters {}'.format(n_iters)

        CMD += ' --ll-history {}'.format(LL_HIST)

        CMD += ' --plot-iter {}'.format(PLOT_HIST)

        CMD += ' --save-samples {}'.format(SAVE_SAMPLES)

        CMD += ' --perf-history {}'.format(PERF_HIST)

        CMD += ' --omega-prior {}'.format(omega_prior)

        CMD += ' --omega-unif-prior {}'.format(omega_unif_prior)

        CMD += ' --leaf-omega-unif-prior {}'.format(leaf_omega_unif_prior)

        CMD += VERBOSITY_LEVEL

        print(CMD)
        print('\n')
