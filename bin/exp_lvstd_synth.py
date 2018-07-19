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

PYTHON_INT = 'ipython3 --'
GEN_BIN = 'bin/lvstd.py'

OUTPUT_DIR = 'exp/synth/lvstd/'
VERBOSITY_LEVEL = ' -v 0 '

# MISS_PERC = [.2]
# BETA_ROWS = (2, 5)
# BETA_COLS = (4, 5)
# TVT_SPLITS = [.7, .1, .2]

S2Z = [1]
S2B = [1]
S2U = [0.001]
S2Y = [1]
S2THETA = [1]
SEEDS = [1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888, 9999, 1337]
EXP_ID = 'real-pos-count-cat'
LL_HIST = 100
SAVE_SAMPLES = 1
PLOT_HIST = 0
PERF_HIST = 1
N_ITERS = [2000]
BURN_IN = 1000
DATA_DIR = 'data/synth'

RAND_SEED = 17
rand_gen = np.random.RandomState(RAND_SEED)

if __name__ == '__main__':

    config_lists = [Ns, Ds,  S2Z, S2B, S2U, S2Y, S2THETA, SEEDS,
                    N_ITERS]
    n_configs = functools.reduce(operator.mul, map(len, config_lists), 1)
    configs = itertools.product(*config_lists)

    date_string = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # date_string = '20180502-202917'

    for i, (N, D, s2z, s2b, s2u, s2y, s2theta, seed, n_iters) in enumerate(configs):

        print('#\n# {}/{}'.format(i + 1, n_configs))

        data_path = os.path.join(DATA_DIR, '*', EXP_ID, str(N), str(D), str(seed))

        CMD = '{} {} {}'.format(PYTHON_INT, GEN_BIN, data_path)

        out_path = os.path.join(OUTPUT_DIR, date_string, EXP_ID, str(N), str(D), str(seed))
        CMD += ' -o {}'.format(out_path)

        # exp_name = os.path.join(str(N), str(D), str(seed))
        # CMD += ' --exp-id {}'.format(exp_name)

        CMD += ' --iters {}'.format(n_iters)

        CMD += ' --burn-in {}'.format(BURN_IN)

        CMD += ' --s2z {}'.format(s2z)

        CMD += ' --s2b {}'.format(s2b)

        CMD += ' --s2u {}'.format(s2u)

        CMD += ' --s2y {}'.format(s2y)

        CMD += ' --s2theta {}'.format(s2theta)

        k = int(D / 2)
        CMD += ' -k {}'.format(k)

        CMD += ' --ll-history {}'.format(LL_HIST)

        CMD += ' --perf-history {}'.format(PERF_HIST)

        CMD += ' --seed {}'.format(RAND_SEED)

        CMD += VERBOSITY_LEVEL

        print(CMD)
        print('\n')
