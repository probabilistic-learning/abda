import os
import itertools
import functools
import operator
import datetime

import matplotlib
matplotlib.use('Agg')

import numpy as np

DATA_DIR = 'data/islv'
DATASETS = [
    # 'abalonePP',
    #         'chessPP',
    # 'dermatologyPP',
    # 'germanPP',
    #         'studentPP',
    # 'winePP',
    # 'adultPP',
    'anneal-UPP',
    'australianPP',
    'autismPP',
    'breastPP',
    'crxPP',
    'diabetesPP',
]

DATASET_SIZES = {'abalonePP': 9,
                 'chessPP': 7,
                 'dermatologyPP': 35,
                 'germanPP': 17,
                 'studentPP': 20,
                 'winePP': 12,
                 'adultPP': 13,
                 'anneal-UPP': 20,
                 'australianPP': 10,
                 'autismPP': 25,
                 'breastPP': 10,
                 'crxPP': 11,
                 'diabetesPP': 8,
                 }

# Ks = [2, 5, 10]

PYTHON_INT = 'ipython3 --'
LVSTD_BIN = 'bin/lvstd.py'
OUTPUT_DIR = 'exp/islv/lvstd/'
VERBOSITY_LEVEL = ' -v 0 '

ITERS = [5000]
BURN_IN = 4000
S2Z = [1]
S2B = [1]
S2U = [0.001]
S2Y = [1]
S2THETA = [1]
LL_HIST = 100
PERF_HIST = 1

MISS_PERC = [0.1, 0.5]
RAND_SEED = 17

# SEEDS = [1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888, 9999, 1337]
SEEDS = [1111, 2222, 3333, 4444, 5555]

rand_gen = np.random.RandomState(RAND_SEED)

if __name__ == '__main__':

    config_lists = [DATASETS, ITERS, S2Z, S2B, S2U, S2Y, S2THETA, MISS_PERC, SEEDS]
    n_configs = functools.reduce(operator.mul, map(len, config_lists), 1)
    configs = itertools.product(*config_lists)

    date_string = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # date_string = '20180503-120524'

    for i, (dataset, iters, s2z, s2b, s2u, s2y, s2theta, miss_perc, seed) in enumerate(configs):

        D = DATASET_SIZES[dataset]
        k = int(D / 2)

        print('#\n# {}/{}'.format(i + 1, n_configs))

        CMD = '{} {}'.format(PYTHON_INT, LVSTD_BIN)

        dataset_path = os.path.join(DATA_DIR, dataset, '{}.mat'.format(dataset))
        CMD += ' {} '.format(dataset_path)

        output_path = os.path.join(OUTPUT_DIR, date_string, dataset,
                                   str(k), str(miss_perc), str(seed))
        CMD += ' -o {}'.format(output_path)

        miss_path = os.path.join(DATA_DIR, dataset, 'miss', str(
            miss_perc), str(seed), 'miss.full.data')
        CMD += ' --miss {}'.format(miss_path)

        CMD += ' --iters {}'.format(iters)

        CMD += ' --burn-in {}'.format(BURN_IN)

        CMD += ' --s2z {}'.format(s2z)

        CMD += ' --s2b {}'.format(s2b)

        CMD += ' --s2u {}'.format(s2u)

        CMD += ' --s2y {}'.format(s2y)

        CMD += ' --s2theta {}'.format(s2theta)

        CMD += ' -k {}'.format(k)

        CMD += ' --ll-history {}'.format(LL_HIST)

        CMD += ' --perf-history {}'.format(PERF_HIST)

        CMD += ' --seed {}'.format(RAND_SEED)

        CMD += VERBOSITY_LEVEL

        print(CMD)
        print('\n')
