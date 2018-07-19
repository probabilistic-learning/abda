import os
import itertools
import functools
import operator
import datetime


import numpy as np

DATA_DIR = 'data/islv'
DATASETS = ['abalonePP',
            'chessPP',
            'germanPP',
            'studentPP',
            'winePP',
            'dermatologyPP',
            'adultPP',
            'anneal-UPP',
            'australianPP',
            'autismPP',
            'breastPP',
            'crxPP',
            'diabetesPP',
            ]
Ns = {'abalonePP': 4177,
      'adultPP': 32561,
      'chessPP': 28056,
      'dermatologyPP': 366,
      'germanPP': 1000,
      'studentPP': 395,
      'winePP': 6497,
      'anneal-UPP': 898,
      'australianPP': 690,
      'autismPP': 3521,
      'breastPP': 681,
      'crxPP': 651,
      'diabetesPP': 768, }
MIN_INST_SLICES = [.1, .2, .3]
COL_SPLIT_THRESHOLD = [.9, .7, .5, .3]

PYTHON_INT = 'ipython3 --'
EXEC_BIN = 'bin/spstd_model_ha1.py'
OUTPUT_DIR = 'exp/islv-inductive/hspn/'
VERBOSITY_LEVEL = ' -v 0 '

LEAF_TYPES = ['pm']
TYPE_PARAM_MAP = [
    'spicky-prior-1',
    'wider-prior-1']
PARAM_INIT = ['default']
PARAM_WEIGHT_INIT = 'uniform'
EXP_ID = 'real-pos-count-cat'
FLAGS = ' --save-model '
LL_HIST = 1
SAVE_SAMPLES = 1
PLOT_HIST = 0
PERF_HIST = 1
N_ITERS = [5000]
BURN_IN = 4000
OMEGA_PRIOR = ['uniform']
OMEGA_UNIF_PRIOR = [10]
LEAF_OMEGA_UNIF_PRIOR = [0.1]

# MISS_PERC = [0.1, 0.5]
RAND_SEED = 17

# SEEDS = [1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888, 9999, 1337]
# SEEDS = [1111, 2222, 3333, 4444, 5555]


rand_gen = np.random.RandomState(RAND_SEED)


def dump_exp_sh_call(date_string, configs):
    lines = []
    for i, (dataset, m_p, threshold, prior, leaf_type, param_init, n_iters, omega_prior,
            omega_unif_prior, leaf_omega_unif_prior) in enumerate(configs):
        CMD = '\necho {}/{}\n'.format(i + 1, len(configs))

        dataset_path = os.path.join(DATA_DIR, dataset)
        CMD += '{} {} {}'.format(PYTHON_INT, EXEC_BIN, dataset_path)

        CMD += ' --dummy-id {}'.format(i)

        out_path = os.path.join(OUTPUT_DIR, date_string, EXP_ID, dataset)
        CMD += ' -o {}'.format(out_path)

        exp_name = os.path.join(str(m_p), str(threshold), prior, omega_prior)
        CMD += ' --exp-id {}'.format(exp_name)

        m = int(m_p * Ns[dataset])
        CMD += ' --min-inst-slice {}'.format(m)

        CMD += ' --col-split-threshold {}'.format(threshold)

        CMD += ' --seed {}'.format(RAND_SEED)

        CMD += ' --leaf-type {}'.format(leaf_type)

        CMD += ' --type-param-map {}'.format(prior)

        CMD += ' --param-init {}'.format(param_init)

        CMD += ' --param-weight-init {}'.format(PARAM_WEIGHT_INIT)

        CMD += FLAGS

        CMD += ' --n-iters {}'.format(n_iters)

        CMD += ' --burn-in {}'.format(BURN_IN)

        CMD += ' --ll-history {}'.format(LL_HIST)

        CMD += ' --plot-iter {}'.format(PLOT_HIST)

        CMD += ' --save-samples {}'.format(SAVE_SAMPLES)

        CMD += ' --perf-history {}'.format(PERF_HIST)

        CMD += ' --omega-prior {}'.format(omega_prior)

        CMD += ' --omega-unif-prior {}'.format(omega_unif_prior)

        CMD += ' --leaf-omega-unif-prior {}'.format(leaf_omega_unif_prior)

        CMD += VERBOSITY_LEVEL

        lines.append(CMD)
    return lines


def get_configs():
    config_lists = [DATASETS, MIN_INST_SLICES, COL_SPLIT_THRESHOLD,
                    TYPE_PARAM_MAP, LEAF_TYPES, PARAM_INIT,
                    N_ITERS, OMEGA_PRIOR, OMEGA_UNIF_PRIOR, LEAF_OMEGA_UNIF_PRIOR]
    return list(itertools.product(*config_lists))


if __name__ == '__main__':
    configs = get_configs()

    date_string = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # date_string = '20180504-202223'

    for line in dump_exp_sh_call(date_string, configs):
        print(line)
