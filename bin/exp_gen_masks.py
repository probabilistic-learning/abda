import os
import itertools
import functools
import operator
import datetime

import matplotlib
matplotlib.use('Agg')

import numpy as np

DATA_DIR = 'data'
DATASETS = {'AbaloneC': 'abalonePP',
            'ChessC': 'chessPP',
            'DermatologyC': 'dermatologyPP',
            'GermanC': 'germanPP',
            'StudentC': 'studentPP',
            'WineC': 'winePP',
            'AdultC': 'adultPP'}


PYTHON_INT = 'ipython3 --'
PP_BIN = 'bin/pre_process_islv_dataset.py'
OUTPUT_DIR = 'data/islv/'
VERBOSITY_LEVEL = ' -v 0 '


MISS_PERC = [0.1, 0.5]
MISS_VAL_PERC = 0.02
RAND_SEED = 17

# SEEDS = [1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888, 9999, 1337]
SEEDS = [1111, 2222, 3333, 4444, 5555]

rand_gen = np.random.RandomState(RAND_SEED)

if __name__ == '__main__':

    config_lists = [list(DATASETS.keys()), MISS_PERC, SEEDS]
    n_configs = functools.reduce(operator.mul, map(len, config_lists), 1)
    configs = itertools.product(*config_lists)

    date_string = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    #date_string = '20180503-120524'

    for i, (dataset, miss_perc, seed) in enumerate(configs):

        print('#\n# {}/{}'.format(i + 1, n_configs))

        CMD = '{} {}'.format(PYTHON_INT, PP_BIN)

        dataset_path = os.path.join(DATA_DIR, '{}.mat'.format(dataset))
        CMD += ' {} '.format(dataset_path)

        CMD += ' -d {} '.format(DATASETS[dataset])

        output_path = os.path.join(OUTPUT_DIR)
        CMD += ' -o {}'.format(output_path)

        CMD += '  --rm-binary '
        CMD += ' --miss-perc {}'.format(miss_perc)
        CMD += ' --miss-val-perc {}'.format(MISS_VAL_PERC)

        CMD += ' --seed {}'.format(seed)

        CMD += VERBOSITY_LEVEL

        print(CMD)
        print('\n')
