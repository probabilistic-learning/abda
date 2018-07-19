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

MIN_PART_SAMPLES = [0.25]
COL_THRESHOLD = [.1]

TYPES = ['real', 'pos', 'count', 'cat']

PYTHON_INT = 'ipython3 --'

GEN_BIN = 'bin/gen_synth_data.py'

OUTPUT_DIR = 'data/synth'
VERBOSITY_LEVEL = ' -v 2 '

MISS_PERC = [.2]

BETA_ROWS = (2, 5)
BETA_COLS = (4, 5)

TVT_SPLITS = [.7, .1, .2]

PRIORS = ['prior-1']

SEEDS = [1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888, 9999, 1337]

EXP_ID = 'real-pos-count-cat'

RAND_SEED = 17
rand_gen = np.random.RandomState(RAND_SEED)

if __name__ == '__main__':

    config_lists = [Ns, Ds, MIN_PART_SAMPLES, COL_THRESHOLD, MISS_PERC, SEEDS, PRIORS]
    n_configs = functools.reduce(operator.mul, map(len, config_lists), 1)
    configs = itertools.product(*config_lists)

    date_string = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    for i, (N, D, m_p, threshold, mv_perc, seed, prior) in enumerate(configs):

        print('#\n# {}/{}'.format(i + 1, n_configs))

        CMD = '{} {} {}/{}'.format(PYTHON_INT, GEN_BIN, date_string, EXP_ID)

        CMD += ' -o {}'.format(OUTPUT_DIR)

        exp_name = os.path.join(str(N), str(D), str(seed))
        CMD += ' --exp-id {}'.format(exp_name)

        #
        # randomly select D types
        types = rand_gen.choice(TYPES, size=D)
        type_str = ' '.join(types)
        CMD += ' -t {}'.format(type_str)

        CMD += ' --samples {}'.format(N)

        m = int(m_p * N)
        CMD += ' --min-instances {}'.format(m)

        CMD += ' --col-split-threshold {}'.format(threshold)

        CMD += ' --seed {}'.format(seed)

        CMD += ' --priors {}'.format(prior)

        CMD += ' --tvt-split {}'.format(' '.join(str(p) for p in TVT_SPLITS))

        CMD += ' --beta-rows {}'.format(' '.join(str(p) for p in BETA_ROWS))
        CMD += ' --beta-cols {}'.format(' '.join(str(p) for p in BETA_COLS))

        CMD += ' --miss-perc {}'.format(mv_perc)

        CMD += VERBOSITY_LEVEL

        print(CMD)
        print('\n')
