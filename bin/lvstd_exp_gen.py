import os
import itertools

import numpy

# from mlutils.datasets import loadMLC


def cartesian_product_dict_list(d):
    return (dict(zip(d, x)) for x in itertools.product(*d.values()))


BASE_PATH = os.path.dirname(__file__)
DATA_DIR = 'data'


DATASETS = ['AbaloneC.mat',
            # 'AdultC.mat',
            'ChessC.mat',
            'DermatologyC.mat',
            'GermanC.mat',
            'StudentC.mat',
            'WineC.mat']


ITERs = [1000]
BURN_INs = [5000]
Ks = [2, 5]
LL_HISTORY = [10]
S2Zs = [1]
S2Bs = [1]
S2Ys = [1]
S2Us = [0.001]
S2THETAs = [1]

PYTHON_INTERPRETER = 'ipython -- '
# PYTHON_INTERPRETER = 'python3 '
CMD_LINE_BIN_PATH = ' bin/lvstd.py '
VERBOSITY_LEVEL = ' -v 2 '
# SEED = 1337

for data, it, b, k, ll_history, s2z, s2b, s2y, s2u, s2theta in itertools.product(reversed(DATASETS),
                                                                                 ITERs,
                                                                                 BURN_INs,
                                                                                 Ks,
                                                                                 LL_HISTORY,
                                                                                 S2Zs,
                                                                                 S2Bs,
                                                                                 S2Ys,
                                                                                 S2Us,
                                                                                 S2THETAs):

    cmd = '{} {}'.format(PYTHON_INTERPRETER, CMD_LINE_BIN_PATH)
    cmd += '{} '.format(os.path.join(DATA_DIR, data))
    cmd += ' -i {} -b {} -k {}'.format(it, b, k)
    cmd += ' --ll-history {}'.format(ll_history)
    cmd += ' --s2z {} --s2b {} --s2y {} --s2u {} --s2theta {}'.format(s2z, s2b, s2y, s2u, s2theta)
    cmd += ' {}'.format(VERBOSITY_LEVEL)

    print(cmd)
