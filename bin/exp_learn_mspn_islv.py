import datetime

from bin import exp_learn_hspn_inductive as ind
from bin import exp_learn_hspn_islv as trn
import numpy as np

rand_gen = np.random.RandomState(ind.RAND_SEED)


def generate_inductive_for(mspn_type="histogram"):
    ind.EXEC_BIN = "bin/mspn_model.py"
    ind.OUTPUT_DIR = "exp/islv-inductive/mspn-" + mspn_type + "/"
    ind.TYPE_PARAM_MAP = ['spicky-prior-1']
    ind.OMEGA_PRIOR = [mspn_type]
    ind.FLAGS = " --save-model --mspn-leaves " + mspn_type
    configs = ind.get_configs()
    date_string = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return ind.dump_exp_sh_call(date_string, configs)


def generate_transductive_for(mspn_type="histogram"):
    trn.EXEC_BIN = "bin/mspn_model.py"
    trn.OUTPUT_DIR = "exp/islv-transductive/mspn-" + mspn_type + "/"
    trn.TYPE_PARAM_MAP = ['spicky-prior-1']
    trn.OMEGA_PRIOR = [mspn_type]
    trn.FLAGS = " --save-model --mspn-leaves " + mspn_type
    configs = trn.get_configs()
    date_string = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return trn.dump_exp_sh_call(date_string, configs)


if __name__ == '__main__':
    with open('scripts/mspn-train-transductive.sh', 'w') as f:
        f.writelines("%s\n" % l for l in generate_transductive_for("histogram"))
        f.writelines("%s\n" % l for l in generate_transductive_for("piecewise"))
    with open('scripts/mspn-train-inductive.sh', 'w') as f:
        f.writelines("%s\n" % l for l in generate_inductive_for("histogram"))
        f.writelines("%s\n" % l for l in generate_inductive_for("piecewise"))

#    print("""
#ipython3 -- bin/mspn_model.py data/islv/anneal-UPP/anneal-UPP.mat --dummy-id 1 -o exp/islv/mspn-hist/20180513-121449/real-pos-count-cat/anneal-UPP --exp-id 0.1/0.7/wider-prior-1/histogram/0.1/2222 --miss data/islv/anneal-UPP/miss/0.1/2222/miss.full.data --min-inst-slice 89 --col-split-threshold 0.7 --seed 17 --leaf-type pm --type-param-map wider-prior-1 --param-init default --param-weight-init histogram --save-model  --n-iters 5000 --burn-in 4000 --ll-history 100 --plot-iter 0 --save-samples 1 --perf-history 1 --omega-prior histogram --omega-unif-prior 10 --leaf-omega-unif-prior 0.1 -v 0
#    """)
