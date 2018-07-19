'''
Created on May 13, 2018

@author: Alejandro Molina
'''
import datetime
import glob
import pickle


def get_command(args, path, metric):
    ds_name = args.output.split('/')[-1]
    exp_id = args.exp_id.split('/')

    # --min-inst-slice 0.1 0.2 0.3

    cmd = "ipython3 -- bin/collect_results.py "
    cmd += path

    cmd += " --threshold 0.3 0.5 0.7"
    cmd += " --prior " + args.type_param_map
    cmd += " --leaf-type " + args.mspn_leaves
    cmd += " --seeds 1111 2222 3333 4444 5555"

    cmd += " --metrics " + metric
    cmd += " --datasets " + ds_name
    cmd += " --mspn"
    cmd += " --data-path data/islv/"

    if metric == "ind-lls":
        # inductive
        cmd += " --min-inst-slice  0.1 0.2 0.3 "
        cmd += " --factors 6 7 8 9 10 -v 0  --mspn  --inductive --burn-in 0 "

        cmd += " > results/islv/mspn/{mspn_type}/mspn.{mspn_type}.islv.{ds_name}.{metric}".format(
            mspn_type=args.mspn_leaves,
            ds_name=ds_name,
            metric=metric
        )
    else:
        # transductive
        cmd += " --min-inst-slice " + exp_id[0]
        miss_percs = exp_id[-2]
        cmd += " --factors 6 7 8 9 10 11 -v 0 "
        cmd += " --miss-percs " + miss_percs

        if metric == "mv-lls-d":
            cmd += "-o results/islv/mspn/lls-d/{miss_percs}/".format(miss_percs=miss_percs)

        cmd += " > results/islv/mspn/{mspn_type}/mspn.{mspn_type}.islv.{ds_name}.{miss_percs}.{metric}".format(
            mspn_type=args.mspn_leaves,
            ds_name=ds_name,
            miss_percs=miss_percs,
            metric=metric
        )
    return cmd


def get_date():
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


def get_script_transductive(path):
    lines = []
    seen = set()
    for filename in glob.iglob(path + '/**/learning_args.pkl', recursive=True):
        with open(filename, "rb") as pklfile:
            args = pickle.load(pklfile)

        # missing percs
        exp = (args.dataset, args.exp_id.split("/")[-2])

        if exp in seen:
            continue

        seen.add(exp)

        exp_path = args.output[:args.output.rfind("/")]

        # print(str(args).replace(",", "\n"))

        lines.append(get_command(args,
                                 path=exp_path,
                                 metric="mv-lls"))
        lines.append(get_command(args,
                                 path=exp_path,
                                 metric="mv-preds-scores"))
    return lines


def get_script_inductive(path):
    lines = []
    seen = set()
    for filename in glob.iglob(path + '/**/learning_args.pkl', recursive=True):
        with open(filename, "rb") as pklfile:
            args = pickle.load(pklfile)

        # missing percs
        exp = args.dataset

        if exp in seen:
            continue

        seen.add(exp)

        exp_path = args.output[:args.output.rfind("/")]

        # print(str(args).replace(",", "\n"))

        lines.append(get_command(args, path=exp_path, metric="ind-lls"))
    return lines


if __name__ == '__main__':
    #    print("""
    # ipython3 -- bin/collect_results.py results/mspn/20180510-133126/real-pos-count-cat/ --min-inst-slice 0.1 --threshold 0.3 0.5 0.7 --prior wider-prior-1 --leaf-type histogram --seeds 1111 2222 3333 4444 5555 --miss-percs 0.1 --metrics mv-lls --datasets abalonePP --mspn --data-path data/islv/ --factors 6 7 8 9 10 -v 0 > results/islv/mspn/hist/mspn.hist.islv.abalone.0.1.mv-lls
    # ipython3 -- bin/collect_results.py results/mspn/20180510-133126/real-pos-count-cat/ --min-inst-slice 0.1 --threshold 0.3 0.5 0.7 --prior wider-prior-1 --leaf-type histogram --seeds 1111 2222 3333 4444 5555 --miss-percs 0.1 --metrics mv-preds-scores --datasets abalonePP --mspn --data-path data/islv/ --factors 6 7 8 9 10 -v 0 > results/islv/mspn/hist/mspn.hist.islv.abalone.0.1.mv-preds
    # ipython3 -- bin/collect_results.py results/mspn/20180510-133126/real-pos-count-cat/ --min-inst-slice 0.1 --threshold 0.3 0.5 0.7 --prior wider-prior-1 --leaf-type histogram --seeds 1111 2222 3333 4444 5555 --miss-percs 0.1 --metrics mv-lls-d --datasets abalonePP --mspn --data-path data/islv/ --factors 6 7 8 9 10 -v 0 -o results/islv/mspn/lls-d/0.1/ > results/islv/mspn/hist/mspn.hist.islv.abalone.0.1.mv-lls-d
    # """)

    with open('scripts/collect_results_transductive_mspn.sh', 'w') as f:
        lines = []
        lines.extend(get_script_transductive("exp/islv-transductive/mspn-histogram/20180514-013405/real-pos-count-cat/"))
        lines.extend(get_script_transductive("exp/islv-transductive/mspn-piecewise/20180514-013405/real-pos-count-cat/"))
        f.writelines("\n".join(lines))

    with open('scripts/collect_results_inductive_mspn.sh', 'w') as f:
        lines = []
        lines.extend(get_script_inductive("exp/islv-inductive/mspn-histogram/20180515-001016/real-pos-count-cat/"))
        lines.extend(get_script_inductive("exp/islv-inductive/mspn-piecewise/20180515-001016/real-pos-count-cat/"))
        f.writelines("\n".join(lines))
