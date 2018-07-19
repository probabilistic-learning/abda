import argparse

try:
    from time import perf_counter
except:
    from time import time
    perf_counter = time

import datetime
import os
import sys
import logging
import pickle
import gzip
import json
import itertools

import glob

import matplotlib
matplotlib.use('Agg')
import numpy as np
import numba

from spn.structure.StatisticalTypes import MetaType


def path_collector(base_path, sub_dir_lists, file_name, exp_num=1):

    paths = []
    for s in itertools.product(*sub_dir_lists):
        s_strs = [str(s_dir) for s_dir in s]
        path = os.path.join(base_path, *s_strs, file_name)
        expanded_path = glob.glob(path)
        # assert len(expanded_path) == exp_num, 'p:{}\na:{}'.format(path, expanded_path)

        if len(expanded_path) < exp_num:
            logging.info('\t missing paths: {}'.format(path))
            if exp_num == 1:
                paths.append(None)
            else:
                paths.append(expanded_path + [None for j in range(len(expanded_path), exp_num)])
        elif exp_num == 1:
            paths.append(expanded_path[0])
        else:
            #
            # more than one path!
            paths.append(expanded_path)
    return paths


def load_numpy_arrays(paths):
    return [np.load(p) for p in paths]


def average_numpy_arrays(np_arrays):
    return [a.mean() for a in np_arrays]


def collect_gt_lls(paths):

    # paths = path_collector(dir,
    #                        sub_dir_lists,
    #                        file_name,
    #                        exp_num=exp_num)
    print(paths)

    lls = load_numpy_arrays(paths)
    return lls


# def collect_avg_lls(dir, sub_dir_lists, file_name, exp_num=1):
def collect_avg_lls(paths):

    # paths = path_collector(dir,
    #                        sub_dir_lists,
    #                        file_name,
    #                        exp_num=exp_num)
    print(paths)

    lls = load_numpy_arrays(paths)
    avg_lls = average_numpy_arrays(lls)
    return np.array(avg_lls)


def extract_spn_lls(path, burn_in, sample_key):

    samples = None
    with gzip.open(path, 'rb') as f:
        res = pickle.load(f)
        samples = res['samples']
    #
    # get only a certain portion
    samples = samples[burn_in:]
    lls = np.array([s[sample_key][:, 0] for s in samples
                    if sample_key in s])
    return lls


def extract_spn_w_dict(path,
                       burn_in,
                       w_key,
                       w_type_key):

    samples = None
    with gzip.open(path, 'rb') as f:
        res = pickle.load(f)
        samples = res['samples']
    #
    # get only a certain portion
    samples = samples[burn_in:]
    w_d_l = [s[w_key] for s in samples if w_key in s]
    w_type_d_l = [s[w_type_key] for s in samples if w_type_key in s]

    return w_d_l, w_type_d_l


def extract_mv_info(path,
                    mv_ll_key='mv-lls',
                    mv_pred_key='mv-preds',
                    mv_score_key='mv-preds-scores',
                    burn_in=4000,):

    samples = None
    with gzip.open(path, 'rb') as f:
        res = pickle.load(f)
        samples = res['samples']
    #
    # get only a certain portion
    samples = samples[burn_in:]
    print('after burn in', len(samples))
    #
    #
    mv_lls = [s[mv_ll_key] for s in samples if mv_ll_key in s]
    mv_preds = [s[mv_pred_key] for s in samples if mv_pred_key in s]
    mv_scores = [s[mv_score_key] for s in samples if mv_score_key in s]

    return mv_lls, mv_preds, mv_scores


def collect_mv_islv(paths,
                    mv_ll_key='mv-lls',
                    mv_pred_key='mv-preds',
                    mv_score_key='mv-preds-scores',
                    burn_in=4000,
                    exp_num=1,
                    # best_key='valid-lls'
                    ):
    mv_lls_list = []
    mv_preds_list = []
    mv_scores_list = []
    for p in paths:
        if p is not None:
            mv_lls, mv_preds, mv_scores = extract_mv_info(p,
                                                          mv_ll_key=mv_ll_key,
                                                          mv_pred_key=mv_pred_key,
                                                          mv_score_key=mv_score_key,
                                                          burn_in=burn_in,)
            mv_lls_list.append(mv_lls)
            mv_preds_list.append(mv_preds)
            mv_scores_list.append(mv_scores)

        else:
            mv_lls_list.append(None)
            mv_preds_list.append(None)
            mv_scores_list.append(None)

    print('BBB', len(mv_lls_list[0]))
    return mv_lls_list, mv_preds_list, mv_scores_list


def select_best_paths_by_lls(paths, burn_in=1000, best_key='valid-lls'):

    best_paths = []
    for grid_paths in paths:
        best_ll = -np.inf
        best_id = None
        for i, p in enumerate(grid_paths):
            if p is not None:
                lls = extract_spn_lls(p, burn_in=burn_in, sample_key=best_key)
                avg_ll = lls.mean()
                if avg_ll > best_ll:
                    best_ll = avg_ll
                    best_id = i
        if best_id is not None:
            print('***\t adding best path {}'.format(grid_paths[best_id]))
            best_paths.append(grid_paths[best_id])
        else:
            print('***\t no best path for configuration {}'.format(grid_paths))
            best_paths.append(None)
    return best_paths


# def collect_spn_avg_lls(dir, sub_dir_lists, file_name,
#                         sample_key='test-lls', burn_in=1000, exp_num=1,
#                         best_key='valid-lls'):
def collect_spn_avg_lls(paths,
                        sample_key='test-lls', burn_in=4000, exp_num=1,
                        best_key='valid-lls'):

    # paths = path_collector(dir,
    #                        sub_dir_lists,
    #                        file_name,
    #                        exp_num=exp_num)
    # print(paths)

    #
    # if each path is potentially a list of sub-paths for the grid
    # we have to select the best
    if exp_num > 1:
        print('more than one path, collecting results from a grid...')
        best_paths = select_best_paths_by_lls(paths, best_key=best_key)

    avg_lls = []
    for p in best_paths:
        # samples = None
        # with gzip.open(p, 'rb') as f:
        #     res = pickle.load(f)
        #     samples = res['samples']
        # #
        # # get only a certain portion
        # samples = samples[burn_in:]
        # lls = np.array([s[sample_key][:, 0] for s in samples
        #                 if sample_key in s])
        if p is not None:
            lls = extract_spn_lls(p, burn_in=burn_in, sample_key=sample_key)
            avg_lls.append(lls.mean(axis=1))
        else:
            avg_lls.append(None)

    # return np.array(avg_lls)
    return avg_lls


def aggr_weight_dicts(w_dict_list):
    n_ws = len(w_dict_list)
    n_features = len(w_dict_list[0].keys())

    w_acc = {i: {k: 0.0 for k, _v in w_dict_list[0][i].items()} for i in range(n_features)}

    for w_d in w_dict_list:
        assert len(w_d.keys()) == n_features
        for d in range(n_features):
            for t, t_w in w_d[d].items():
                w_acc[d][t] += (1 / n_ws * w_d[d][t])

    return w_acc


def weight_dict_list_to_sample_matrixes(w_dict_list, meta_types_list):

    cont_features = set()
    disc_features = set()

    # n_features = len(w_dict_list[0].keys())
    def get_label(t):
        try:
            return t.__name__
        except:
            return t.name

    for i, w_d in enumerate(w_dict_list):
        # if w_d is not None:
        for d in w_d.keys():
            for t, t_w in w_d[d].items():
                if meta_types_list[i][d] == MetaType.REAL:
                    cont_features.add(t)
                elif meta_types_list[i][d] == MetaType.DISCRETE:
                    disc_features.add(t)

    cont_features = list(sorted([get_label(c_f) for c_f in cont_features]))
    inv_cont_map = {f: i for i, f in enumerate(cont_features)}
    print('dealing with CONT features:', cont_features)
    disc_features = list(sorted([get_label(d_f) for d_f in disc_features]))
    inv_disc_map = {f: i for i, f in enumerate(disc_features)}
    print('dealing with DISC features:', disc_features)

    n_cont_features = len(cont_features)
    n_disc_features = len(disc_features)

    cont_w_samples = []
    disc_w_samples = []

    for i, w_d in enumerate(w_dict_list):
        # if w_d is not None:
        for d in w_d.keys():
            cont_sample = np.zeros(n_cont_features)
            disc_sample = np.zeros(n_disc_features)
            for t, t_w in w_d[d].items():
                if meta_types_list[i][d] == MetaType.REAL:
                    cont_sample[inv_cont_map[get_label(t)]] = t_w
                    cont_w_samples.append(cont_sample)
                elif meta_types_list[i][d] == MetaType.DISCRETE:
                    disc_sample[inv_disc_map[get_label(t)]] = t_w
                    disc_w_samples.append(disc_sample)

    return np.array(cont_w_samples), cont_features, np.array(disc_w_samples), disc_features


def weight_dict_list_to_sample_matrixes_mean(w_dict_list, meta_types_list):

    cont_features = set()
    disc_features = set()

    # n_features = len(w_dict_list[0].keys())
    def get_label(t):
        try:
            return t.__name__
        except:
            return t.name

    for i, w_d in enumerate(w_dict_list):
        # if w_d is not None:
        for d in w_d.keys():
            for t, t_w in w_d[d].items():
                if meta_types_list[i][d] == MetaType.REAL:
                    cont_features.add(t)
                elif meta_types_list[i][d] == MetaType.DISCRETE:
                    disc_features.add(t)

    cont_features = list(sorted([get_label(c_f) for c_f in cont_features]))
    inv_cont_map = {f: i for i, f in enumerate(cont_features)}
    print('dealing with CONT features:', cont_features)
    disc_features = list(sorted([get_label(d_f) for d_f in disc_features]))
    inv_disc_map = {f: i for i, f in enumerate(disc_features)}
    print('dealing with DISC features:', disc_features)

    n_cont_features = len(cont_features)
    n_disc_features = len(disc_features)

    cont_w_samples = []
    disc_w_samples = []

    for i, w_d in enumerate(w_dict_list):
        # if w_d is not None:
        cont_w_mean_list = []
        disc_w_mean_list = []
        for d in w_d.keys():
            cont_sample = np.zeros(n_cont_features)
            disc_sample = np.zeros(n_disc_features)
            for t, t_w in w_d[d].items():
                if meta_types_list[i][d] == MetaType.REAL:
                    cont_sample[inv_cont_map[get_label(t)]] = t_w
                    # cont_w_samples.append(cont_sample)
                    cont_w_mean_list.append(cont_sample)
                elif meta_types_list[i][d] == MetaType.DISCRETE:
                    disc_sample[inv_disc_map[get_label(t)]] = t_w
                    # disc_w_samples.append(disc_sample)
                    disc_w_mean_list.append(disc_sample)
        c_w_m = np.array(cont_w_mean_list)
        cont_m_sample = np.mean(c_w_m, axis=0)
        if cont_m_sample.ndim > 0:
            cont_w_samples.append(cont_m_sample)
        d_w_m = np.array(disc_w_mean_list)
        disc_m_sample = np.mean(d_w_m, axis=0)
        if disc_m_sample.ndim > 0:
            disc_w_samples.append(disc_m_sample)

    return np.array(cont_w_samples), cont_features, np.array(disc_w_samples), disc_features

# def collect_type_weights(dir, sub_dir_lists,
#                          file_name='data.stats',
#                          w_key='spn-W', w_type_key='spn-type-W',
#                          exp_num=1):


def collect_type_weights(paths,
                         w_key='spn-W', w_type_key='spn-type-W',
                         ):

    # paths = path_collector(dir,
    #                        sub_dir_lists,
    #                        file_name,
    #                        exp_num=exp_num)
    # print(paths)

    w_dict_list = []
    w_type_dict_list = []
    for p in paths:
        with open(p, 'rb') as f:
            stats = pickle.load(f)
            w_dict_list.append(stats[w_key])
            w_type_dict_list.append(stats[w_type_key])

    return w_dict_list, w_type_dict_list


# def collect_meta_types(dir, sub_dir_lists,
#                        file_name='data.stats',
#                        stats_key='meta-types',
#                        exp_num=1):
def collect_meta_types(paths,
                       stats_key='meta-types',
                       # exp_num=1
                       ):

    # paths = path_collector(dir,
    #                        sub_dir_lists,
    #                        file_name,
    #                        exp_num=exp_num)
    # print(paths)

    meta_type_list = []
    for p in paths:
        with open(p, 'rb') as f:
            stats = pickle.load(f)
            meta_type_list.append(stats[stats_key])

    return meta_type_list


# def collect_spn_type_weights(dir, sub_dir_lists, file_name,
#                              w_key='global-W', w_type_key='global-type-W',
#                              burn_in=1000, exp_num=None,
#                              best_key='valid-lls'):
def collect_spn_type_weights(paths,
                             w_key='global-W', w_type_key='global-type-W',
                             burn_in=4000, exp_num=None,
                             best_key='valid-lls'):

    # paths = path_collector(dir,
    #                        sub_dir_lists,
    #                        file_name,
    #                        exp_num=exp_num)
    # print(paths)

    #
    # if each path is potentially a list of sub-paths for the grid
    # we have to select the best
    if exp_num > 1:
        print('more than one path, collecting results from a grid...')
        best_paths = select_best_paths_by_lls(paths, best_key=best_key)

    w_dict_list = []
    w_type_dict_list = []
    for p in best_paths:

        if p is not None:
            w_d_l, w_type_d_l = extract_spn_w_dict(p,
                                                   burn_in=burn_in,
                                                   w_key=w_key,
                                                   w_type_key=w_type_key,)
            #
            # aggregate
            aggr_w_d = aggr_weight_dicts(w_d_l)
            aggr_w_type_d = aggr_weight_dicts(w_type_d_l)
            w_dict_list.append(aggr_w_d)
            w_type_dict_list.append(aggr_w_type_d)
        else:
            w_dict_list.append(None)
            w_type_dict_list.append(None)

    return w_dict_list, w_type_dict_list


if __name__ == '__main__':

    #########################################
    # creating the opt parser
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", type=str,
                        help='The main exp dir')

    parser.add_argument("--spn-dir", type=str,
                        default='exp/synth/hspn',
                        help='The main exp dir for spns')

    parser.add_argument('-o', '--output', type=str,
                        default='./exp/',
                        help='Output path to exp result')

    parser.add_argument('--tuple', type=int, nargs='+',
                        default=(10, 7),
                        help='A tuple of integers')

    parser.add_argument('-N', type=int, nargs='+',
                        default=[],
                        help='Sample sizes')

    parser.add_argument('-K', type=int, nargs='+',
                        default=[],
                        help='LVSTD lv sizes')

    parser.add_argument('-D', type=int, nargs='+',
                        default=[],
                        help='Feature sizes')

    parser.add_argument('--seed', type=int, nargs='+',
                        default=[],
                        help='Seeds adopted')

    parser.add_argument('--prior', type=str, nargs='+',
                        default=[],
                        help='priors to get')

    parser.add_argument('--omega-prior', type=str, nargs='+',
                        default=[],
                        help='omega priors to get')

    parser.add_argument('--threshold', type=str, nargs='+',
                        default=[],
                        help='col threshold to get')

    parser.add_argument('--min-inst-slice', type=str, nargs='+',
                        default=[],
                        help='m to get')

    parser.add_argument('--burn-in', type=int,
                        default=4000,
                        help='Burn in')

    parser.add_argument('--exp-id', type=str,
                        default=None,
                        help='Dataset output suffix')

    parser.add_argument('--miss-perc', type=float,
                        default=None,
                        help='Miss perc')

    parser.add_argument('--collect-weights-synth', action='store_true',
                        help='Collecting weights from synthetic exps')

    parser.add_argument('--collect-lls-synth', action='store_true',
                        help='Collecting lls from synthetic exps')

    parser.add_argument('--old-mode', action='store_true',
                        help='Collecting old exps, no miss-perc in path')

    parser.add_argument('--collect-mv-islv-lvstd', type=str, nargs='+',
                        default=None,
                        help='dataset name to collect missing value result for LVSTD')

    parser.add_argument('--collect-mv-islv-hspn', type=str, nargs='+',
                        default=None,
                        help='dataset name to collect missing value result for HSPN')

    parser.add_argument('-v', '--verbose', type=int, nargs='?',
                        default=1,
                        help='Verbosity level')
    #
    # parsing the args
    args = parser.parse_args()

    #
    # setting verbosity level
    if args.verbose == 1:
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    elif args.verbose == 2:
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    # #
    # # creating output dirs if they do not exist
    # date_string = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    out_path = os.path.join(args.output, args.exp_id)
    # else:
    #     out_path = os.path.join(args.output,  '{}_{}'.format(dataset_name, date_string))
    os.makedirs(out_path, exist_ok=True)

    args_out_path = os.path.join(out_path, 'args.json')
    json_args = json.dumps(vars(args))
    logging.info("Starting with arguments:\n%s\n\tdumped at %s", json_args, args_out_path)
    with open(args_out_path, 'w') as f:
        f.write(json_args)

    #
    # setting up the seed
    # rand_gen = np.random.RandomState(args.seed)

    ##################################
    #
    # retrieving type preds
    if args.collect_weights_synth:
        logging.info('\n\nCollecting type and param form weights for synth exp\n\n')
        synth_w_exp_path = os.path.join(out_path, 'w-synth-res')
        os.makedirs(synth_w_exp_path, exist_ok=True)

        gt_sub_dir_lists = [['*'], [args.exp_id], args.N, args.D, args.seed]
        gt_paths = path_collector(args.dir,
                                  gt_sub_dir_lists,
                                  file_name='data.stats',
                                  exp_num=1)
        gt_paths_map = {i: p for i, p in enumerate(gt_paths)}
        gt_paths_map_path = os.path.join(synth_w_exp_path, 'gt.path.map.pickle')

        with open(gt_paths_map_path, 'wb') as f:
            pickle.dump(gt_paths_map, f)
            logging.info('\t\tdumped gt path map to {}'.format(gt_paths_map_path))

        print('\nlooking for GT paths'.format(gt_paths))

        gt_meta_types = collect_meta_types(  # args.dir, gt_sub_dir_lists,
            # 'data.stats',
            gt_paths,
            'meta-types')
        #
        # saving meta type list
        gt_mt_path = os.path.join(synth_w_exp_path, 'gt.mt.pickle')
        with open(gt_mt_path, 'wb') as f:
            pickle.dump(gt_meta_types, f)
            logging.info('\t\tdumped gt meta types to {}'.format(gt_mt_path))

        gt_ws, gt_type_ws = collect_type_weights(gt_paths,
                                                 w_key='spn-W', w_type_key='spn-type-W')

        gt_ws_path = os.path.join(synth_w_exp_path, 'gt.ws.pickle')
        with open(gt_ws_path, 'wb') as f:
            pickle.dump((gt_ws, gt_type_ws), f)
            logging.info('\t\tdumped gt type weights to {}'.format(gt_ws_path))

        spn_sub_dir_lists = [['*'], [args.exp_id], args.N, args.D, args.seed,
                             ['*'], ['*'], ['*'],
                             ['*']]
        spn_paths = path_collector(args.spn_dir,
                                   spn_sub_dir_lists,
                                   file_name='result-dump.pklz',
                                   exp_num=2)
        spn_paths_map = {i: p for i, p in enumerate(spn_paths)}
        spn_paths_map_path = os.path.join(synth_w_exp_path, 'spn.path.map.pickle')

        with open(spn_paths_map_path, 'wb') as f:
            pickle.dump(spn_paths_map, f)
            logging.info('\t\tdumped spn path map to {}'.format(spn_paths_map_path))

        spn_ws, spn_type_ws = collect_spn_type_weights(spn_paths,
                                                       w_key='global-W', w_type_key='global-type-W',
                                                       burn_in=args.burn_in, exp_num=2,
                                                       best_key='valid-lls')

        spn_ws_path = os.path.join(synth_w_exp_path, 'spn.ws.pickle')
        with open(spn_ws_path, 'wb') as f:
            pickle.dump((spn_ws, spn_type_ws), f)
            logging.info('\t\tdumped spn type weights to {}'.format(spn_ws_path))

        assert len(gt_ws) == len(gt_type_ws)
        assert len(spn_ws) == len(spn_type_ws)
        assert len(gt_ws) == len(spn_ws)

    #
    #
    # ipython3 -- bin/collect_results_synth.py data/synth/ --spn-dir exp/synth/hspn/ -N 2000 5000 10000 -D 4 8 16 --seed 1111 1337 2222 3333 4444 5555 6666 7777 8888 9999 --collect-lls-synth -o trash --exp-id real-pos-count-cat
    #
    if args.collect_lls_synth:

        synth_lls_exp_path = os.path.join(out_path, 'lls-synth-res')
        os.makedirs(synth_lls_exp_path, exist_ok=True)

        test_ll_name = 'test.lls.npy'
        gt_sub_dir_lists = [['*'], [args.exp_id], args.N, args.D, args.seed, ['lls']]
        gt_paths = path_collector(args.dir,
                                  gt_sub_dir_lists,
                                  file_name=test_ll_name,
                                  exp_num=1)
        gt_paths_map = {i: p for i, p in enumerate(gt_paths)}
        gt_paths_map_path = os.path.join(synth_lls_exp_path, 'gt.path.map.pickle')

        with open(gt_paths_map_path, 'wb') as f:
            pickle.dump(gt_paths_map, f)
            logging.info('\t\tdumped gt path map to {}'.format(gt_paths_map_path))

        gt_test_lls = collect_gt_lls(gt_paths)
        avg_gt_test_lls = np.array([lls.mean() for lls in gt_test_lls])
        print('ALL AVG test LL', avg_gt_test_lls.mean(), avg_gt_test_lls.min(),
              avg_gt_test_lls.max())
        assert len(gt_test_lls) == len(gt_paths)

        gt_lls_path = os.path.join(synth_lls_exp_path, 'gt-test-lls')
        with open(gt_lls_path, 'wb') as f:
            pickle.dump(gt_test_lls, f)
            logging.info('\t\tdumped gt test lls to {}'.format(gt_lls_path))

        # for n in args.N:
        #     sub_dir_lists = [['*'],  [args.exp_id], [n], args.D, args.seed, ['lls']]
        #     avg_test_lls = collect_avg_lls(args.dir, sub_dir_lists, test_ll_name)
        #     print('N: {} AVG test LL'.format(n), avg_test_lls.mean(),
        #           avg_test_lls.min(), avg_test_lls.max())

        # for d in args.D:
        #     sub_dir_lists = [['*'], [args.exp_id], args.N, [d], args.seed, ['lls']]
        #     avg_test_lls = collect_avg_lls(args.dir, sub_dir_lists, test_ll_name)
        #     print('D: {} AVG test LL'.format(d), avg_test_lls.mean(),
        #           avg_test_lls.min(), avg_test_lls.max())

        #########################
        #
        # error bar scatter plot

        # #
        # # GT
        # sub_dir_lists = [['*'], [args.exp_id], args.N, args.D, args.seed, ['lls']]
        # gt_test_lls = collect_avg_lls(args.dir, sub_dir_lists, test_ll_name)
        # avg_gt_test_lls = gt_test_lls  # .mean(axis=1)
        # print('GT AVG test LL', avg_gt_test_lls.mean(), avg_gt_test_lls.min(),
        #       avg_gt_test_lls.max())

        # gt_lls_path = os.path.join(out_path, 'gt-test-lls')
        # np.save(gt_lls_path, avg_gt_test_lls)

        #
        # spn
        # sub_dir_lists = [['*'], [args.exp_id], args.N, args.D, args.seed,
        #                  args.min_inst_slice, args.threshold, args.prior,
        #                  args.omega_prior]
        spn_sub_dir_lists = [['*'], [args.exp_id], args.N, args.D, args.seed,
                             ['*'], ['*'], ['*'],
                             ['*']]
        spn_paths = path_collector(args.spn_dir,
                                   spn_sub_dir_lists,
                                   file_name='result-dump.pklz',
                                   exp_num=2)
        spn_paths_map = {i: p for i, p in enumerate(spn_paths)}
        spn_paths_map_path = os.path.join(synth_lls_exp_path, 'spn.path.map.pickle')

        with open(spn_paths_map_path, 'wb') as f:
            pickle.dump(spn_paths_map, f)
            logging.info('\t\tdumped spn path map to {}'.format(spn_paths_map_path))

        spn_test_lls = collect_spn_avg_lls(spn_paths,
                                           sample_key='test-lls', burn_in=args.burn_in, exp_num=2,
                                           best_key='valid-lls')
        # avg_spn_test_lls = np.array([lls.mean() for lls in spn_test_lls])
        # print('SPN AVG test LL', avg_spn_test_lls.mean(), avg_spn_test_lls.min(),
        #       avg_spn_test_lls.max())
        assert len(spn_test_lls) == len(spn_paths)

        spn_lls_path = os.path.join(synth_lls_exp_path, 'spn-test-lls')
        with open(spn_lls_path, 'wb') as f:
            pickle.dump(spn_test_lls, f)
            logging.info('\t\tdumped spn test lls to {}'.format(spn_lls_path))

        # min_spn_lls = spn_test_lls.min(axis=1)
        # max_spn_lls = spn_test_lls.max(axis=1)
        # avg_spn_yerr = abs(max_spn_lls - min_spn_lls) / 2
        # print('SPN AVG test LL', avg_spn_test_lls, min_spn_lls, max_spn_lls)
        # spn_err_path = os.path.join(out_path, 'spn-test-errs')
        # np.save(spn_err_path, avg_spn_yerr)

        # import matplotlib.pyplot as plt
        # from matplotlib.backends.backend_pdf import PdfPages

        # fig, ax = plt.subplots(figsize=(8, 8))
        # ax.errorbar(avg_gt_test_lls, avg_spn_test_lls,
        #             yerr=avg_spn_yerr, fmt='o')
        # ll_min = min(avg_gt_test_lls.min(), avg_spn_test_lls.min())
        # ll_max = max(avg_gt_test_lls.max(), avg_spn_test_lls.max())
        # eps = .5
        # print((ll_min - eps, ll_max + eps))
        # ax.plot([ll_min - eps, ll_max + eps], [ll_min - eps, ll_max + eps], "r--")
        # plt.title("Simplest errorbars, 0.2 in x, 0.4 in y")
        # # plt.show()
        # scatter_path = os.path.join(out_path, 'synth-scatter.pdf')
        # pp = PdfPages(scatter_path)
        # pp.savefig(fig, bbox_inches='tight')
        # pp.close()

    #
    # Missing values
    #
    # ipython3 -- bin/collect_results_synth.py exp/islv/lvstd/20180503-120524/  -K 2 5 10 --collect-mv-islv-lvstd germanPP -o trash --miss-perc 0.1 --exp-id islv-lvstd
    if args.collect_mv_islv_lvstd:

        logging.info('\n\nLooking fro MVs for LVSTD for datasets:{} (miss perc {})'.format(
            args.collect_mv_islv_lvstd, args.miss_perc))

        islv_mv_exp_path = os.path.join(out_path, 'islv-lvstd')
        for dataset in args.collect_mv_islv_lvstd:

            logging.info('\n\tconsidering {}'.format(dataset))
            islv_d_mv_exp_path = os.path.join(islv_mv_exp_path, dataset, str(args.miss_perc))
            os.makedirs(islv_d_mv_exp_path, exist_ok=True)
            # lvstd_sub_dir_lists = [[dataset], args.K]

            #
            # FIXME: add miss perc in the path as well
            #
            lvstd_sub_dir_lists = None
            if args.old_mode:
                lvstd_sub_dir_lists = [[dataset], args.K]
            else:
                lvstd_sub_dir_lists = [[dataset], args.K, [args.miss_perc]]
            lvstd_paths = path_collector(args.dir,
                                         lvstd_sub_dir_lists,
                                         file_name='result-dump.pklz',
                                         exp_num=1)
            lvstd_paths_map = {i: p for i, p in enumerate(lvstd_paths)}
            lvstd_paths_map_path = os.path.join(
                islv_d_mv_exp_path, 'lvstd-{}.path.map.pickle'.format(dataset))

            with open(lvstd_paths_map_path, 'wb') as f:
                pickle.dump(lvstd_paths_map, f)
                logging.info('\t\tdumped lvstd path map to {}'.format(lvstd_paths_map_path))

            lvstd_mv_lls, lvstd_mv_preds, lvstd_mv_scores = \
                collect_mv_islv(lvstd_paths,
                                mv_ll_key='mv-lls',
                                mv_pred_key='mv-preds',
                                mv_score_key='mv-preds-scores',
                                burn_in=args.burn_in, exp_num=1,
                                # best_key='valid-lls'
                                )
            # avg_spn_test_lls = np.array([lls.mean() for lls in spn_test_lls])
            # print('SPN AVG test LL', avg_spn_test_lls.mean(), avg_spn_test_lls.min(),
            #       avg_spn_test_lls.max())
            assert len(lvstd_mv_lls) == len(lvstd_paths)
            assert len(lvstd_mv_preds) == len(lvstd_paths)
            assert len(lvstd_mv_scores) == len(lvstd_paths)

            lvstd_mv_path = os.path.join(islv_d_mv_exp_path, 'lvstd-{}-{}-mv-lls.pickle'.format(dataset,
                                                                                                args.miss_perc))
            with open(lvstd_mv_path, 'wb') as f:
                pickle.dump(lvstd_mv_lls, f)
                logging.info('\t\tdumped lvstd mv lls for dataset {} {} to {}'.format(dataset,
                                                                                      args.miss_perc,
                                                                                      lvstd_mv_path))

            lvstd_mv_path = os.path.join(islv_d_mv_exp_path, 'lvstd-{}-{}-mv-preds.pickle'.format(dataset,
                                                                                                  args.miss_perc))
            with open(lvstd_mv_path, 'wb') as f:
                pickle.dump(lvstd_mv_preds,                             f)
                logging.info('\t\tdumped lvstd mv preds for dataset {} {} to {}'.format(dataset,
                                                                                        args.miss_perc,
                                                                                        lvstd_mv_path))

            lvstd_mv_path = os.path.join(islv_d_mv_exp_path, 'lvstd-{}-{}-mv-preds-scores.pickle'.format(dataset,
                                                                                                         args.miss_perc))
            with open(lvstd_mv_path, 'wb') as f:
                pickle.dump(lvstd_mv_scores, f)
                logging.info('\t\tdumped lvstd mv for dataset {} {} to {}'.format(dataset,
                                                                                  args.miss_perc,
                                                                                  lvstd_mv_path))

    #
    # HSPN
    # ipython3 -- bin/collect_results_synth.py None --spn-dir exp/islv/hspn/20180504-202223/real-pos-count-cat/  --min-inst-slice 0.1 0.2 0.3 --threshold 0.3 0.5 --prior spicky-prior-1 wider-prior-1 --omega-prior uniform  --collect-mv-islv-hspn germanPP -o trash --miss-perc 0.1 --exp-id islv-hspn
    if args.collect_mv_islv_hspn:

        logging.info('\n\nLooking fro MVs for HSPN for datasets:{} (miss perc {})'.format(
            args.collect_mv_islv_hspn, args.miss_perc))

        hspn_islv_mv_exp_path = os.path.join(out_path, 'islv-hspn')
        for dataset in args.collect_mv_islv_hspn:

            logging.info('\n\tconsidering {}'.format(dataset))
            hspn_islv_d_mv_exp_path = os.path.join(hspn_islv_mv_exp_path,
                                                   dataset, str(args.miss_perc))
            os.makedirs(hspn_islv_d_mv_exp_path, exist_ok=True)
            # lvstd_sub_dir_lists = [[dataset], args.K]

            #
            # FIXME: add miss perc in the path as well
            #
            hspn_sub_dir_lists = [[dataset], args.min_inst_slice, args.threshold,
                                  args.prior, args.omega_prior, [args.miss_perc]]
            hspn_paths = path_collector(args.spn_dir,
                                        hspn_sub_dir_lists,
                                        file_name='result-dump.pklz',
                                        exp_num=1)
            hspn_paths_map = {i: p for i, p in enumerate(hspn_paths)}
            hspn_paths_map_path = os.path.join(
                hspn_islv_d_mv_exp_path, 'hspn-{}.path.map.pickle'.format(dataset))

            with open(hspn_paths_map_path, 'wb') as f:
                pickle.dump(hspn_paths_map, f)
                logging.info('\t\tdumped hspn path map to {}'.format(hspn_paths_map_path))

            hspn_mv_lls, hspn_mv_preds, hspn_mv_scores = \
                collect_mv_islv(hspn_paths,
                                mv_ll_key='mv-lls',
                                mv_pred_key='mv-preds',
                                mv_score_key='mv-preds-scores',
                                burn_in=args.burn_in, exp_num=1,
                                # best_key='valid-lls'
                                )

            assert len(hspn_mv_lls) == len(hspn_paths)
            assert len(hspn_mv_lls) == len(hspn_paths)

            hspn_mv_path = os.path.join(hspn_islv_d_mv_exp_path, 'hspn-{}-{}-mv-lls.pickle'.format(dataset,
                                                                                                   args.miss_perc))
            with open(hspn_mv_path, 'wb') as f:
                pickle.dump(hspn_mv_lls, f)
                logging.info('\t\tdumped hspn mv lls for dataset {} {} to {}'.format(dataset,
                                                                                     args.miss_perc,
                                                                                     hspn_mv_path))

            hspn_mv_path = os.path.join(hspn_islv_d_mv_exp_path, 'hspn-{}-{}-mv-preds.pickle'.format(dataset,
                                                                                                     args.miss_perc))
            with open(hspn_mv_path, 'wb') as f:
                pickle.dump(hspn_mv_preds, f)
                logging.info('\t\tdumped hspn mv preds for dataset {} {} to {}'.format(dataset,
                                                                                       args.miss_perc,
                                                                                       hspn_mv_path))

            hspn_mv_path = os.path.join(hspn_islv_d_mv_exp_path, 'hspn-{}-{}-mv-preds-scores.pickle'.format(dataset,
                                                                                                            args.miss_perc))
            with open(hspn_mv_path, 'wb') as f:
                pickle.dump(hspn_mv_scores, f)
                logging.info('\t\tdumped hspn mv preds scores for dataset {} {} to {}'.format(dataset,
                                                                                              args.miss_perc,
                                                                                              hspn_mv_path))
