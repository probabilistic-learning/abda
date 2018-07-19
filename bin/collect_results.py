import argparse

try:
    from time import perf_counter
except:
    from time import time
    perf_counter = time
from collections import defaultdict
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
from numpy.testing import assert_array_equal
import scipy.io
import numba

from explain import print_perf_dict, compute_perf_dict_miss
from spn.structure.StatisticalTypes import MetaType
from bin.spstd_model_ha1 import compute_predictions_dict_miss, load_islv_mat


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


def weight_dicts_to_matrix(w_dict_list, features=['MSE', 'MAE', 'RMSE', 'M-RMSE', 'R-RMSE', 'S-RMSE']):

    n_ws = len(w_dict_list)
    n_features = len(w_dict_list[0].keys())
    feature_set = set(features)
    # ord_types = {t: i for i, t in enumerate(w_dict_list[0][0].keys())}
    # inv_ord_types = {i: t for i, t in enumerate(w_dict_list[0][0].keys())}
    ord_types = {t: i for i, t in enumerate(features)}
    inv_ord_types = {i: t for i, t in enumerate(features)}
    print('OT', ord_types)
    print('IT', inv_ord_types)
    n_dims = len(ord_types)
    w_acc = np.zeros((n_ws, n_features, n_dims))
    w_acc[:] = np.nan

    for s, w_d in enumerate(w_dict_list):
        assert len(w_d.keys()) == n_features
        for d in range(n_features):
            for t, t_w in w_d[d].items():
                if t in feature_set:
                    w_acc[s, d, ord_types[t]] = t_w

    assert np.isnan(w_acc).sum() == 0

    return w_acc, ord_types, inv_ord_types


def std_weight_dicts(w_dict_list, features=['MSE', 'MAE', 'RMSE', 'M-RMSE', 'R-RMSE', 'S-RMSE']):

    n_ws = len(w_dict_list)
    n_features = len(w_dict_list[0].keys())
    feature_set = set(features)
    # ord_types = {t: i for i, t in enumerate(w_dict_list[0][0].keys())}
    # inv_ord_types = {i: t for i, t in enumerate(w_dict_list[0][0].keys())}
    ord_types = {t: i for i, t in enumerate(features)}
    inv_ord_types = {i: t for i, t in enumerate(features)}
    print('OT', ord_types)
    print('IT', inv_ord_types)
    n_dims = len(ord_types)
    w_acc = np.zeros((n_ws, n_features, n_dims))

    for s, w_d in enumerate(w_dict_list):
        assert len(w_d.keys()) == n_features
        for d in range(n_features):
            for t, t_w in w_d[d].items():
                if t in feature_set:
                    w_acc[s, d, ord_types[t]] = t_w

    w_stds = w_acc.std(axis=0)

    aggr_w = defaultdict(dict)
    for d in range(n_features):
        for t in range(n_dims):
            aggr_w[d][inv_ord_types[t]] = w_stds[d, t]

    return aggr_w


def aggr_mv_score_dict(score_dict, m, ):
    aggr_measure = 0
    for d, m_score_map in score_dict.items():
        aggr_measure += m_score_map[m]
    D = len(score_dict)
    return aggr_measure / D


def aggr_mv_score_dict_meta_types(score_dict, m_cont, m_disc, meta_types):

    cont_aggr_measure = 0
    disc_aggr_measure = 0

    for d, m_score_map in score_dict.items():
        if meta_types[d] == MetaType.REAL:
            cont_aggr_measure += m_score_map[m_cont]
        elif meta_types[d] == MetaType.DISCRETE:
            disc_aggr_measure += m_score_map[m_disc]

    D_cont = len([d for d in score_dict.keys() if meta_types[d] == MetaType.REAL])
    D_disc = len([d for d in score_dict.keys() if meta_types[d] == MetaType.DISCRETE])

    c_c = cont_aggr_measure / D_cont if D_cont else np.inf
    c_d = disc_aggr_measure / D_disc if D_disc else np.inf
    return c_c, c_d


def perf_dict_to_array(perf_dict, m):
    D = len(perf_dict)
    perf_array = np.zeros(D)
    for d in range(D):
        perf_array[d] = perf_dict[d][m]

    return perf_array


def dump_perf_d(perf_dict, output_path, m):
    perf_array = perf_dict_to_array(perf_dict, m)
    np.save(output_path, perf_array)


def load_mask(path):
    m = None
    with open(path, 'rb') as f:
        m = pickle.load(f)
    return m


def test_mask(mask, mask_val):
    assert mask.shape == mask_val.shape

    N, D = mask.shape
    mask_test = np.zeros((N, D), dtype=bool)

    for n in range(N):
        for d in range(D):
            if mask[n, d] and not mask_val[n, d]:
                mask_test[n, d] = True

    assert_array_equal(mask_val & mask_test, np.zeros((N, D), dtype=bool))
    assert_array_equal(mask_val | mask_test, mask)
    return mask_test


def linearize_mask(X, mask, mask_val):
    assert mask.shape == mask_val.shape

    N, D = mask.shape
    assert X.shape == mask.shape

    ext_mask = []
    for n in range(N):
        for d in range(D):
            if mask[n, d]:
                if not np.isnan(X[n, d]):
                    if mask_val[n, d]:
                        ext_mask.append(True)
                    else:
                        ext_mask.append(False)

    ext_mask = np.array(ext_mask)
    # print((~np.isnan(X) & mask).sum())
    # print(len(ext_mask))
    assert len(ext_mask) == (~np.isnan(X) & mask).sum()
    assert ext_mask.sum() == (~np.isnan(X) & mask & mask_val).sum()

    return np.array(ext_mask, dtype=bool)


def linearize_mask_mf_preds(X, mask, mask_val):
    assert mask.shape == mask_val.shape

    N, D = mask.shape
    assert X.shape == mask.shape

    ext_mask = []
    for n in range(N):
        for d in range(D):
            if np.isnan(X[n, d]) or mask[n, d]:

                if mask_val[n, d]:
                    ext_mask.append(True)
                else:
                    ext_mask.append(False)

    ext_mask = np.array(ext_mask)
    # print((~np.isnan(X) & mask).sum())
    # print(len(ext_mask))
    # assert len(ext_mask) == (~np.isnan(X) & mask).sum()
    # assert ext_mask.sum() == (~np.isnan(X) & mask & mask_val).sum()

    return np.array(ext_mask, dtype=bool)


def split_path(path, factors):
    path = os.path.normpath(path)
    split_path = path.split(os.sep)
    return np.array(split_path)[factors]


def path_collector(base_path, sub_dir_lists, file_name, exp_num=1):

    paths = []
    for s in itertools.product(*sub_dir_lists):
        s_strs = [str(s_dir) for s_dir in s]
        # print('SRSDSDS', s_strs)
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


def compute_aggr_mask(par_list, aggr_list):

    configs = list(itertools.product(*par_list))
    aggr_masks = [j for j, _s in enumerate(configs) for i in range(len(aggr_list))]
    print('AGGR MASKS', aggr_masks)
    return configs, aggr_masks


def extract_mv_info(path,
                    m_key='mv-lls',
                    count_key='id',
                    aggr_func=None,
                    burn_in=0,):

    # samples = None
    with gzip.open(path, 'rb') as f:
        res = pickle.load(f)

    m_samples = res[m_key]
    m_counts = res[count_key]
    logging.info('before burn in {} {}'.format(len(m_samples), len(m_counts)))
    tot_samples = len(m_samples)
    #
    # get only a certain portion
    m_samples = m_samples[burn_in:]
    m_counts = m_counts[burn_in:]
    logging.info('after burn in {} {}'.format(len(m_samples), len(m_counts)))
    assert len(m_samples) == len(m_counts)
    assert len(m_samples) + burn_in == tot_samples

    # if aggr_func:
    #     m_samples = [aggr_func(m) for m in m_samples]

    return m_counts, m_samples


def extract_best_sample_info(path,
                             m_key=None,
                             count_key='id'):

    # samples = None
    with gzip.open(path, 'rb') as f:
        best_sample = pickle.load(f)

    perfs = best_sample[m_key]
    count = best_sample[count_key]

    return count, perfs


AGGR_FUNCS_MAP = {'mv-lls': np.mean,
                  'mv-preds-lls': np.mean,
                  'mv-preds-scores': None}


def collect_best_sample_islv(paths,
                             m_key=None,
                             count_key='id',
                             ):
    m_list = []
    c_list = []
    for p in paths:
        if p is not None:
            m_c, m_i = extract_best_sample_info(p,
                                                m_key=m_key,
                                                count_key=count_key)
            m_list.append([m_i])
            c_list.append([m_c])

        else:
            m_list.append(None)
            c_list.append(None)

    assert len(c_list) == len(m_list)
    print('BBB', len(m_list[0]))
    return c_list, m_list


def collect_mv_islv(paths,
                    m_key='mv-lls',
                    count_key='id',
                    burn_in=0,
                    # best_key='valid-lls'
                    ):
    m_list = []
    c_list = []
    for p in paths:
        if p is not None:
            m_c, m_i = extract_mv_info(p,
                                       m_key=m_key,
                                       count_key=count_key,
                                       burn_in=burn_in,
                                       # aggr_func=AGGR_FUNCS_MAP[m_key]
                                       )
            m_list.append(m_i)
            c_list.append(m_c)

        else:
            m_list.append(None)
            c_list.append(None)

    assert len(c_list) == len(m_list)
    print('BBB', len(m_list[0]), len(m_list))
    return c_list, m_list


STATS_MAP = {'mean': np.mean,
             'min': np.min,
             'max': np.max,
             'std': np.std}


def compute_retrieve_stats(m_counts, m_list,
                           mv_valid_array_list,
                           mv_test_array_list,
                           aggr_func=np.mean,
                           stats=STATS_MAP):

    stats_list = []
    for _counts, perfs, mv_val, mv_test in zip(m_counts, m_list,
                                               mv_valid_array_list,
                                               mv_test_array_list):

        if perfs is not None:
            logging.info('Length of samples {}'.format(len(perfs)))
            aggr_perfs = defaultdict(dict)
            stat_res = defaultdict(list)
            for perf in perfs:
                # if len(mv_val) != len(perf):
                #     logging.info('mismatch')
                #     s_val_m = np.nan
                # else:
                s_val_m = aggr_func(perf[mv_val])
                s_test_m = aggr_func(perf[mv_test])
                s_full_m = aggr_func(perf)
                stat_res['valid'].append(s_val_m)
                stat_res['test'].append(s_test_m)
                stat_res['full'].append(s_full_m)
            for stat_name, stat_fn in stats.items():
                aggr_perfs['valid'][stat_name] = stat_fn(stat_res['valid'])
                aggr_perfs['full'][stat_name] = stat_fn(stat_res['full'])
                aggr_perfs['test'][stat_name] = stat_fn(stat_res['test'])

            stats_list.append(aggr_perfs)
        else:
            stats_list.append(None)

    return stats_list


def compute_retrieve_stats_inductive(m_counts, m_list,
                                     aggr_func=np.mean,
                                     stats=STATS_MAP):

    stats_list = []
    for _counts, perfs in zip(m_counts, m_list):

        logging.info('Length of samples {}'.format(len(perfs)))
        aggr_perfs = {}
        stat_res = []
        for perf in perfs:
            # if len(mv_val) != len(perf):
            #     logging.info('mismatch')
            #     s_val_m = np.nan
            # else:
            s_m = aggr_func(perf)
            stat_res.append(s_m)
        for stat_name, stat_fn in stats.items():
            aggr_perfs[stat_name] = stat_fn(stat_res)

        stats_list.append(aggr_perfs)

    return stats_list


def compute_retrieve_best_stats_inductive(m_counts, m_list,
                                          stats=STATS_MAP):

    stats_list = []
    for _counts, perfs in zip(m_counts, m_list):

        logging.info('Length of samples {}'.format(len(perfs)))
        aggr_perfs = {}
        assert len(perfs) == 1, len(perfs)
        for stat_name, stat_fn in stats.items():
            aggr_perfs[stat_name] = stat_fn(perfs[0])

        stats_list.append(aggr_perfs)

    return stats_list


def retrieve_mapping(X, X_miss):

    N, D = X.shape
    assert X_miss.shape == X.shape

    mapping_d = []
    for n in range(N):
        for d in range(D):
            if not np.isnan(X[n, d]):
                if X_miss[n, d]:
                    mapping_d.append(d)

    return np.array(mapping_d)


def mean_d(lls, linear_mask, mapping_d):

    D = np.max(mapping_d) + 1
    logging.debug('{}'.format(lls.shape))
    # assert lls.shape[1] == 1, lls.shape
    aggr_lls = np.zeros(D)
    aggr_count = np.zeros(D, dtype=int)

    for i in range(lls.shape[0]):
        if linear_mask[i]:
            d = mapping_d[i]
            aggr_lls[d] += lls[i]
            aggr_count[d] += 1

    aggr_lls /= aggr_count

    return aggr_lls


def compute_retrieve_stats_d(m_counts, m_list,
                             mv_valid_array_list,
                             mv_test_array_list,
                             mapping_d_list):

    stats_list = []
    for _counts, perfs, mv_val, mv_test, mapping_d in zip(m_counts, m_list,
                                                          mv_valid_array_list,
                                                          mv_test_array_list,
                                                          mapping_d_list):

        D = np.max(mapping_d) + 1
        logging.info('Length of samples {}'.format(len(perfs)))
        aggr_perfs = {}
        stat_res = defaultdict(list)
        for perf in perfs:
            # if len(mv_val) != len(perf):
            #     logging.info('mismatch')
            #     s_val_m = np.nan
            # else:
            s_val_m = mean_d(perf, mv_val, mapping_d)
            s_test_m = mean_d(perf, mv_test, mapping_d)
            s_full_m = mean_d(perf, np.ones(perf.shape, dtype=bool), mapping_d)
            stat_res['valid'].append(s_val_m)
            stat_res['test'].append(s_test_m)
            stat_res['full'].append(s_full_m)

        aggr_perfs['valid'] = np.mean(stat_res['valid'], axis=0)
        aggr_perfs['full'] = np.mean(stat_res['full'], axis=0)
        aggr_perfs['test'] = np.mean(stat_res['test'], axis=0)

        assert len(aggr_perfs['valid']) == D
        assert len(aggr_perfs['full']) == D
        assert len(aggr_perfs['test']) == D

        stats_list.append(aggr_perfs)

    return stats_list


def compute_retrieve_aggr_stats_d(m_counts, m_list,
                                  aggr_ids,
                                  mv_valid_array_list,
                                  mv_test_array_list,
                                  mapping_d_list):

    unique_configs = np.unique(aggr_ids)
    stats_list = defaultdict(dict)
    stat_res = defaultdict(lambda: defaultdict(list))
    D = None
    for _counts, perfs, i, mv_val, mv_test, mapping_d in zip(m_counts, m_list, aggr_ids,
                                                             mv_valid_array_list,
                                                             mv_test_array_list,
                                                             mapping_d_list):
        D = np.max(mapping_d) + 1
        logging.info('Length of samples {} {}'.format(len(perfs), i))
        # stat_res = defaultdict(list)
        for perf in perfs:
            s_val_m = mean_d(perf, mv_val, mapping_d)
            s_test_m = mean_d(perf, mv_test, mapping_d)
            s_full_m = mean_d(perf, np.ones(perf.shape, dtype=bool), mapping_d)
            stat_res[i]['valid'].append(s_val_m)
            stat_res[i]['test'].append(s_test_m)
            stat_res[i]['full'].append(s_full_m)

    aggr_perfs = [{} for i in unique_configs]
    aggr_stds = [{} for i in unique_configs]
    for i in unique_configs:
        # stats_list[i]['valid'] = aggr_func(stat_res[i]['valid'])
        # stats_list[i]['test'] = aggr_func(stat_res[i]['test'])
        # stats_list[i]['full'] = aggr_func(stat_res[i]['full'])

        # print('PP', i, stats_list[i]['valid'], stats_list[i]['full'], stats_list[i]['test'])

        stats_val = np.array(stat_res[i]['valid'])
        print(stats_val.shape)
        stats_full = np.array(stat_res[i]['full'])
        print(stats_full.shape)
        stats_test = np.array(stat_res[i]['test'])
        print(stats_test.shape)
        aggr_perfs[i]['valid'] = np.mean(stats_val, axis=0)
        aggr_perfs[i]['full'] = np.mean(stats_full, axis=0)
        aggr_perfs[i]['test'] = np.mean(stats_test, axis=0)

        aggr_stds[i]['valid'] = np.std(stats_val, axis=0)
        aggr_stds[i]['full'] = np.std(stats_full, axis=0)
        aggr_stds[i]['test'] = np.std(stats_test, axis=0)

        assert len(aggr_perfs[i]['valid']) == D
        assert len(aggr_perfs[i]['full']) == D
        assert len(aggr_perfs[i]['test']) == D

        assert len(aggr_stds[i]['valid']) == D
        assert len(aggr_stds[i]['full']) == D
        assert len(aggr_stds[i]['test']) == D

    return aggr_perfs, aggr_stds


def compute_retrieve_aggr_stats(m_counts, m_list,
                                aggr_ids,
                                mv_valid_array_list,
                                mv_test_array_list,
                                aggr_func=np.mean,
                                stats=STATS_MAP):

    unique_configs = np.unique(aggr_ids)
    stats_list = defaultdict(dict)
    stat_res = defaultdict(lambda: defaultdict(list))
    for _counts, perfs, i, mv_val, mv_test in zip(m_counts, m_list, aggr_ids,
                                                  mv_valid_array_list,
                                                  mv_test_array_list):
        if perfs is not None:
            logging.info('Length of samples {} {}'.format(len(perfs), i))
            # stat_res = defaultdict(list)
            for perf in perfs:
                s_val_m = aggr_func(perf[mv_val])
                s_test_m = aggr_func(perf[mv_test])
                s_full_m = aggr_func(perf)
                stat_res[i]['valid'].append(s_val_m)
                stat_res[i]['test'].append(s_test_m)
                stat_res[i]['full'].append(s_full_m)
        else:
            stat_res[i]['valid'].append(np.nan)
            stat_res[i]['test'].append(np.nan)
            stat_res[i]['full'].append(np.nan)

    aggr_perfs = [defaultdict(dict) for i in unique_configs]
    for i in unique_configs:
        # stats_list[i]['valid'] = aggr_func(stat_res[i]['valid'])
        # stats_list[i]['test'] = aggr_func(stat_res[i]['test'])
        # stats_list[i]['full'] = aggr_func(stat_res[i]['full'])

        # print('PP', i, stats_list[i]['valid'], stats_list[i]['full'], stats_list[i]['test'])
        for stat_name, stat_fn in stats.items():
            aggr_perfs[i]['valid'][stat_name] = stat_fn(stat_res[i]['valid'])
            aggr_perfs[i]['full'][stat_name] = stat_fn(stat_res[i]['full'])
            aggr_perfs[i]['test'][stat_name] = stat_fn(stat_res[i]['test'])

    return aggr_perfs


def compute_retrieve_aggr_stats_inductive(m_counts, m_list,
                                          aggr_ids,
                                          aggr_func=np.mean,
                                          stats=STATS_MAP):

    unique_configs = np.unique(aggr_ids)
    stat_res = defaultdict(list)
    for _counts, perfs, i, in zip(m_counts, m_list, aggr_ids):
        logging.info('Length of samples {} {}'.format(len(perfs), i))
        # stat_res = defaultdict(list)
        for perf in perfs:
            s_m = aggr_func(perf)
            stat_res[i].append(s_m)

    aggr_perfs = [{} for i in unique_configs]
    for i in unique_configs:

        for stat_name, stat_fn in stats.items():
            aggr_perfs[i][stat_name] = stat_fn(stat_res[i])

    return aggr_perfs


def print_m_stats(paths, m_stat_list, best_key='valid', stats_sets=['full', 'valid', 'test'], factors=None):

    for stats_map, p in zip(m_stat_list, paths):
        if factors is not None:
            if p is not None:
                f = split_path(p, factors)
                print('\t'.join(f))
            else:
                print('None')
        else:
            if p is not None:
                print('\t'.join(str(c) for c in p))
            else:
                print('None')
        if stats_map is not None:
            for split in stats_sets:
                print(split, '\t', '\t'.join(str(k) for k in stats_map[split].keys()))
                print('\t', '\t'.join('{:.5f}'.format(stats_map[split][k])
                                      for k in stats_map[split].keys()))
                # for stat_name, stat_val in stats_map[split].items():
                #     print(stat_name, stat_val)
        else:
            print('None')


def print_m_stats_inductive(paths, m_stat_list, best_m_stat_list, split_name,
                            factors=None):

    for stats_map, best_stats_map, p in zip(m_stat_list, best_m_stat_list, paths):
        if factors is not None:
            if p is not None:
                f = split_path(p, factors)
                print('\t'.join(f))
            else:
                print('None')
        else:
            if p is not None:
                print('\t'.join(str(c) for c in p))
            else:
                print('None')
        if stats_map is not None:
            if best_stats_map:
                print(split_name, '\t', '\t'.join(str(k) for k in stats_map.keys()), '\t',
                      '\t'.join('best-{}'.format(k) for k in best_stats_map.keys()))
                print('\t', '\t'.join('{:.5f}'.format(stats_map[k])
                                      for k in stats_map.keys()), '\t',
                      '\t'.join('{:.5f}'.format(best_stats_map[k])
                                for k in best_stats_map.keys()))
            else:
                print(split_name, '\t', '\t'.join(str(k) for k in stats_map.keys()))
                print('\t', '\t'.join('{:.5f}'.format(stats_map[k])
                                      for k in stats_map.keys()))


from bin.spstd_model_ha1 import sp_infer_data_types_ha1, retrieve_best_sample

if __name__ == '__main__':

    #########################################
    # creating the opt parser
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", type=str,
                        help='Paths to exp folders')

    parser.add_argument('-o', '--output', type=str,
                        default='./res/',
                        help='Output path to exp result')

    parser.add_argument('-N', type=int, nargs='+',
                        default=[],
                        help='Sample sizes')

    parser.add_argument('-K', type=int, nargs='+',
                        default=[],
                        help='LVSTD lv sizes')

    parser.add_argument('-D', type=int, nargs='+',
                        default=[],
                        help='Feature sizes')

    parser.add_argument('--seeds', type=int, nargs='+',
                        default=[],
                        help='Seeds adopted')

    parser.add_argument('--factors', type=int, nargs='+',
                        default=[],
                        help='Ids in path to parse hyperparameters')

    parser.add_argument('--prior', type=str, nargs='+',
                        default=[],
                        help='priors to get')

    parser.add_argument('--datasets', type=str, nargs='+',
                        default=None,
                        help='list of datasets to use')

    parser.add_argument('--data-path', type=str,
                        default='data/islv',
                        help='path to original dataset')

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
                        default=0,
                        help='Burn in')

    parser.add_argument('--exp-id', type=str,
                        default=None,
                        help='Dataset output suffix')

    parser.add_argument('--count-key', type=str,
                        default='id',
                        help='Key id for results')

    parser.add_argument('--miss-percs', type=float, nargs='+',
                        default=[],
                        help='Miss perc values')

    parser.add_argument('--leaf-type', type=str, nargs='+',
                        default=[],
                        help='Leaf type for MSPNs (histogram|piecewise)')

    parser.add_argument('--metrics', type=str, nargs='+',
                        default=['mv-lls', 'mv-preds', 'mv-preds-lls', 'mv-preds-scores'],
                        help='Metrics to extract from samples')

    parser.add_argument('--hspn', action='store_true',
                        help='HSPN exps')

    parser.add_argument('--mspn', action='store_true',
                        help='MSPN exps')

    parser.add_argument('--lvstd', action='store_true',
                        help='LVSTD exps')

    parser.add_argument('--inductive', action='store_true',
                        help=' exps')

    parser.add_argument('--fig-size', type=int, nargs='+',
                        default=(10, 7),
                        help='A tuple for the explanation fig size ')

    parser.add_argument('--show-plots', action='store_true',
                        help='Whether to show by screen the plots')

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

    #
    # creating output dirs if they do not exist
    date_string = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # if args.exp_id:
    #     out_path = os.path.join(args.output, args.exp_id)
    # else:
    #     out_path = os.path.join(args.output,  '{}_{}'.format(dataset_name, date_string))
    out_path = args.output
    os.makedirs(out_path, exist_ok=True)

    args_out_path = os.path.join(out_path, 'args.json')
    json_args = json.dumps(vars(args))
    logging.info("Starting with arguments:\n%s\n\tdumped at %s", json_args, args_out_path)
    with open(args_out_path, 'w') as f:
        f.write(json_args)

    #
    # setting up the seed
    rand_gen = np.random.RandomState(17)

    for dataset in args.datasets:

        print(dataset)

        if args.inductive:
            print('We are in the inductive setting')

            if args.hspn:
                #
                # composing experiments paths
                # exp/islv/hspn/20180508-155801/real-pos-count-cat/abalonePP/0.1/0.7/wider-prior-1/uniform/0.5/1111/

                islv_sub_dir_lists = [[dataset], args.min_inst_slice, args.threshold,
                                      args.prior, args.omega_prior]
                islv_ind_exp_path = os.path.join(out_path, 'ind-hspn-islv')

                # config_params, aggr_indexes = compute_aggr_mask([args.min_inst_slice, args.threshold,
                #                                                  args.prior, args.omega_prior,
                #                                                  args.miss_percs],
                #                                                 )

                model_name = 'hspn'
            elif args.mspn:

                islv_sub_dir_lists = [[dataset], args.min_inst_slice, args.threshold,
                                      args.prior, args.leaf_type]
                islv_ind_exp_path = os.path.join(out_path, 'ind-mspn-islv')

                # config_params, aggr_indexes = compute_aggr_mask([args.min_inst_slice, args.threshold,
                #                                                  args.prior, args.omega_prior,
                #                                                  args.miss_percs],
                #                                                 )

                model_name = 'mspn'

            for m in args.metrics:

                print(m)
                islv_d_ind_exp_path = os.path.join(islv_ind_exp_path, m)
                os.makedirs(islv_d_ind_exp_path, exist_ok=True)

                if m == 'ind-lls':

                    for split_name in ['train', 'valid', 'test']:

                        split_ll_paths = path_collector(args.dir,
                                                        islv_sub_dir_lists,
                                                        file_name='{}-lls.pklz'.format(split_name),
                                                        exp_num=1)
                        logging.info('\n\n{} paths for metric {}:\n\t\t{}'.format(split_name,
                                                                                  m, split_ll_paths))

                        split_islv_paths_map = {i: p for i, p in enumerate(split_ll_paths)}
                        islv_paths_map_path = os.path.join(islv_d_ind_exp_path,
                                                           'split-{}-{}-{}.path.map.pickle'.format(split_name,
                                                                                                   model_name,
                                                                                                   dataset, m))

                        split_m_counts, split_m_list = collect_mv_islv(split_ll_paths,
                                                                       m_key='{}-lls'.format(
                                                                           split_name),
                                                                       count_key=args.count_key, burn_in=args.burn_in)

                        assert len(split_m_counts) == len(split_m_list)

                        split_m_stat_list = compute_retrieve_stats_inductive(split_m_counts, split_m_list,
                                                                             stats=STATS_MAP)
                        # aggr_m_stat_list = compute_retrieve_aggr_stats_inductive(split_m_counts,
                        #                                                          split_m_list,
                        #                                                          aggr_indexes,
                        #                                                          stats=STATS_MAP)

                        # print_m_stats(config_params, aggr_m_stat_list,
                        #               best_key='valid')

                        best_split_m_stat_list = [None for _s in split_m_stat_list]

                        if args.hspn:
                            #
                            # load best sample
                            best_split_ll_paths = path_collector(args.dir,
                                                                 islv_sub_dir_lists,
                                                                 file_name='best-{}-lls.pklz'.format(
                                                                     split_name),
                                                                 exp_num=1)
                            assert len(best_split_ll_paths) == len(split_m_stat_list)
                            logging.info('\n\n Best {} paths for metric {}:\n\t\t{}'.format(split_name,
                                                                                            m, best_split_ll_paths))

                            best_split_m_counts, best_split_m_list = collect_best_sample_islv(best_split_ll_paths,
                                                                                              m_key='{}-lls'.format(
                                                                                                  split_name),
                                                                                              count_key=args.count_key)
                            best_split_m_stat_list = compute_retrieve_best_stats_inductive(best_split_m_counts,
                                                                                           best_split_m_list,
                                                                                           stats=STATS_MAP)

                        print_m_stats_inductive(split_ll_paths, split_m_stat_list,
                                                best_split_m_stat_list,
                                                split_name,
                                                factors=args.factors)

        else:
            print('We are in the transductive setting')

            #
            # retrieving the dataset and friends
            data_path = args.data_path
            dataset_path = os.path.join(data_path, dataset, '{}.mat'.format(dataset))

            if args.mspn:

                #
                # composing experiments paths
                # exp/islv/hspn/20180508-155801/real-pos-count-cat/abalonePP/0.1/0.7/wider-prior-1/uniform/0.5/1111/

                islv_sub_dir_lists = [[dataset], args.min_inst_slice, args.threshold,
                                      args.prior, args.leaf_type,
                                      args.miss_percs, args.seeds]
                islv_mv_exp_path = os.path.join(out_path, 'mv-mspn-islv')

                config_params, aggr_indexes = compute_aggr_mask([args.min_inst_slice, args.threshold,
                                                                 args.prior, args.leaf_type,
                                                                 args.miss_percs],
                                                                args.seeds)

                model_name = 'mspn'

                X, meta_types, domains = load_islv_mat(dataset_path)
            elif args.hspn:
                #
                # composing experiments paths
                # exp/islv/hspn/20180508-155801/real-pos-count-cat/abalonePP/0.1/0.7/wider-prior-1/uniform/0.5/1111/

                islv_sub_dir_lists = [[dataset], args.min_inst_slice, args.threshold,
                                      args.prior, args.omega_prior,
                                      args.miss_percs, args.seeds]
                islv_mv_exp_path = os.path.join(out_path, 'mv-hspn-islv')

                config_params, aggr_indexes = compute_aggr_mask([args.min_inst_slice, args.threshold,
                                                                 args.prior, args.omega_prior,
                                                                 args.miss_percs],
                                                                args.seeds)

                model_name = 'hspn'

                X, meta_types, domains = load_islv_mat(dataset_path)
            elif args.lvstd:

                #
                #
                islv_sub_dir_lists = [[dataset], args.K, args.miss_percs, args.seeds]
                islv_mv_exp_path = os.path.join(out_path, 'mv-lvstd-islv')

                config_params, aggr_indexes = compute_aggr_mask(
                    [args.K, args.miss_percs], args.seeds)

                model_name = 'lvstd'

                data_dict = scipy.io.loadmat(dataset_path)
                logging.info('Loaded mat dataset {}'.format(dataset_path))

                X = data_dict['X']
                X = X.astype(np.float64)
                logging.info('Loaded data with shape: {}'.format(X.shape))
                X_orig = np.copy(X)

                C = data_dict['T'].flatten().astype(np.int64)
                logging.info('\thMeta-types:{}'.format(C))

                #
                # shifting discrete
                for d in range(X.shape[1]):
                    if C[d] == 4:
                        X[:, d] -= 1

                #
                # loading meta types
                meta_types = None
                stats_map = None
                stats_path = os.path.join(data_path, dataset, 'data.stats')
                with open(stats_path, 'rb') as f:
                    stats_map = pickle.load(f)
                meta_types = stats_map['meta-types']

            else:
                raise ValueError('Unrecognized model')

            X_min = np.nanmin(X, axis=0)
            X_max = np.nanmax(X, axis=0)
            X_mean = np.nanmean(X, axis=0)
            # X_std = np.sqrt(np.nansum(X - X_mean, axis=0))
            X_std = np.nanstd(X, axis=0) / np.sqrt((~np.isnan(X)).sum(axis=0))
            print('X MIN', X_min)
            print('X MAX', X_max)
            print('X MEAN', X_mean)
            print('X STD', X_std)
            X_std = scipy.stats.sem(X, axis=0, nan_policy='omit')
            print('X STD', X_std)

            #
            # retrieving the masks
            mask_sub_dir_lists = [[dataset], ['miss'], args.miss_percs, args.seeds]
            miss_mask_paths = path_collector(data_path,
                                             mask_sub_dir_lists,
                                             file_name='miss.full.data',
                                             exp_num=1)
            logging.info('\n\npaths for masks {}:\n\t\t'.format(miss_mask_paths))
            miss_masks = [load_mask(p) for p in miss_mask_paths]
            logging.info('\tloaded {} missing value masks'.format(len(miss_masks)))

            miss_val_mask_paths = path_collector(data_path,
                                                 mask_sub_dir_lists,
                                                 file_name='miss.val.data',
                                                 exp_num=1)
            logging.info('\n\npaths for val masks {}:\n\t\t'.format(miss_val_mask_paths))
            miss_val_masks = [load_mask(p) for p in miss_val_mask_paths]
            logging.info('\tloaded {} missing value VAL masks'.format(len(miss_val_masks)))

            miss_test_masks = [test_mask(m, m_v) for m, m_v in zip(miss_masks,
                                                                   miss_val_masks)]
            logging.info('\tcomputed {} missing value TEST masks'.format(len(miss_test_masks)))

            #
            # multiplying masks for hyperparameters
            # miss_mask_list = [m for k in args.K for m in miss_masks]
            # # print('\n\nALL paths for masks {}:\n\t\t{}'.format(miss_mask_list))
            # miss_val_mask_list = [m for k in args.K for m in miss_val_masks]
            # # print('\n\nALL val paths for masks {}:\n\t\t{}'.format(miss_val_mask_list))
            # miss_test_mask_list = [m for k in args.K for m in miss_test_masks]

            miss_mask_list = [m for k in config_params for m in miss_masks]
            miss_val_mask_list = [m for k in config_params for m in miss_val_masks]
            miss_test_mask_list = [m for k in config_params for m in miss_test_masks]

            mapping_d_list = [retrieve_mapping(X, m) for m in miss_mask_list]

            assert len(miss_mask_list) == len(miss_val_mask_list)
            assert len(miss_mask_list) == len(miss_test_mask_list)

            for m in args.metrics:

                print(m)
                islv_d_mv_exp_path = os.path.join(islv_mv_exp_path, m)
                os.makedirs(islv_d_mv_exp_path, exist_ok=True)

                m_name = None
                if m == 'mv-lls-d':
                    m_name = 'mv-lls'
                else:
                    m_name = m

                islv_ll_paths = path_collector(args.dir,
                                               islv_sub_dir_lists,
                                               file_name='{}.pklz'.format(m_name),
                                               exp_num=1)
                logging.info('\n\npaths for metric {}:\n\t\t{}'.format(m, islv_ll_paths))
                islv_paths_map = {i: p for i, p in enumerate(islv_ll_paths)}
                islv_paths_map_path = os.path.join(islv_d_mv_exp_path,
                                                   '{}-{}-{}.path.map.pickle'.format(model_name,
                                                                                     dataset, m))

                m_counts, m_list = collect_mv_islv(islv_ll_paths, m_key=m_name,
                                                   count_key=args.count_key, burn_in=args.burn_in)

                assert len(m_counts) == len(m_list)
                print(len(miss_mask_list), len(m_counts))
                assert len(m_counts) == len(miss_mask_list)

                #
                # mv
                mv_valid_array_list = [linearize_mask(X, m, m_v) for m, m_v in zip(miss_mask_list,
                                                                                   miss_val_mask_list)]
                mv_test_array_list = [linearize_mask(X, m, m_t) for m, m_t in zip(miss_mask_list,
                                                                                  miss_test_mask_list)]

                # mv_valid_array_list = [linearize_mask_mf_preds(X, m, m_v) for m, m_v in zip(miss_mask_list,
                #                                                                    miss_val_mask_list)]
                # mv_test_array_list = [linearize_mask_mf_preds(X, m, m_t) for m, m_t in zip(miss_mask_list,
                #                                                                   miss_test_mask_list)]

                if m == 'mv-lls-d':
                    if args.mspn:
                        islv_d_mv_exp_path = os.path.join(islv_d_mv_exp_path, args.leaf_type[-1])
                        os.makedirs(islv_d_mv_exp_path, exist_ok=True)

                    m_stat_d_list, m_stds_d_list = compute_retrieve_aggr_stats_d(m_counts, m_list, aggr_indexes,
                                                                                 mv_valid_array_list, mv_test_array_list,
                                                                                 mapping_d_list)

                    for split_map, split_stds_map in zip(m_stat_d_list, m_stds_d_list):
                        for split_name, lls_d in split_map.items():
                            print(split_name, '\t', '\t'.join('{:.5f}'.format(l) for l in lls_d))
                            lls_d_path = os.path.join(islv_d_mv_exp_path, '{}-{}-lls-d'.format(dataset,
                                                                                               split_name))
                            np.save(lls_d_path, lls_d)
                            logging.info('Dumped ll d-wise to {}'.format(lls_d_path))

                        for split_name, stds_d in split_stds_map.items():
                            print(split_name, '\t', '\t'.join('{:.5f}'.format(l) for l in stds_d))
                            stds_d_path = os.path.join(islv_d_mv_exp_path, '{}-{}-stds-d'.format(dataset,
                                                                                                 split_name))
                            np.save(stds_d_path, stds_d)
                            logging.info('Dumped stds d-wise to {}'.format(stds_d_path))

                elif m == 'mv-lls' or m == 'mv-preds-lls':
                    m_stat_list = compute_retrieve_stats(m_counts, m_list,
                                                         mv_valid_array_list, mv_test_array_list, stats=STATS_MAP)
                    aggr_m_stat_list = compute_retrieve_aggr_stats(m_counts, m_list, aggr_indexes,
                                                                   mv_valid_array_list,
                                                                   mv_test_array_list, stats=STATS_MAP)
                    print_m_stats(islv_ll_paths, m_stat_list,
                                  best_key='valid', factors=args.factors)
                    print_m_stats(config_params, aggr_m_stat_list,
                                  best_key='valid')

                elif m == 'mv-preds-scores':

                    islv_hat_paths = path_collector(args.dir,
                                                    islv_sub_dir_lists,
                                                    file_name='mv-preds.pklz',
                                                    exp_num=1)
                    _h_counts, X_hat_lists = collect_mv_islv(islv_hat_paths, m_key='mv-preds',
                                                             count_key=args.count_key, burn_in=args.burn_in)
                    assert len(X_hat_lists) == len(miss_mask_list)

                    #
                    # computing perf dict

                    cont_metrics = ['MSE', 'MAE', 'RMSE', 'M-RMSE',
                                    'R-RMSE', 'S-RMSE']  # ['MSE', 'MAE']
                    disc_metrics = ['MSE', 'MAE',
                                    'ACC',
                                    'MSLE', 'RMSE', 'M-RMSE',
                                    'R-RMSE', 'S-RMSE']  # ['MSE', 'MSLE', 'MAE', 'ACC']

                    full_mse_list = []
                    full_mae_list = []
                    val_mse_list = []
                    val_mae_list = []
                    test_mse_list = []
                    test_mae_list = []
                    full_mse_cont_list = []
                    full_acc_disc_list = []
                    val_mse_cont_list = []
                    val_acc_disc_list = []
                    test_mse_cont_list = []
                    test_acc_disc_list = []

                    full_metrics_list = defaultdict(list)
                    val_metrics_list = defaultdict(list)
                    test_metrics_list = defaultdict(list)

                    full_perf_d_list = []
                    val_perf_d_list = []
                    test_perf_d_list = []
                    for X_miss, X_val_miss, X_test_miss, X_hat_list, f_p_d_list, p in zip(miss_mask_list,
                                                                                          miss_val_mask_list,
                                                                                          miss_test_mask_list,
                                                                                          X_hat_lists,
                                                                                          m_list,
                                                                                          islv_hat_paths):

                        f = split_path(p, args.factors)
                        print('\t'.join(f))

                        X_m = np.copy(X)
                        X_m[X_miss] = np.nan

                        X_eval = np.copy(X)
                        X_eval[~X_miss] = np.nan

                        val_perfs = []
                        full_perfs = []
                        test_perfs = []

                        # print(X_val_miss.sum(axis=0))
                        # print(X_test_miss.sum(axis=0))
                        for X_hat, f_p_d in zip(X_hat_list, f_p_d_list):

                            # print('H shape', X_hat.shape)
                            # print('XHat\n', X_hat)
                            # print('XHatsum', (~np.isnan(X_hat)).sum(), X_miss.sum(), )
                            # print('Xeval\n', X_eval)
                            # print('Xevalsum', (~np.isnan(X_eval)).sum(), X_miss.sum(), )
                            if args.lvstd:

                                #
                                # shifting discrete
                                for d in range(X.shape[1]):
                                    if C[d] == 4:
                                        X_hat[:, d] -= 1

                                val_perf_dict = compute_perf_dict_miss(X_m, X, X_hat, X_val_miss, C,
                                                                       X_min, X_max, X_mean, X_std,
                                                                       continuous_metrics=cont_metrics,
                                                                       discrete_metrics=disc_metrics)

                                test_perf_dict = compute_perf_dict_miss(X_m, X, X_hat, X_test_miss, C,
                                                                        X_min, X_max, X_mean, X_std,
                                                                        continuous_metrics=cont_metrics,
                                                                        discrete_metrics=disc_metrics)

                                full_perf_dict = compute_perf_dict_miss(X_m, X, X_hat, X_miss, C,
                                                                        X_min, X_max, X_mean, X_std,
                                                                        continuous_metrics=cont_metrics,
                                                                        discrete_metrics=disc_metrics)

                            elif args.hspn or args.mspn:

                                val_perf_dict = compute_predictions_dict_miss(X_eval, X_hat, X_val_miss, meta_types,
                                                                              X_min, X_max, X_mean, X_std,
                                                                              )
                                test_perf_dict = compute_predictions_dict_miss(X_eval, X_hat, X_test_miss, meta_types,
                                                                               X_min, X_max, X_mean, X_std,
                                                                               )
                                full_perf_dict = compute_predictions_dict_miss(X_eval, X_hat, X_miss, meta_types,
                                                                               X_min, X_max, X_mean, X_std,
                                                                               )

                            else:
                                raise ValueError('Unrecognized model')

                            full_perfs.append(full_perf_dict)
                            val_perfs.append(val_perf_dict)
                            test_perfs.append(test_perf_dict)

                            # print('VALID')
                            # print_perf_dict(val_perf_dict)
                            # print('TEST')
                            # print_perf_dict(test_perf_dict)
                            # print('FULL')
                            # print_perf_dict(full_perf_dict)
                            # print('ORIG')
                            # print_perf_dict(f_p_d)

                            # assert f_p_d == full_perf_dict
                            # for d, v in f_p_d.items():
                            #     for k, m_k in v.items():
                            #         if k != 'MSLE':
                            #             # print(d, m_k, full_perf_dict[d][k], X_hat[d], X[d], k)
                            #             assert np.isclose(m_k, full_perf_dict[d][k], rtol=1e-4), '{} {} {} {} {} {}'.format(
                            #                 d, m_k, full_perf_dict[d][k], X_hat[d], X[d], k)

                        #
                        # aggregating perf dict by samples
                        val_perf_d = aggr_weight_dicts(val_perfs)
                        test_perf_d = aggr_weight_dicts(test_perfs)
                        full_perf_d = aggr_weight_dicts(full_perfs)

                        full_perf_d_list.append(full_perf_d)
                        val_perf_d_list.append(val_perf_d)
                        test_perf_d_list.append(test_perf_d)

                        #
                        # print just one feature aggregated for all?
                        full_metrics_dict = {}
                        val_metrics_dict = {}
                        test_metrics_dict = {}

                        for mc in cont_metrics:
                            full_metrics_dict[mc] = (aggr_mv_score_dict(full_perf_d, m=mc))
                            full_metrics_list[mc].append(full_metrics_dict[mc])
                        print('full\t{}'.format('\t'.join(mc for mc in cont_metrics)))
                        print('\t{}'.format('\t'.join('{:.5f}'.format(
                            full_metrics_dict[mc]) for mc in cont_metrics)))

                        # full_mse_score_aggr = aggr_mv_score_dict(full_perf_d, m='MSE')
                        # full_mae_score_aggr = aggr_mv_score_dict(full_perf_d, m='MAE')
                        # print('full\tMSE\tMAE')
                        # print('\t{:.5f}\t{:.5f}'.format(full_mse_score_aggr, full_mae_score_aggr))
                        # full_mse_list.append(full_mse_score_aggr)
                        # full_mae_list.append(full_mae_score_aggr)

                        for mc in cont_metrics:
                            val_metrics_dict[mc] = (aggr_mv_score_dict(val_perf_d, m=mc))
                            val_metrics_list[mc].append(val_metrics_dict[mc])
                        print('valid\t{}'.format('\t'.join(mc for mc in cont_metrics)))
                        print('\t{}'.format('\t'.join('{:.5f}'.format(
                            val_metrics_dict[mc]) for mc in cont_metrics)))

                        # val_mse_score_aggr = aggr_mv_score_dict(val_perf_d, m='MSE')
                        # val_mae_score_aggr = aggr_mv_score_dict(val_perf_d, m='MAE')
                        # print('valid\tMSE\tMAE')
                        # print('\t{:.5f}\t{:.5f}'.format(val_mse_score_aggr, val_mae_score_aggr))
                        # val_mse_list.append(val_mse_score_aggr)
                        # val_mae_list.append(val_mae_score_aggr)

                        for mc in cont_metrics:
                            test_metrics_dict[mc] = (aggr_mv_score_dict(test_perf_d, m=mc))
                            test_metrics_list[mc].append(test_metrics_dict[mc])
                        print('test\t{}'.format('\t'.join(mc for mc in cont_metrics)))
                        print('\t{}'.format('\t'.join('{:.5f}'.format(
                            test_metrics_dict[mc]) for mc in cont_metrics)))

                        # test_mse_score_aggr = aggr_mv_score_dict(test_perf_d, m='MSE')
                        # test_mae_score_aggr = aggr_mv_score_dict(test_perf_d, m='MAE')
                        # print('test\tMSE\tMAE')
                        # print('\t{:.5f}\t{:.5f}'.format(test_mse_score_aggr, test_mae_score_aggr))
                        # test_mse_list.append(test_mse_score_aggr)
                        # test_mae_list.append(test_mae_score_aggr)

                        #
                        # or aggregate cont and disc independently?
                        full_c_mse, full_d_acc = aggr_mv_score_dict_meta_types(full_perf_d,
                                                                               m_cont='MSE',
                                                                               m_disc='ACC',
                                                                               meta_types=meta_types)
                        print('full\tMSE (cont)\t0/1 loss (disc)')
                        print('\t{:.5f}\t{:.5f}'.format(full_c_mse, full_d_acc))
                        full_mse_cont_list.append(full_c_mse)
                        full_acc_disc_list.append(full_d_acc)

                        val_c_mse, val_d_acc = aggr_mv_score_dict_meta_types(val_perf_d,
                                                                             m_cont='MSE',
                                                                             m_disc='ACC',
                                                                             meta_types=meta_types)
                        print('valid\tMSE (cont)\t0/1 loss (disc)')
                        print('\t{:.5f}\t{:.5f}'.format(val_c_mse, val_d_acc))
                        val_mse_cont_list.append(val_c_mse)
                        val_acc_disc_list.append(val_d_acc)

                        test_c_mse, test_d_acc = aggr_mv_score_dict_meta_types(test_perf_d,
                                                                               m_cont='MSE',
                                                                               m_disc='ACC',
                                                                               meta_types=meta_types)
                        print('test\tMSE (cont)\t0/1 loss (disc)')
                        print('\t{:.5f}\t{:.5f}'.format(test_c_mse, test_d_acc))
                        test_mse_cont_list.append(test_c_mse)
                        test_acc_disc_list.append(test_d_acc)

                    # full_perf_d_a = aggr_weight_dicts(full_perf_d_list)
                    # full_perf_d_s = std_weight_dicts(full_perf_d_list)

                    # val_perf_d_a = aggr_weight_dicts(val_perf_d_list)
                    # val_perf_d_s = std_weight_dicts(val_perf_d_list)

                    # test_perf_d_a = aggr_weight_dicts(test_perf_d_list)
                    # test_perf_d_s = std_weight_dicts(test_perf_d_list)

                    # for mc in cont_metrics:

                    #     full_perf_d_out_path = os.path.join(islv_d_mv_exp_path,
                    #                                         '{}-full-{}-mv-preds'.format(dataset, mc))
                    #     dump_perf_d(full_perf_d_a, full_perf_d_out_path, m=mc)
                    #     logging.info('Dropped full perf for {} to {}'.format(
                    #         mc, full_perf_d_out_path))
                    #     #
                    #     full_perf_d_out_path = os.path.join(islv_d_mv_exp_path,
                    #                                         '{}-full-{}-mv-preds-stds'.format(dataset, mc))
                    #     dump_perf_d(full_perf_d_s, full_perf_d_out_path, m=mc)
                    #     logging.info('Dropped full perf for {} to {}'.format(
                    #         mc, full_perf_d_out_path))

                    #     val_perf_d_out_path = os.path.join(islv_d_mv_exp_path,
                    #                                        '{}-valid-{}-mv-preds'.format(dataset, mc))
                    #     dump_perf_d(val_perf_d_a, val_perf_d_out_path, m=mc)
                    #     logging.info('Dropped val perf for {} to {}'.format(m, val_perf_d_out_path))
                    #     val_perf_d_out_path = os.path.join(islv_d_mv_exp_path,
                    #                                        '{}-valid-{}-mv-preds-stds'.format(dataset, mc))
                    #     dump_perf_d(val_perf_d_s, val_perf_d_out_path, m=mc)
                    #     logging.info('Dropped val perf for {} to {}'.format(
                    #         mc, val_perf_d_out_path))

                    #     test_perf_d_out_path = os.path.join(islv_d_mv_exp_path,
                    #                                         '{}-test-{}-mv-preds'.format(dataset, mc))
                    #     dump_perf_d(test_perf_d_a, test_perf_d_out_path, m=mc)
                    #     logging.info('Dropped test perf for {} to {}'.format(
                    #         mc, test_perf_d_out_path))
                    #     test_perf_d_out_path = os.path.join(islv_d_mv_exp_path,
                    #                                         '{}-test-{}-mv-preds-stds'.format(dataset, mc))
                    #     dump_perf_d(test_perf_d_s, test_perf_d_out_path, m=mc)
                    #     logging.info('Dropped test perf for {} to {}'.format(
                    #         mc, test_perf_d_out_path))

                    #
                    # print aggregation by config
                    for i, config in enumerate(config_params):
                        s_ids = (np.array(aggr_indexes) == i)
                        logging.info(s_ids)
                        print(config)

                        config_path = [str(c) for c in config]

                        config_path = os.path.join(islv_d_mv_exp_path, dataset,
                                                   *config_path)
                        os.makedirs(config_path, exist_ok=True)

                        full_config_d = np.array(full_perf_d_list)[s_ids]
                        full_perf_d_a = aggr_weight_dicts(full_config_d)
                        full_perf_d_s = std_weight_dicts(full_config_d)

                        full_dist, full_o, full_i = weight_dicts_to_matrix(full_config_d)
                        full_dist_path = os.path.join(config_path,
                                                      'full-err-dist.pickle')
                        with open(full_dist_path, 'wb') as f:
                            pickle.dump({'err_matrix': full_dist,
                                         'err_map': full_o,
                                         'err_inv_map': full_i}, f)
                        logging.info('Dropped full dist for {} to {}'.format(
                            mc, full_dist_path))

                        val_config_d = np.array(val_perf_d_list)[s_ids]
                        val_perf_d_a = aggr_weight_dicts(val_config_d)
                        val_perf_d_s = std_weight_dicts(val_config_d)
                        val_dist, val_o, val_i = weight_dicts_to_matrix(val_config_d)
                        val_dist_path = os.path.join(config_path,
                                                     'valid-err-dist.pickle')
                        with open(val_dist_path, 'wb') as f:
                            pickle.dump({'err_matrix': val_dist,
                                         'err_map': val_o,
                                         'err_inv_map': val_i}, f)
                        logging.info('Dropped valid dist for {} to {}'.format(
                            mc, val_dist_path))

                        test_config_d = np.array(test_perf_d_list)[s_ids]
                        test_perf_d_a = aggr_weight_dicts(test_config_d)
                        test_perf_d_s = std_weight_dicts(test_config_d)
                        test_dist, test_o, test_i = weight_dicts_to_matrix(test_config_d)
                        test_dist_path = os.path.join(config_path,
                                                      'test-err-dist.pickle')
                        with open(test_dist_path, 'wb') as f:
                            pickle.dump({'err_matrix': test_dist,
                                         'err_map': test_o,
                                         'err_inv_map': test_i}, f)
                        logging.info('Dropped test dist for {} to {}'.format(
                            mc, test_dist_path))

                        for mc in cont_metrics:

                            full_perf_d_out_path = os.path.join(config_path,
                                                                '{}-full-{}-mv-preds'.format(dataset, mc))
                            dump_perf_d(full_perf_d_a, full_perf_d_out_path, m=mc)
                            logging.info('Dropped full perf for {} to {}'.format(
                                mc, full_perf_d_out_path))
                            #
                            full_perf_d_out_path = os.path.join(config_path,
                                                                '{}-full-{}-mv-preds-stds'.format(dataset, mc))
                            dump_perf_d(full_perf_d_s, full_perf_d_out_path, m=mc)
                            logging.info('Dropped full perf for {} to {}'.format(
                                mc, full_perf_d_out_path))

                            val_perf_d_out_path = os.path.join(config_path,
                                                               '{}-valid-{}-mv-preds'.format(dataset, mc))
                            dump_perf_d(val_perf_d_a, val_perf_d_out_path, m=mc)
                            logging.info('Dropped val perf for {} to {}'.format(
                                m, val_perf_d_out_path))
                            val_perf_d_out_path = os.path.join(config_path,
                                                               '{}-valid-{}-mv-preds-stds'.format(dataset, mc))
                            dump_perf_d(val_perf_d_s, val_perf_d_out_path, m=mc)
                            logging.info('Dropped val perf for {} to {}'.format(
                                mc, val_perf_d_out_path))

                            test_perf_d_out_path = os.path.join(config_path,
                                                                '{}-test-{}-mv-preds'.format(dataset, mc))
                            dump_perf_d(test_perf_d_a, test_perf_d_out_path, m=mc)
                            logging.info('Dropped test perf for {} to {}'.format(
                                mc, test_perf_d_out_path))
                            test_perf_d_out_path = os.path.join(config_path,
                                                                '{}-test-{}-mv-preds-stds'.format(dataset, mc))
                            dump_perf_d(test_perf_d_s, test_perf_d_out_path, m=mc)
                            logging.info('Dropped test perf for {} to {}'.format(
                                mc, test_perf_d_out_path))

                            full_m = np.array(full_metrics_list[mc])[s_ids]
                            print('full\t{}\tmean\tmin\tmax\tstd'.format(mc))
                            print('\t\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}'.format(full_m.mean(),
                                                                              full_m.min(),
                                                                              full_m.max(),
                                                                              full_m.std()))

                            val_m = np.array(val_metrics_list[mc])[s_ids]
                            print('val\t{}\tmean\tmin\tmax\tstd'.format(mc))
                            print('\t\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}'.format(val_m.mean(),
                                                                              val_m.min(),
                                                                              val_m.max(),
                                                                              val_m.std()))

                            test_m = np.array(test_metrics_list[mc])[s_ids]
                            print('test\t{}\tmean\tmin\tmax\tstd'.format(mc))
                            print('\t\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}'.format(test_m.mean(),
                                                                              test_m.min(),
                                                                              test_m.max(),
                                                                              test_m.std()))

                        # #
                        # # full dist
                        # full_perf_m, full_o_t, full_inv_o_t = weight_dicts_to_matrix(
                        #     full_perf_d_list)
                        # full_perf_m_out_path = os.path.join(islv_d_mv_exp_path,
                        #                                     '{}-full-{}-mv-preds-dist'.format(dataset))
                        # dump_perf_dist(full_perf_d_s, full_perf_m_out_path, m=mc)
                        # logging.info('Dropped full perf for {} to {}'.format(
                        #     mc, full_perf_d_out_path))

                        # full_mse = np.array(full_mse_list)[s_ids]
                        # print('full\tmse\tmean\tmin\tmax\tstd')
                        # print('\t\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}'.format(full_mse.mean(),
                        #                                                   full_mse.min(),
                        #                                                   full_mse.max(),
                        #                                                   full_mse.std()))
                        # full_mae = np.array(full_mae_list)[s_ids]
                        # print('full\tmae\tmean\tmin\tmax\tstd')
                        # print('\t\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}'.format(full_mae.mean(),
                        #                                                   full_mae.min(),
                        #                                                   full_mae.max(),
                        #                                                   full_mae.std()))
                        # val_mse = np.array(val_mse_list)[s_ids]
                        # print('valid\tmse\tmean\tmin\tmax\tstd')
                        # print('\t\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}'.format(val_mse.mean(),
                        #                                                   val_mse.min(),
                        #                                                   val_mse.max(),
                        #                                                   val_mse.std()))
                        # val_mae = np.array(val_mae_list)[s_ids]
                        # print('valid\tmae\tmean\tmin\tmax\tstd')
                        # print('\t\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}'.format(val_mae.mean(),
                        #                                                   val_mae.min(),
                        #                                                   val_mae.max(),
                        #                                                   val_mae.std()))
                        # test_mse = np.array(test_mse_list)[s_ids]
                        # print('test\tmse\tmean\tmin\tmax\tstd')
                        # print('\t\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}'.format(test_mse.mean(),
                        #                                                   test_mse.min(),
                        #                                                   test_mse.max(),
                        #                                                   test_mse.std()))
                        # test_mae = np.array(test_mae_list)[s_ids]
                        # print('test\tmae\tmean\tmin\tmax\tstd')
                        # print('\t\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}'.format(test_mae.mean(),
                        #                                                   test_mae.min(),
                        #                                                   test_mae.max(),
                        #                                                   test_mae.std()))
                        #########
                        full_mse_c = np.array(full_mse_cont_list)[s_ids]
                        print('full\tMSE (cont)\tmean\tmin\tmax\tstd')
                        print('\t\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}'.format(full_mse_c.mean(),
                                                                          full_mse_c.min(),
                                                                          full_mse_c.max(),
                                                                          full_mse_c.std()))
                        full_acc_d = np.array(full_acc_disc_list)[s_ids]
                        print('full\t0/1 loss (disc)\tmean\tmin\tmax\tstd')
                        print('\t\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}'.format(full_acc_d.mean(),
                                                                          full_acc_d.min(),
                                                                          full_acc_d.max(),
                                                                          full_acc_d.std()))
                        val_mse_c = np.array(val_mse_cont_list)[s_ids]
                        print('valid\tMSE (cont)\tmean\tmin\tmax\tstd')
                        print('\t\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}'.format(val_mse_c.mean(),
                                                                          val_mse_c.min(),
                                                                          val_mse_c.max(),
                                                                          val_mse_c.std()))
                        val_acc_d = np.array(val_acc_disc_list)[s_ids]
                        print('valid\t0/1 loss (disc)\tmean\tmin\tmax\tstd')
                        print('\t\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}'.format(val_acc_d.mean(),
                                                                          val_acc_d.min(),
                                                                          val_acc_d.max(),
                                                                          val_acc_d.std()))
                        test_mse_c = np.array(test_mse_cont_list)[s_ids]
                        print('test\tMSE (cont)\tmean\tmin\tmax\tstd')
                        print('\t\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}'.format(test_mse_c.mean(),
                                                                          test_mse_c.min(),
                                                                          test_mse_c.max(),
                                                                          test_mse_c.std()))
                        test_acc_d = np.array(test_acc_disc_list)[s_ids]
                        print('test\t0/1 loss (disc)\tmean\tmin\tmax\tstd')
                        print('\t\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}'.format(test_acc_d.mean(),
                                                                          test_acc_d.min(),
                                                                          test_acc_d.max(),
                                                                          test_acc_d.std()))
