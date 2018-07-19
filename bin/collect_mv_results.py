
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

import numpy as np
import numba
from spn.structure.StatisticalTypes import MetaType


PATH_NAME = 'path.map.pickle'
LLS_FILE_NAME = 'mv-lls.pickle'
SCORES_FILE_NAME = 'mv-preds-scores.pickle'


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


def split_path(path, factors):
    path = os.path.normpath(path)
    split_path = path.split(os.sep)
    return np.array(split_path)[factors]


if __name__ == '__main__':

    #########################################
    # creating the opt parser
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", type=str,
                        help='directory to files')

    parser.add_argument('-o', '--output', type=str,
                        default='./exp/',
                        help='Output path to exp result')

    parser.add_argument('--factors', type=int, nargs='+',
                        default=[5, 6, 7, 8, 9],
                        help='A tuple of integers')

    parser.add_argument('--seed', type=int, nargs='?',
                        default=1337,
                        help='Seed for the random generator')

    parser.add_argument('--exp-id', type=str,
                        default=None,
                        help='Dataset input suffix')

    parser.add_argument('--stats', type=str,
                        default=None,
                        help='Dataset statistics')

    parser.add_argument('--miss-perc', type=str,
                        default=None,
                        help='missing perc')

    parser.add_argument('--flag', action='store_true',
                        help='A boolean argument')

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
    # dataset_name = 'tempdata'

    # if args.exp_id:
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
    rand_gen = np.random.RandomState(args.seed)

    # out_log_path = os.path.join(out_path,  'exp.log')
    # logging.info('Opening log file... {}'.format(out_log_path))

    #
    # loading meta types
    meta_types = None
    if args.stats is not None:
        stats_map = None
        with open(args.stats, 'rb') as f:
            stats_map = pickle.load(f)
        meta_types = stats_map['meta-types']

    #
    # open path map name
    path_map_path = os.path.join(args.dir, '{}.{}'.format(args.exp_id, PATH_NAME))
    path_map = None
    with open(path_map_path, 'rb') as f:
        path_map = pickle.load(f)
        logging.info('loaded path file name map {}'.format(path_map_path))

    #
    # open lls
    mv_lls_path = os.path.join(args.dir, '{}-{}-{}'.format(args.exp_id,
                                                           args.miss_perc,
                                                           LLS_FILE_NAME))
    mv_lls = None
    with open(mv_lls_path, 'rb') as f:
        mv_lls = pickle.load(f)
        logging.info('loaded mv lls file  {} {}'.format(mv_lls_path, len(mv_lls[0])))

    #
    # open scores
    mv_preds_scores_path = os.path.join(args.dir, '{}-{}-{}'.format(args.exp_id,
                                                                    args.miss_perc,
                                                                    SCORES_FILE_NAME))

    for (i, path), mv_lls_p in zip(path_map.items(), mv_lls):
        if mv_lls_p and path:
            factors = split_path(path, np.array(args.factors))
            avg_mv_lls_p = np.array([mvl.mean() for mvl in mv_lls_p])
            print('{}\t{}\t{}\t{}\t{}'.format(i, '\t'.join(factors),
                                              # len(mv_lls_p), avg_mv_lls_p.shape,
                                              np.mean(avg_mv_lls_p), np.min(avg_mv_lls_p), np.max(avg_mv_lls_p)))
        else:
            print('missing value for path', path)

    mv_preds_scores = None
    with open(mv_preds_scores_path, 'rb') as f:
        mv_preds_scores = pickle.load(f)
        logging.info('loaded mv scores file  {}'.format(mv_preds_scores_path))

    for (i, path), mv_scores_p in zip(path_map.items(), mv_preds_scores):
        if mv_scores_p and path:
            factors = split_path(path, np.array(args.factors))
            aggr_scores = np.array([aggr_mv_score_dict(mvs, m='MSE') for mvs in mv_scores_p])
            aggr_scores_d = [aggr_mv_score_dict_meta_types(mvs,
                                                           m_cont='MSE',
                                                           m_disc='ACC',
                                                           meta_types=meta_types)
                             for mvs in mv_scores_p]
            aggr_scores_cont = np.array([mv_c for mv_c, _mv_d in aggr_scores_d])
            aggr_scores_disc = np.array([mv_d for _mv_c, mv_d in aggr_scores_d])
            print('{}\t{}\t{}\t{}\t{}'.format(i, '\t'.join(factors),
                                              aggr_scores.mean(),
                                              aggr_scores_cont.mean(), aggr_scores_disc.mean()))

        else:
            print('missing value for path', path)
