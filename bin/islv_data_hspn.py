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

from visualize import visualize_data_partition, reorder_data_partitions, visualize_histogram
from bin.spstd_model_ha1 import load_islv_mat


FEATURE_VIS_DIR = 'vis-features'

if __name__ == '__main__':

    #########################################
    # creating the opt parser
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str,
                        help='Path to dataset (ISLV) in .mat format')

    parser.add_argument('-o', '--output', type=str,
                        default='./data/',
                        help='Output path to converted dataset format')

    parser.add_argument('-d', '--dataset-name', type=str,
                        help='Dataset name')

    parser.add_argument('--tvt-split', type=float, nargs='+',
                        #default=[0.7, 0.1, 0.2],
                        default=None,
                        help='Train, validation and test split percentages')

    parser.add_argument('--seed', type=int, nargs='?',
                        default=1337,
                        help='Seed for the random generator')

    parser.add_argument('--show-plots', action='store_true',
                        help='showing plots')

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
    dataset_name = args.dataset_name
    out_path = os.path.join(args.output,  '{}'.format(dataset_name))
    os.makedirs(out_path, exist_ok=True)

    args_out_path = os.path.join(out_path, 'args.json')
    json_args = json.dumps(vars(args))
    logging.info("Starting with arguments:\n%s\n\tdumped at %s", json_args, args_out_path)
    with open(args_out_path, 'w') as f:
        f.write(json_args)

    #
    # setting up the seed
    rand_gen = np.random.RandomState(args.seed)

    X, meta_types, domains = load_islv_mat(args.dataset)
    logging.info('Loaded dataset from {}'.format(args.dataset))
    logging.info('\twith shape {}\n\tmeta-types: {}\n\tdomains: {}'.format(X, meta_types, domains))

    #
    # extracting statistics from it
    min_X = np.array([np.min(X[:, d]) for d in range(X.shape[1])])
    logging.info('Min X: {}'.format(min_X))
    max_X = np.array([np.max(X[:, d]) for d in range(X.shape[1])])
    logging.info('Max X: {}'.format(max_X))

    N, D = X.shape

    #
    # resaving full data
    full_data_path = os.path.join(out_path, 'full.data')
    with open(full_data_path, 'wb') as f:
        pickle.dump(X, f)
    logging.info('Saved full X data to {}'.format(full_data_path))

    #
    # saving statistics
    data_stats = {'min': min_X, 'max': max_X, 'domains': domains,
                  'types': None, 'meta-types': meta_types,
                  'spn-W': None,
                  'spn-type-W': None,
                  'full-type-W': None,
                  'train-type-W': None,
                  'valid-type-W': None,
                  'test-type-W': None, }
    data_stats_path = os.path.join(out_path, 'data.stats')
    with open(data_stats_path, 'wb') as f:
        pickle.dump(data_stats, f)

    fvis_out_path = os.path.join(out_path, FEATURE_VIS_DIR)
    os.makedirs(fvis_out_path, exist_ok=True)
    #
    # visualizing single features
    for d in range(D):
        f_hist_out_path = os.path.join(fvis_out_path, 'd{}.hist'.format(d))
        visualize_histogram(X[:, d], show_fig=args.show_plots,
                            output=f_hist_out_path)

    ##########################################################################################
    # splitting
    #
    if args.tvt_split:
        train_valid_test_splits = np.array(args.tvt_split)
        train_valid_test_splits = train_valid_test_splits / train_valid_test_splits.sum()
        cum_splits = np.cumsum(train_valid_test_splits)[:-1]

        ids = np.arange(N)
        split_ids = (cum_splits * N).astype(np.int64)
        # print(split_ids)
        rand_gen.shuffle(ids)
        train_ids, valid_ids, test_ids = np.split(ids, split_ids)
        print(train_ids, valid_ids, test_ids)

        assert len(train_ids) + len(valid_ids) + len(test_ids) == N
        assert len(set(train_ids).intersection(valid_ids)) == 0
        assert len(set(test_ids).intersection(valid_ids)) == 0
        assert len(set(test_ids).intersection(train_ids)) == 0

        X_train = X[train_ids]
        X_valid = X[valid_ids]
        X_test = X[test_ids]

        #
        # visualizing partitions and saving data to disk
        for x_split,  split_name in zip([X_train, X_valid, X_test],
                                        ['train', 'valid', 'test']):

            split_data_path = os.path.join(out_path, '{}.data'.format(split_name))
            with open(split_data_path, 'wb') as f:
                pickle.dump(x_split, f)
            logging.info('Saved {} X data to {}'.format(split_name, split_data_path))

            #
            # visualizing single features
            for d in range(D):
                fvis_split_out_path = os.path.join(fvis_out_path, split_name)
                os.makedirs(fvis_split_out_path, exist_ok=True)
                f_hist_out_path = os.path.join(fvis_split_out_path,  'd{}.hist'.format(d))
                visualize_histogram(x_split[:, d], show_fig=args.show_plots,
                                    output=f_hist_out_path)
