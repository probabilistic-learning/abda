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

import scipy.io
import numpy as np
from numpy.testing import assert_array_equal
import numba

from spn.structure.StatisticalTypes import MetaType
from visualize import visualize_data_partition, reorder_data_partitions, visualize_histogram
from bin.spstd_model_ha1 import load_islv_mat
# from mlutils.datasets import loadMLC

FEATURE_VIS_DIR = 'vis-features'
EPS = 1e-6

FEATURE_TYPES = {'continuous',  # for all continuous data types (all-real/positive/interval)
                 'non-pos-continuous',  # special case for real types with negative values observed
                 # 'categorical',  # not used anymore just for backward compatibility with MSPNs
                 'discrete',  # for all discrete data (categorical/ordinal/count)
                 'binary',  # special, undistinguishable case for discrete RVs
                 }


def load_feature_info(feature_info_path,
                      endline='.',
                      sep=':',
                      domain_sep=',',
                      float_dtype=np.float64,
                      int_dtype=np.int32):
    """
    Loads a .feature file and returns:
      - a list of feature names (strings),
      - a list of feature meta-types in FEATURE_TYPES (continuous/non-pos-continuous/discrete/binary)
      - a list of ordered numpy arrays of floats (continuous RVs) or integers (discrete RVs) as domains
    """

    lines = None
    with open(feature_info_path, 'r') as f:
        lines = f.readlines()

    logging.info('Read {} lines from {}'.format(len(lines),
                                                feature_info_path))
    #
    # removing empty lines
    lines = [l for l in lines if l.strip()]
    logging.info('{} samples after removing empty lines'.format(len(lines)))

    feature_names = []
    feature_types = []
    domains = []

    # cont_domain_str = ['min', 'max']

    for l in lines:
        #
        # stripping endline
        l = l.strip()
        assert l[-1] == endline, l
        # tokenizing
        tokens = l[:-1].split(sep)

        assert len(tokens) == 2 or len(tokens) == 3, len(tokens)

        #
        # tokens[0] RV name
        # tokens[1] RV type
        # tokens[2] optional domain
        feature_names.append(tokens[0].strip())
        ftype = tokens[1].strip()
        assert ftype in FEATURE_TYPES, ftype
        feature_types.append(ftype)
        if len(tokens) == 3:
            domain_str = tokens[2].strip()
            domain_vals = np.array([d.strip() for d in domain_str.split(domain_sep)])

            #
            # categorical and discrete case, map everything to integers from 0 to K
            if ftype == 'categorical' or ftype == 'discrete' or ftype == 'binary':
                domain_vals = domain_vals.astype(int_dtype)

            #
            # discrete, map everything that is not a number to an unused number
            elif ftype == 'continuous' or ftype == 'non-pos-continuous':
                domain_vals = domain_vals.astype(float_dtype)

            else:
                raise NotImplementedError('Unrecognized feature type {}'.format(tokens))
            domains.append(np.sort(domain_vals))
        else:
            domains.append(None)

    return feature_names, feature_types, domains


def loadMLC(dsname,
            # base_path=os.path.dirname(__file__),
            data_dir="datasets/MLC/proc-db/proc/"):

    dataset_path = os.path.join(data_dir, dsname)
    feature_names, feature_types, domains = load_feature_info('{}.features'.format(dataset_path))

    train = np.loadtxt("{}.train.data".format(dataset_path), delimiter=",")
    test = np.loadtxt("{}.test.data".format(dataset_path), delimiter=",")
    valid = np.loadtxt("{}.valid.data".format(dataset_path), delimiter=",")

    return ((train, valid, test), feature_names, feature_types, domains)


if __name__ == '__main__':

    #########################################
    # creating the opt parser
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str,
                        help='Path to MSPN dataset folder format')

    parser.add_argument('-o', '--output', type=str,
                        default='./data/islv',
                        help='Output path to converted dataset format')

    parser.add_argument('-d', '--dataset-name', type=str,
                        help='Dataset name')

    parser.add_argument('--miss-perc', type=float,
                        default=None,
                        help='Percentage of missing values to generate (for transductive setting)')

    parser.add_argument('--miss-val-perc', type=float,
                        default=None,
                        help='Percentage of missing values to generate for validation (for transductive setting)')

    parser.add_argument('--tvt-split', type=float, nargs='+',
                        # default=[0.7, 0.1, 0.2],
                        default=None,
                        help='Train, validation and test split percentages (for inductive setting)')

    parser.add_argument('--seed', type=int, nargs='?',
                        default=1337,
                        help='Seed for the random generator')

    parser.add_argument('--show-plots', action='store_true',
                        help='showing plots')

    parser.add_argument('--rm-binary', action='store_true',
                        help='Whether to remove binary features')

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
    dataset_name_pp = '{}PP'.format(dataset_name)
    out_path = os.path.join(args.output,  '{}'.format(dataset_name_pp))
    os.makedirs(out_path, exist_ok=True)

    args_out_path = os.path.join(out_path, 'args.json')
    json_args = json.dumps(vars(args))
    logging.info("Starting with arguments:\n%s\n\tdumped at %s", json_args, args_out_path)
    with open(args_out_path, 'w') as f:
        f.write(json_args)

    #
    # setting up the seed
    rand_gen = np.random.RandomState(args.seed)

    #
    # loading data
    (train, valid, test), feature_names, types, t_domains = loadMLC(args.dataset_name,
                                                                    data_dir=args.dataset,
                                                                    )

    assert train.shape[1] == valid.shape[1]
    assert valid.shape[1] == test.shape[1]
    D = train.shape[1]

    logging.info('\n\nLoaded dataset {}'.format(args.dataset))
    logging.info('\n\twith shapes ({} {} {})\n\tfeature names: {}\n\ttypes:{}\n\tdomains: {}\n\n'.format(train.shape,
                                                                                                         valid.shape,
                                                                                                         test.shape,
                                                                                                         feature_names,
                                                                                                         types, t_domains))

    meta_types = [None for d in range(D)]
    islv_meta_types = np.zeros(D, dtype=np.int64)

    for i, t in enumerate(types):
        if t == 'binary':
            meta_types[i] = MetaType.BINARY
            islv_meta_types[i] = 3

        elif t == 'continuous':
            meta_types[i] = MetaType.REAL
            if np.all(t_domains[i] >= 0):
                islv_meta_types[i] = 1
            else:
                islv_meta_types[i] = 2

        elif t == 'non-pos-continuous':
            meta_types[i] = MetaType.REAL
            islv_meta_types[i] = 2

        elif t == 'discrete':
            meta_types[i] = MetaType.DISCRETE
            islv_meta_types[i] = 4

    meta_types = np.array(meta_types)

    logging.info('Meta type conversion:\n\thspn:{}\n\tislv:{}'.format(meta_types,
                                                                      islv_meta_types))

    #
    # gluing all dataset splits in a single file
    X = np.concatenate((train, valid, test), axis=0)
    assert X.shape[1] == train.shape[1]
    assert X.shape[0] == train.shape[0] + valid.shape[0] + test.shape[0]
    logging.info('Concatenated into full dataset of shape {}'.format(X.shape))
    N, D = X.shape

    #
    # adding small eps to zero
    # domains = np.max(X, axis=0) + 1
    domains = []
    for d in range(D):
        #
        # to avoid numerical errors in positive real data
        if meta_types[d] == MetaType.REAL:
            X[np.isclose(X[:, d], 0), d] = EPS
            domains.append(np.array([np.min(X[:, d]), np.max(X[:, d])]))
        elif meta_types[d] == MetaType.DISCRETE or meta_types[d] == MetaType.BINARY:
            domains.append(np.arange(int(np.max(X[:, d], axis=0)) + 1))

    print(domains)
    # #
    # # inherent missing values coded as -1?
    # # making them nans
    # in_mv_ids = (X == -1)
    # logging.info('Found {} inherent missing values'.format(in_mv_ids.sum()))
    # X[in_mv_ids] = np.nan
    # X_islv_orig[in_mv_ids] = np.nan

    #
    # extracting statistics from it
    min_X = np.array([np.nanmin(X[:, d]) for d in range(D)])
    logging.info('Min X: {}'.format(min_X))
    max_X = np.array([np.nanmax(X[:, d]) for d in range(D)])
    logging.info('Max X: {}'.format(max_X))

    #
    # removing binary features
    if args.rm_binary:
        logging.info('Preparing to remove binary features...')
        not_binary_f_ids = np.array([meta_types[d] != MetaType.BINARY
                                     for d in range(D)], dtype=bool)
        logging.info('\tnot binary features found {}'.format(not_binary_f_ids))
        X = X[:, not_binary_f_ids]
        logging.info('\tnew dataset shape: {}'.format(X.shape))
        meta_types = meta_types[not_binary_f_ids]
        islv_meta_types = islv_meta_types[not_binary_f_ids]
        logging.info('\tnew meta types: {}'.format(meta_types))
        domains = [domains[d] for d in range(D) if not_binary_f_ids[d]]

        logging.info('\tnew domains: {}'.format(domains))
        min_X = min_X[not_binary_f_ids]
        logging.info('\tnew mins: {}'.format(min_X))
        max_X = max_X[not_binary_f_ids]
        logging.info('\tnew maxs: {}'.format(max_X))

        feature_names = np.array(feature_names)[not_binary_f_ids]
        feature_names = {i: f for i, f in enumerate(feature_names)}
        logging.info('\tnew feature names: {}'.format(max_X))

        N, D = X.shape

    assert len(islv_meta_types) == D

    #
    # converting in ISLV format
    islv_X = np.copy(X)
    R = np.ones(D)
    for d in range(D):
        if meta_types[d] == MetaType.DISCRETE:
            islv_X[:, d] += 1
            R[d] = max_X[d] + 1

    logging.info('Converted into ISLV format with\n\tX:{}\n\tC:{}\n\tR:{}'.format(islv_X,
                                                                                  islv_meta_types,
                                                                                  R))
    #
    # saving in ISLV .mat format
    full_islv_path = os.path.join(out_path, '{}.mat'.format(dataset_name_pp))
    islv_data = {'N': N, 'D': D,
                 'X': islv_X,
                 'R': R.astype(np.float64),  # for the original C code...
                 'T': islv_meta_types}
    scipy.io.savemat(full_islv_path, islv_data)
    logging.info('Saved data to ISLV format to {}'.format(full_islv_path))

    #
    # resaving full data in new format
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
                  'test-type-W': None,
                  'feature-names': feature_names}
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

    #
    # missing values (on full dataset)
    if args.miss_perc:
        logging.info('\n\nMissing value mask generation')
        # X_miss = rand_gen.binomial(p=args.miss_perc, n=1, size=X.shape).astype(bool)
        n_obs = X.shape[0] * X.shape[1]
        logging.info('\t\tthere are {} obs'.format(n_obs))
        X_miss = np.zeros(n_obs, dtype=bool)
        miss_perc = None
        n_mv = None
        n_mv_val = None
        val_mv_ids = None
        if args.miss_val_perc:
            miss_perc = args.miss_val_perc + args.miss_perc
        else:
            miss_perc = args.miss_perc
        n_mv = int(n_obs * miss_perc)
        logging.info('\t\tconsidering {} to be missing'.format(n_mv))

        mv_ids = rand_gen.choice(n_obs, replace=False, size=n_mv)
        if args.miss_val_perc:
            n_mv_val = int(n_obs * args.miss_val_perc)
            val_mv_ids = mv_ids[:n_mv_val]
            X_miss_val = np.zeros(n_obs, dtype=bool)
            X_miss_val[val_mv_ids] = True
            X_miss_val = X_miss_val.reshape(X.shape)
            assert X_miss_val.sum() == n_mv_val
            assert X_miss_val.shape == X.shape

        X_miss[mv_ids] = True
        assert X_miss.sum() == n_mv
        X_miss = X_miss.reshape(X.shape)

        # rand_gen.shuffle(X_miss)
        assert X_miss.sum() == n_mv
        assert X_miss.shape == X.shape
        if args.miss_val_perc:
            assert_array_equal(X_miss_val & X_miss, X_miss_val)
            miss_val_data_path = os.path.join(out_path, 'miss', str(
                args.miss_perc), str(args.seed))
            os.makedirs(miss_val_data_path, exist_ok=True)
            miss_val_data_path = os.path.join(miss_val_data_path, 'miss.val.data')
            # out_path, 'miss-val-{}-{}-{}.full.data'.format(args.miss_perc, args.miss_val_perc, args.seed))
            with open(miss_val_data_path, 'wb') as f:
                pickle.dump(X_miss_val, f)
            logging.info('Saved miss val full X data to {}\n\twith {} missing values'.format(miss_val_data_path,
                                                                                             X_miss_val.sum()))
        # X_miss = rand_gen.binomial(p=args.miss_perc, n=1, size=X.shape).astype(bool)
        miss_data_path = os.path.join(out_path, 'miss', str(args.miss_perc), str(args.seed))
        os.makedirs(miss_data_path, exist_ok=True)
        miss_data_path = os.path.join(miss_data_path, 'miss.full.data')
        # out_path, 'miss-{}-{}.full.data'.format(args.miss_perc, args.seed))
        with open(miss_data_path, 'wb') as f:
            pickle.dump(X_miss, f)
        logging.info('Saved miss full X data to {}\n\twith {} missing values'.format(miss_data_path,
                                                                                     X_miss.sum()))

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
            logging.info('\t with shape {}'.format(x_split.shape))

            #
            # visualizing single features
            for d in range(D):
                fvis_split_out_path = os.path.join(fvis_out_path, split_name)
                os.makedirs(fvis_split_out_path, exist_ok=True)
                f_hist_out_path = os.path.join(fvis_split_out_path,  'd{}.hist'.format(d))
                visualize_histogram(x_split[:, d], show_fig=args.show_plots,
                                    output=f_hist_out_path)
