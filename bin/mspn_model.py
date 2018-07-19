"""
Automatic and Tractable Density Estimation with Bayesian Sum-Product Networks

MODEL-HA1

@author: antonio vergari

--------

"""

from filelock import FileLock

from bin.collect_mv_results import aggr_mv_score_dict, aggr_mv_score_dict_meta_types
from bin.spstd_model_ha1 import *
from spn.structure.leaves.histogram.Histograms import create_histogram_leaf
from spn.structure.leaves.histogram.Inference import add_histogram_inference_support
from spn.structure.leaves.histogram.Text import add_histogram_text_support
from spn.structure.leaves.piecewise.Inference import add_piecewise_inference_support
from spn.structure.leaves.piecewise.PiecewiseLinear import create_piecewise_leaf
from spn.structure.leaves.piecewise.Text import add_piecewise_text_support
from spn.structure.leaves.typedleaves.Text import add_typed_leaves_text_support

try:
    from time import perf_counter
except:
    from time import time

    perf_counter = time
import datetime
import os
import sys

sys.setrecursionlimit(15500)
import logging
import pickle
import gzip

import numpy as np
from numpy.testing import assert_array_equal

from spn.structure.Base import *
from spn.structure.leaves.parametric.Inference import add_parametric_inference_support
from spn.structure.leaves.parametric.Parametric import *
from spn.io.Text import spn_to_str_equation
from spn.algorithms.Statistics import get_structure_stats_dict2
from spn.algorithms.LearningWrappers import learn_mspn_with_missing
from tfspn.explain import print_perf_dict


def dump_obj(opath, fname, obj):
    out_file_path = os.path.join(opath, fname)
    with open(out_file_path, 'wb') as f:
        pickle.dump(obj, f)


if __name__ == '__main__':

    #########################################
    #
    # parsing the args
    parser = get_parse_args()
    parser.add_argument('--mspn-leaves', type=str,
                        default="histogram",
                        help='What types of leaves to create')
    args = parser.parse_args()

    if args.exp_id:
        out_path = os.path.join(args.output, args.exp_id)
        os.makedirs(out_path, exist_ok=True)
    else:
        raise Exception("exp_id needed")

    logging.basicConfig(filename=os.path.join(out_path, 'exp.log'), level=logging.DEBUG)

    rand_gen = np.random.RandomState(args.seed)

    logging.info("Starting with arguments:\n%s", args)

    dump_obj(out_path, "learning_args.pkl", args)

    # load data
    #
    X_full, X_train, X_valid, X_test = None, None, None, None
    meta_types, domains = None, None
    g_true_W = None

    logging.info('Looking into dir {}...\n'.format(args.dataset))

    # full
    full_data_path = os.path.join(args.dataset, 'full.data')
    if os.path.exists(full_data_path) and args.miss is None:
        load_start_t = perf_counter()
        with open(full_data_path, 'rb') as f:
            X_full = pickle.load(f)
        load_end_t = perf_counter()
        logging.info('Loaded full data {} from {} (in {} secs)'.format(X_full.shape,
                                                                       full_data_path,
                                                                       load_end_t - load_start_t))
    else:
        logging.info('Cannot load full dataset {}...\tskipping'.format(full_data_path))

    # train
    train_data_path = os.path.join(args.dataset, 'train.data')
    if os.path.exists(train_data_path):
        load_start_t = perf_counter()
        with open(train_data_path, 'rb') as f:
            X_train = pickle.load(f)
        load_end_t = perf_counter()
        logging.info('Loaded train data {} from {} (in {} secs)'.format(X_train.shape,
                                                                        train_data_path,
                                                                        load_end_t - load_start_t))
    else:
        logging.info('Cannot load train dataset {}...\tskipping'.format(train_data_path))
        X_train, meta_types, domains = load_islv_mat(args.dataset)

    # valid
    valid_data_path = os.path.join(args.dataset, 'valid.data')
    if os.path.exists(valid_data_path) and args.miss is None:
        load_start_t = perf_counter()
        with open(valid_data_path, 'rb') as f:
            X_valid = pickle.load(f)
        load_end_t = perf_counter()
        logging.info('Loaded valid data {} from {} (in {} secs)'.format(X_valid.shape,
                                                                        valid_data_path,
                                                                        load_end_t - load_start_t))
    else:
        logging.info('Cannot load valid dataset {}...\tskipping'.format(valid_data_path))

    # test
    test_data_path = os.path.join(args.dataset, 'test.data')
    if os.path.exists(valid_data_path) and args.miss is None:
        load_start_t = perf_counter()
        with open(test_data_path, 'rb') as f:
            X_test = pickle.load(f)
        load_end_t = perf_counter()
        logging.info('Loaded test data {} from {} (in {} secs)'.format(X_test.shape,
                                                                       test_data_path,
                                                                       load_end_t - load_start_t))
    else:
        logging.info('Cannot load test dataset {}...\tskipping'.format(test_data_path))

    if X_train is not None and X_valid is not None:
        assert X_train.shape[1] == X_valid.shape[1]
    if X_test is not None and X_valid is not None:
        assert X_test.shape[1] == X_valid.shape[1]
    if X_full is not None and X_valid is not None:
        assert X_full.shape[1] == X_valid.shape[1]
    if X_train is not None and X_valid is not None and X_test is not None and X_full is not None:
        assert X_train.shape[0] + X_valid.shape[0] + X_test.shape[0] == X_full.shape[0]

    #
    # loading the missing value mask
    X_miss = None
    X_train_copy = None
    X_eval = None
    if args.miss is not None:
        logging.info('Looking for pickle at dir... {}'.format(args.miss))
        miss_data_path = args.miss

        try:
            load_start_t = perf_counter()
            with open(miss_data_path, 'rb') as f:
                X_miss = pickle.load(f)
            load_end_t = perf_counter()
        except:
            logging.info('FAILED to load pickle, trying matlab .mat file')
            load_start_t = perf_counter()
            X_miss = load_islv_miss(miss_data_path, shape=X_train.shape)
            load_end_t = perf_counter()

        logging.info('Loaded missing data mask {} from {} (in {} secs)'.format(X_miss.shape,
                                                                               miss_data_path,
                                                                               load_end_t - load_start_t))
        assert X_miss.shape == X_train.shape, (X_miss.shape, X_train.shape)

        X_eval = np.copy(X_train)
        X_eval[~X_miss] = np.nan

        # masking the train
        X_train_orig = np.copy(X_train)
        X_train[X_miss] = np.nan

        assert_array_equal(np.isnan(X_train_orig) | X_miss, np.isnan(X_train))

        logging.info('\n\nDEALING with {} missing values '.format(np.isnan(X_train).sum()))

    logging.info('train:{}'.format(X_train.shape))

    if meta_types is None:
        stats_map = {}
        data_stats_path = os.path.join(args.dataset, 'data.stats')
        with open(data_stats_path, 'rb') as f:
            stats_map = pickle.load(f)
        meta_types = stats_map['meta-types']
        domains = stats_map['domains']
        g_true_W = stats_map.get('full-type-W')

    ds_context = Context(meta_types=meta_types)
    ds_context.domains = domains

    if args.mspn_leaves == "histogram":
        leaves = create_histogram_leaf
    elif args.mspn_leaves == "piecewise":
        leaves = create_piecewise_leaf

    learn_start_t = perf_counter()
    spn = learn_mspn_with_missing(X_train,
                                  ds_context,
                                  min_instances_slice=args.min_inst_slice,
                                  threshold=args.col_split_threshold,
                                  linear=True,
                                  memory=None,
                                  leaves=leaves

                                  )

    learn_end_t = perf_counter()
    learning_time = learn_end_t - learn_start_t
    logging.info(
        '\n\nLearned spn in {} secs\n\t with stats:\n\t{}'.format(learning_time, get_structure_stats_dict2(spn)))

    dump_obj(out_path, 'spn.model.pkl', spn)

    add_typed_leaves_text_support()
    add_parametric_inference_support()
    add_histogram_inference_support()
    add_histogram_text_support()
    add_piecewise_text_support()
    add_piecewise_inference_support()
    logging.info(spn_to_str_equation(spn))
    logging.info(spn.scope)

    infer_start_t = perf_counter()

    infer_end_t = perf_counter()
    #print('Done in {}'.format(infer_end_t - infer_start_t))

    samples = []
    sample = {}

    if X_miss is None:
        X = X_train
        logging.info('Training on original train')
    else:
        logging.info('Training on train split with MISSING VALUES')
        X = X_train
        X_eval_ext = extend_matrix_one_value_per_row(X_eval)

    X_min = np.nanmin(X, axis=0)
    X_max = np.nanmax(X, axis=0)
    X_mean = np.nanmean(X, axis=0)
    # X_std = np.sqrt(np.nansum(X - X_mean, axis=0))
    X_std = np.nanstd(X, axis=0) / np.sqrt((~np.isnan(X)).sum(axis=0))

    if X_miss is not None:
        # print(X_eval_ext.shape)
        mv_lls = log_likelihood(spn, X_eval_ext, context=ds_context)
        # assert mv_lls.shape[0] == X_miss.sum()
        assert mv_lls.shape[0] == (~np.isnan(X_eval)).sum()
        logging.info('\t computed missing value LL mean: {} min: {} max:{}'.format(mv_lls.mean(),
                                                                                   mv_lls.min(),
                                                                                   mv_lls.max()))
        sample['mv-lls'] = mv_lls
        #
        # predictions
        mv_mpe_ass, mv_mpe_lls = mpe(spn, X, context=ds_context)
        X_eval_mask = np.isnan(X_eval)
        # forcing masking for those datasets with inherent missing values
        mv_mpe_ass[X_eval_mask] = np.nan
        assert_array_equal(np.isnan(X_eval), np.isnan(mv_mpe_ass))

        mv_preds = compute_predictions_dict(X_eval, mv_mpe_ass, meta_types,
                                            X_min, X_max, X_mean, X_std)
        logging.info('\t computed missing value PREDS')
        # print_perf_dict(mv_preds)

        aggr_scores = aggr_mv_score_dict(mv_preds, m='MSE')
        aggr_scores_d = aggr_mv_score_dict_meta_types(mv_preds,
                                                      m_cont='MSE',
                                                      m_disc='ACC',
                                                      meta_types=meta_types)

        sample['mv-preds'] = mv_mpe_ass
        sample['mv-preds-lls'] = mv_mpe_lls
        sample['mv-preds-scores'] = mv_preds
        sample['id'] = 0

        samples.append(sample)

        dump_samples_to_pickle(samples, out_path, out_file_name='mv-lls.pklz',
                               key='mv-lls', count_key='id')
        dump_samples_to_pickle(samples, out_path, out_file_name='mv-preds.pklz',
                               key='mv-preds', count_key='id')
        dump_samples_to_pickle(samples, out_path, out_file_name='mv-preds-lls.pklz',
                               key='mv-preds-lls', count_key='id')
        dump_samples_to_pickle(samples, out_path, out_file_name='mv-preds-scores.pklz',
                               key='mv-preds-scores', count_key='id')

    else:
        # inductive
        logging.info('\nComputing model sample likelihood:')
        for x_split, split_name in zip([X_train, X_valid, X_test],
                                       ['train', 'valid', 'test']):
            if x_split is not None:
                split_data_lls = log_likelihood(spn, x_split, context=ds_context)

                assert split_data_lls.shape[0] == x_split.shape[0]
                logging.info('\t{} data LL mean:{} min:{} max:{}'.format(split_name,
                                                                         split_data_lls.mean(),
                                                                         split_data_lls.min(),
                                                                         split_data_lls.max()))

                sample['{}-lls'.format(split_name)] = split_data_lls

        sample['id'] = 0
        samples.append(sample)

        dump_samples_to_pickle(samples, out_path, out_file_name='train-lls.pklz',
                               key='train-lls', count_key='id')
        dump_samples_to_pickle(samples, out_path, out_file_name='valid-lls.pklz',
                               key='valid-lls', count_key='id')
        dump_samples_to_pickle(samples, out_path, out_file_name='test-lls.pklz',
                               key='test-lls', count_key='id')

