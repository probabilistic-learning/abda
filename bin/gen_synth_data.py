import argparse

from spn.structure.leaves.typedleaves.Text import add_typed_leaves_text_support
from spn.structure.leaves.typedleaves.TypedLeaves import INV_TYPE_PARAM_MAP, get_type_partitioning_leaves

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
import json
from collections import defaultdict, OrderedDict

import numpy as np
import numba
import scipy.io

from spn.algorithms.Inference import log_likelihood, compute_global_type_weights
from spn.algorithms.LearningWrappers import learn_rand_spn
from spn.structure.StatisticalTypes import MetaType, Type, META_TYPE_MAP
from spn.structure.Base import Context
from spn.structure.Base import Leaf, get_nodes_by_type, assign_ids, rebuild_scopes_bottom_up

from spn.structure.leaves.parametric.Parametric import *
from spn.structure.leaves.parametric.Inference import add_parametric_inference_support
from spn.algorithms.Posteriors import *


from spn.io.Text import to_JSON, spn_to_str_equation

from spn.algorithms.Sampling import sample_instances
from spn.algorithms.Statistics import get_structure_stats_dict

from visualize import visualize_data_partition, reorder_data_partitions, visualize_histogram

TYPE_NAME_MAP = {'real': Type.REAL,
                 'int': Type.INTERVAL,
                 'pos': Type.POSITIVE,
                 'cat': Type.CATEGORICAL,
                 'ord': Type.ORDINAL,
                 'count': Type.COUNT,
                 'bin': Type.BINARY}

META_TYPE_NAME_MAP = {'real': MetaType.REAL,
                      'bin': MetaType.BINARY,
                      'disc': MetaType.DISCRETE}

META_TYPES_ISLV = {MetaType.REAL: 2,
                   MetaType.BINARY: 3,
                   MetaType.DISCRETE: 4}

PARAM_TYPE_MAP = {Type.REAL: [(Gaussian, {'mean': 5, 'stdev': 5}),
                              (Gaussian, {'mean': 100, 'stdev': 2}),
                              (Gaussian, {'mean': 30, 'stdev': 1.3}),
                              (Gaussian, {'mean': 10, 'stdev': 3}),
                              (Gaussian, {'mean': 15, 'stdev': 0.8}),
                              (Gaussian, {'mean': -45, 'stdev': 1.7}),
                              (Gaussian, {'mean': 65, 'stdev': 5}),
                              (Gaussian, {'mean': 0, 'stdev': 1}),
                              (Gaussian, {'mean': -10, 'stdev': 5})],
                  Type.COUNT: [(Geometric, {'p': 0.02}),
                               (Geometric, {'p': 0.22}),
                               (Geometric, {'p': 0.52}),
                               (Geometric, {'p': 0.89}),
                               (Poisson, {'mean': 10}),
                               (Poisson, {'mean': 6}),
                               (Poisson, {'mean': 3}),
                               (Poisson, {'mean': 15})],
                  Type.POSITIVE: [(Gamma, {'alpha': 20, 'beta': 5}),
                                  (Gamma, {'alpha': 20, 'beta': 1.2}),
                                  (Gamma, {'alpha': 20, 'beta': 3.5}),
                                  (Gamma, {'alpha': 20, 'beta': 10}),
                                  (Exponential, {'l': 2}),
                                  (Exponential, {'l': 5}),
                                  (Exponential, {'l': 15}),
                                  (Exponential, {'l': 25}), ],
                  Type.BINARY: [(Bernoulli, {'p': 0.1}),
                                (Bernoulli, {'p': 0.5}),
                                (Bernoulli, {'p': 0.9})],
                  Type.CATEGORICAL: [(Categorical, {'p': np.array([0.1, 0.2, 0.7])}),
                                     (Categorical, {'p': np.array([0.1, 0.05, 0.05, 0.8])}),
                                     (Categorical, {'p': np.array([0.1, 0.1, 0.5, 0.05, 0.05, 0.1, 0.1])})]
                  }


DEFAULT_TYPE_PARAM_MAP = {
    MetaType.REAL: OrderedDict({
        Type.REAL: OrderedDict({Gaussian: {'params': {'mean': 0, 'stdev': 1},
                                           'prior': PriorNormalInverseGamma(m_0=1, V_0=10, a_0=1, b_0=1)}}),
        Type.POSITIVE: OrderedDict({Gamma: {'params': {'alpha': None, 'beta': 1},
                                            'prior': PriorGamma(a_0=1, b_0=1)},
                                    Exponential: {'params': {'l': 2},
                                                  'prior': PriorGamma(a_0=1, b_0=1)},

                                    }),
    }),
    MetaType.DISCRETE: OrderedDict({
        Type.CATEGORICAL: {Categorical: {'params': {'p': None},
                                         'prior': PriorDirichlet(alphas_0=None)}},
        Type.COUNT: OrderedDict({Geometric: {'params': {'p': 0.5},
                                             'prior': PriorBeta(a_0=1, b_0=1)},
                                 Poisson: {'params': {'mean': 2},
                                           'prior': PriorGamma(a_0=1, b_0=1)}}),

    }),
}

DEFAULT_PRIOR_MAP = {
    'prior-1': {Gaussian: PriorNormalInverseGamma(m_0=0, V_0=30, a_0=10, b_0=10),
                Gamma: PriorGamma(a_0=10, b_0=10),
                Geometric: PriorBeta(a_0=0.5, b_0=1),
                Poisson: PriorGamma(a_0=100, b_0=10),
                Exponential: PriorGamma(a_0=20, b_0=5),
                Categorical: PriorDirichlet(alphas_0=None),
                LogNormal: PriorNormal(mu_0=0, tau_0=0.01),
                Bernoulli: PriorBeta(a_0=10, b_0=10)},
}


PARTITIONING_DIR = 'partitioning'
LLS_DIR = 'lls'
FEATURE_VIS_DIR = 'vis-features'


def generate_param_type_feature_map(feature_types, param_type_map=PARAM_TYPE_MAP):
    """
    Duplicate the possible distributions in the map for each feature sharing the correct type
    """
    param_feature_map = defaultdict(dict)

    for j, t in enumerate(feature_types):
        param_feature_map[t][j] = list(param_type_map[t])

    return param_feature_map


def estimate_global_type_weights_from_partitioning(type_P, inv_type_map=INV_TYPE_PARAM_MAP):

    N, D = type_P.shape

    est_W = {}
    for d in range(D):
        param_ids, param_counts = np.unique(type_P[:, d], return_counts=True)
        assert param_counts.sum() == N, param_counts
        param_ws = param_counts / N

        est_W[d] = {inv_type_map[p_id]: p_w for p_id, p_w in zip(param_ids,
                                                                 param_ws)}
    return est_W


if __name__ == '__main__':

    #########################################
    # creating the opt parser
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str,
                        help='Dataset name')

    parser.add_argument('-o', '--output', type=str,
                        default='./data/synth/',
                        help='Output path to dataset generation result')

    parser.add_argument('-N', '--samples', type=int,
                        help='Number of samples (rows) for the matrix')

    parser.add_argument('-D', '--features', type=int, nargs='?',
                        default=None,
                        help='Number of features (cols) for the matrix. Optional')

    parser.add_argument('-t', '--types', type=str, nargs='+',
                        default=None,
                        help='Sequence of feature types (optional)')

    parser.add_argument('-m', '--meta-types', type=str, nargs='+',
                        default=None,
                        help='Sequence of feature meta-types (optional)')

    parser.add_argument('--miss-perc', type=float,
                        default=.2,
                        help='Percentage missing values in the training set to generate')

    parser.add_argument('--min-instances', type=int,
                        default=None,
                        help='Min number of samples (rows) to stop splitting')

    parser.add_argument('--beta-rows', type=float, nargs='+',
                        default=(2, 5),
                        help='Beta a, b parameters to draw a percentage to split on rows')

    parser.add_argument('--beta-cols', type=float, nargs='+',
                        default=(4, 5),
                        help='Beta a, b parameters to draw a percentage to split on cols')

    parser.add_argument('--tvt-split', type=float, nargs='+',
                        default=[0.7, 0.1, 0.2],
                        help='Train, validation and test split percentages')

    parser.add_argument('--col-split-threshold', type=float,
                        default=0.4,
                        help='Success rate percentage to randomly split on columns')

    parser.add_argument('--priors', type=str,
                        default='prior-1',
                        help='Prior data set')

    parser.add_argument('--seed', type=int, nargs='?',
                        default=1337,
                        help='Seed for the random generator')

    parser.add_argument('--exp-id', type=str,
                        default=None,
                        help='Dataset output suffix')

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
    date_string = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    dataset_name = args.dataset

    if args.exp_id:
        out_path = os.path.join(args.output, dataset_name, args.exp_id,)
    else:
        out_path = os.path.join(args.output, dataset_name, date_string)
    os.makedirs(out_path, exist_ok=True)

    part_out_path = os.path.join(out_path, PARTITIONING_DIR)
    os.makedirs(part_out_path, exist_ok=True)
    fvis_out_path = os.path.join(out_path, FEATURE_VIS_DIR)
    os.makedirs(fvis_out_path, exist_ok=True)
    lls_out_path = os.path.join(out_path, LLS_DIR)
    os.makedirs(lls_out_path, exist_ok=True)

    args_out_path = os.path.join(out_path, 'args.json')
    json_args = json.dumps(vars(args))
    logging.info("Starting with arguments:\n%s\n\tdumped at %s", json_args, args_out_path)
    with open(args_out_path, 'w') as f:
        f.write(json_args)

    #
    # setting up the seed
    rand_gen = np.random.RandomState(args.seed)

    #
    # type, meta_type selection
    #
    assert any([args.features,
                args.types,
                args.meta_types]), 'At least one among D, type, meta-types must be specified'

    N = args.samples
    D = None
    types = None
    meta_types = None
    if args.features is None:
        if args.types:
            D = len(args.types)
        elif args.meta_types:
            D = len(args.meta_types)
    else:
        D = args.features

    if args.types:
        types = np.array([TYPE_NAME_MAP[t] for t in args.types])
        #
        # inferring them from types
        if not args.meta_types:
            meta_types = np.array([t.meta_type for t in types])

    elif args.meta_types:
        meta_types = np.array([META_TYPE_NAME_MAP[m] for m in args.meta_types])
        #
        # randomly selecting them
        types = []
        for m in meta_types:
            types.append(rand_gen.choice(META_TYPE_MAP[m]))
        types = np.array(types)

    logging.info('Defined for D={} features:\n\ttypes:\t{}\n\tmeta-types:\t{}'.format(D, types,
                                                                                      meta_types))

    #
    # create fake data for randomized structure learning
    data = rand_gen.normal(loc=0, scale=1, size=(N, D))

    #
    # filling up a context object with meta info
    ds_context = Context(meta_types=meta_types)
    ds_context.types = types
    ds_context.priors = DEFAULT_PRIOR_MAP[args.priors]

    #
    # TODO: generating a random map for parametric forms
    # type_param_map = generate_param_type_feature_map(types, PARAM_TYPE_MAP)
    type_param_map = DEFAULT_TYPE_PARAM_MAP
    print('\n\n\nGEN PARAM TYPE MAP\n\n')
    ds_context.param_form_map = type_param_map

    add_parametric_inference_support()
    add_typed_leaves_text_support()

    #
    # RANDOM STRUCTURE LEARNING
    learn_start_t = perf_counter()
    spn = learn_rand_spn(data,
                         ds_context,
                         min_instances_slice=args.min_instances,
                         row_a=args.beta_rows[0], row_b=args.beta_rows[1],
                         col_a=args.beta_cols[0], col_b=args.beta_cols[1],
                         col_threshold=args.col_split_threshold,
                         memory=None,
                         rand_gen=rand_gen)

    rebuild_scopes_bottom_up(spn)
    assign_ids(spn)
    learn_end_t = perf_counter()

    stats = get_structure_stats_dict(spn)
    logging.info('\n\nLearned spn in {} with stats:\n\t{}'.format(learn_end_t - learn_start_t,
                                                                  stats))

    print(spn_to_str_equation(spn))
    print(spn.scope)

    #
    # storing the spn on file
    spn_output_path = os.path.join(out_path, 'spn.model.pkl')
    store_start_t = perf_counter()
    with open(spn_output_path, 'wb') as f:
        pickle.dump(spn, f)
    store_end_t = perf_counter()
    logging.info('Stored spn to {} (in {} secs)'.format(spn_output_path,
                                                        store_end_t - store_start_t))

    #
    # actual sampling, generating the data
    # returning a partition matrix (discarding the Zs?)
    sample_start_t = perf_counter()
    X, _Z, P = sample_instances(spn,
                                D, N,
                                rand_gen,
                                return_Zs=True,
                                return_partition=True,
                                dtype=np.float64)
    sample_end_t = perf_counter()

    assert X.shape == (N, D), X.shape
    assert X.shape == P.shape, X.shape
    logging.info('\nDrawn {} samples form the SPN'.format(N))

    type_P = get_type_partitioning_leaves(spn, P)

    est_full_W = estimate_global_type_weights_from_partitioning(type_P,
                                                                inv_type_map=INV_TYPE_PARAM_MAP)
    logging.info('\nEstimated global type W:\n\t{}\n'.format(est_full_W))

    #
    # storing full data on file
    full_data_path = os.path.join(out_path, 'full.data')
    with open(full_data_path, 'wb') as f:
        pickle.dump(X, f)
    logging.info('Saved full X data to {}'.format(full_data_path))

    full_part_path = os.path.join(part_out_path, 'full.partition')
    with open(full_part_path, 'wb') as f:
        pickle.dump(P, f)
    logging.info('Saved full P partition to {}'.format(full_part_path))

    full_type_part_path = os.path.join(part_out_path, 'full.type.partition')
    with open(full_type_part_path, 'wb') as f:
        pickle.dump(type_P, f)
    logging.info('Saved full type P partition to {}'.format(full_type_part_path))

    #
    # TODO: additional preprocessing? (like making all values start from 1? like for ISVL)

    #
    # visualizing single features
    for d in range(D):
        f_hist_out_path = os.path.join(fvis_out_path, 'd{}.hist'.format(d))
        visualize_histogram(X[:, d], show_fig=args.show_plots,
                            output=f_hist_out_path)

    #
    # visualizing the leaf partitioning
    inv_leaf_map = {l.id: spn_to_str_equation(l) + " id: " + str(l.id)  # l.__class__.__name__
                    for l in get_nodes_by_type(spn, Leaf)}
    vis_part_out_path = os.path.join(part_out_path, 'samples.partitioning.full')
    title_str = "{} samples from spn with {} sums {} prods {} leaves".format(N,
                                                                             stats['sum'],
                                                                             stats['prod'],
                                                                             stats['leaf'])
    visualize_data_partition(P, color_map_ids=inv_leaf_map,
                             title=title_str, output=vis_part_out_path, show_fig=args.show_plots)

    #
    # vis type partitioning
    vis_type_part_out_path = os.path.join(part_out_path, 'samples.type.partitioning.full')
    title_str = "types for {} samples from spn with {} sums {} prods {} leaves".format(N,
                                                                                       stats['sum'],
                                                                                       stats['prod'],
                                                                                       stats['leaf'])
    visualize_data_partition(type_P,
                             color_map_ids=INV_TYPE_PARAM_MAP,
                             title=title_str,
                             output=vis_type_part_out_path, show_fig=args.show_plots)

    #
    # reordering it
    reord_ids = reorder_data_partitions(P)
    ord_vis_part_out_path = os.path.join(part_out_path, 'ordered.samples.partitioning.full')
    title_str = "ordered {} samples from spn with {} sums {} prods {} leaves".format(N,
                                                                                     stats['sum'],
                                                                                     stats['prod'],
                                                                                     stats['leaf'])
    visualize_data_partition(P[reord_ids], color_map_ids=inv_leaf_map,
                             title=title_str, output=ord_vis_part_out_path, show_fig=args.show_plots)

    ord_vis_type_part_out_path = os.path.join(part_out_path,
                                              'ordered.samples.type.partitioning.full')
    title_str = "ordered types for {} samples from spn with {} sums {} prods {} leaves".format(N,
                                                                                               stats['sum'],
                                                                                               stats['prod'],
                                                                                               stats['leaf'])
    visualize_data_partition(type_P[reord_ids],
                             color_map_ids=INV_TYPE_PARAM_MAP,
                             title=title_str,
                             output=ord_vis_type_part_out_path, show_fig=args.show_plots)

    #
    # splitting it into train, validation, test
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

    P_train = P[train_ids]
    P_valid = P[valid_ids]
    P_test = P[test_ids]

    PT_train = type_P[train_ids]
    PT_valid = type_P[valid_ids]
    PT_test = type_P[test_ids]

    #
    # generating missing values?
    if args.miss_perc:
        X_miss = rand_gen.binomial(p=args.miss_perc, n=1, size=X_train.shape).astype(bool)
        miss_data_path = os.path.join(out_path, 'miss.train.data')
        with open(miss_data_path, 'wb') as f:
            pickle.dump(X_miss, f)
        logging.info('Saved miss train X data to {}'.format(miss_data_path))

    est_train_W = estimate_global_type_weights_from_partitioning(PT_train,
                                                                 inv_type_map=INV_TYPE_PARAM_MAP)
    logging.info('\nEstimated train global type W:\n\t{}\n'.format(est_train_W))
    est_valid_W = estimate_global_type_weights_from_partitioning(PT_valid,
                                                                 inv_type_map=INV_TYPE_PARAM_MAP)
    logging.info('\nEstimated valid global type W:\n\t{}\n'.format(est_valid_W))
    est_test_W = estimate_global_type_weights_from_partitioning(PT_test,
                                                                inv_type_map=INV_TYPE_PARAM_MAP)
    logging.info('\nEstimated test global type W:\n\t{}\n'.format(est_test_W))

    #
    # extracting statistics from it
    min_X = np.array([np.min(X[:, d]) for d in range(X.shape[1])])
    logging.info('Min X: {}'.format(min_X))
    max_X = np.array([np.max(X[:, d]) for d in range(X.shape[1])])
    logging.info('Max X: {}'.format(max_X))
    domains = [np.unique(X[:, d]) for d in range(X.shape[1])]
    logging.info('Domains X: {}'.format(domains))
    #
    # FIXME
    global_W = compute_global_type_weights(spn, aggr_type=False)
    # global_W = None
    global_type_W = compute_global_type_weights(spn, aggr_type=True)
    # global_type_W = None
    logging.info('global W: {}'.format(global_W))
    logging.info('global type aggr W: {}'.format(global_type_W))

    data_stats = {'min': min_X, 'max': max_X, 'domains': domains,
                  'types': types, 'meta-types': meta_types,
                  'spn-W': global_W,
                  'spn-type-W': global_type_W,
                  'full-type-W': est_full_W,
                  'train-type-W': est_train_W,
                  'valid-type-W': est_valid_W,
                  'test-type-W': est_test_W, }
    data_stats_path = os.path.join(out_path, 'data.stats')
    with open(data_stats_path, 'wb') as f:
        pickle.dump(data_stats, f)

    #
    # accomodate for Categorical, "enlarging" their support
    for l in get_nodes_by_type(spn, Parametric):
        if isinstance(l, Categorical):
            old_k = l.k
            print('k', l.k, 'p', len(l.p))
            new_k = max(int(max_X[l.scope[0]]) + 1, old_k)
            enlarged_p = np.zeros(new_k, dtype=l.p.dtype)
            enlarged_p[:old_k] = l.p
            l.p = enlarged_p

    #
    # saving into ISLV format
    types_islv = np.array([[META_TYPES_ISLV[t] for t in meta_types]]).astype(np.float64)
    X_islv = np.array(X_train)
    disc_types = meta_types == MetaType.DISCRETE
    print(disc_types)
    X_islv[:, disc_types] = X_train[:, disc_types] + 1
    islv_data = {'N': N, 'D': D,
                 'X': X_islv,
                 'R': (max_X + 1).astype(np.float64).reshape(1, -1),
                 'T': types_islv}
    islv_out_path = os.path.join(out_path, 'train.data.islv')
    scipy.io.savemat(islv_out_path, islv_data)
    logging.info('Saved data to ISLV format to {}'.format(islv_out_path))
    if args.miss_perc:
        islv_X_miss_ids, = np.nonzero(X_miss.flatten(order='F'))
        islv_X_miss = (islv_X_miss_ids + 1)
        assert len(islv_X_miss_ids) == X_miss.sum()
        miss_islv_data = {'miss': islv_X_miss}
        miss_islv_out_path = os.path.join(out_path, 'train.data.islvMiss')
        scipy.io.savemat(miss_islv_out_path, miss_islv_data)
        logging.info('Saved miss data to ISLV format to {}'.format(miss_islv_out_path))

    #
    # visualizing partitions and saving data to disk
    for x_split, p_split, t_split, split_name in zip([X_train, X_valid, X_test],
                                                     [P_train, P_valid, P_test],
                                                     [PT_train, PT_valid, PT_test],
                                                     ['train', 'valid', 'test']):
        vis_part_out_path = os.path.join(
            part_out_path, 'samples.partitioning.{}'.format(split_name))
        title_str = "{} {} samples from spn with {} sums {} prods {} leaves".format(x_split.shape[0],
                                                                                    split_name,
                                                                                    stats['sum'],
                                                                                    stats['prod'],
                                                                                    stats['leaf'])
        visualize_data_partition(p_split,
                                 color_map_ids=inv_leaf_map,
                                 title=title_str,
                                 output=vis_part_out_path, show_fig=args.show_plots)

        vis_type_out_path = os.path.join(part_out_path,
                                         'samples.type.partitioning.{}'.format(split_name))
        title_str = "types for {} {} samples from spn with {} sums {} prods {} leaves".format(x_split.shape[0],
                                                                                              split_name,
                                                                                              stats['sum'],
                                                                                              stats['prod'],
                                                                                              stats['leaf'])
        visualize_data_partition(t_split,
                                 color_map_ids=INV_TYPE_PARAM_MAP,
                                 title=title_str,
                                 output=vis_type_out_path, show_fig=args.show_plots)

        #
        # reordering it
        reord_ids = reorder_data_partitions(p_split)
        ord_vis_part_out_path = os.path.join(part_out_path,
                                             'ordered.samples.partitioning.{}'.format(split_name))
        title_str = "ordered {} {} samples from spn with {} sums {} prods {} leaves".format(x_split.shape[0],
                                                                                            split_name,
                                                                                            stats['sum'],
                                                                                            stats['prod'],
                                                                                            stats['leaf'])
        visualize_data_partition(p_split[reord_ids], color_map_ids=inv_leaf_map,
                                 title=title_str,
                                 output=ord_vis_part_out_path, show_fig=args.show_plots)

        ord_vis_type_out_path = os.path.join(part_out_path,
                                             'ordered.samples.type.partitioning.{}'.format(split_name))
        title_str = "ordered types for {} {} samples from spn with {} sums {} prods {} leaves".format(x_split.shape[0],
                                                                                                      split_name,
                                                                                                      stats['sum'],
                                                                                                      stats['prod'],
                                                                                                      stats['leaf'])
        visualize_data_partition(t_split[reord_ids],
                                 color_map_ids=INV_TYPE_PARAM_MAP,
                                 title=title_str,
                                 output=ord_vis_type_out_path, show_fig=args.show_plots)

        split_data_path = os.path.join(out_path, '{}.data'.format(split_name))
        with open(split_data_path, 'wb') as f:
            pickle.dump(x_split, f)
        logging.info('Saved {} X data to {}'.format(split_name, split_data_path))

        split_part_path = os.path.join(part_out_path, '{}.partition'.format(split_name))
        with open(split_part_path, 'wb') as f:
            pickle.dump(p_split, f)
        logging.info('Saved {} P partition to {}'.format(split_name, split_part_path))

        type_split_part_path = os.path.join(part_out_path, '{}.type.partition'.format(split_name))
        with open(type_split_part_path, 'wb') as f:
            pickle.dump(t_split, f)
        logging.info('Saved {} T partition to {}'.format(split_name, type_split_part_path))

    #
    # likelihood evaluation

    full_data_lls = log_likelihood(spn, X)
    logging.info('\n\tFull data LL mean:{} min:{} max:{}'.format(full_data_lls.mean(),
                                                                 full_data_lls.min(),
                                                                 full_data_lls.max()))
    full_lls_path = os.path.join(lls_out_path, 'full.lls')
    np.save(full_lls_path, full_data_lls)
    logging.info('Saved lls on full data to {}'.format(full_lls_path))

    for x_split, split_name in zip([X_train, X_valid, X_test],
                                   ['train', 'valid', 'test']):
        split_data_lls = log_likelihood(spn, x_split)
        logging.info('\t{} data LL mean:{} min:{} max:{}'.format(split_name,
                                                                 split_data_lls.mean(),
                                                                 split_data_lls.min(),
                                                                 split_data_lls.max()))

        split_lls_path = os.path.join(lls_out_path, '{}.lls'.format(split_name))
        np.save(split_lls_path, split_data_lls)
        logging.info('Saved lls on {} data to {}'.format(split_name, split_lls_path))
