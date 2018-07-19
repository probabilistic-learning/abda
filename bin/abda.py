
"""
Automatic and Tractable Density Estimation with Bayesian Sum-Product Networks

MODEL-HA1

@author: antonio vergari

--------

"""

import argparse


from spn.structure.leaves.typedleaves.TypedLeaves import TypeLeaf, get_type_partitioning_leaves, TypeMixture, INV_TYPE_PARAM_MAP
from spn.structure.leaves.typedleaves.Text import add_typed_leaves_text_support
import functools
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
from collections import defaultdict, Counter, OrderedDict

import numpy as np
from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_array_equal
import scipy.stats
from spn.structure.Base import Context, Sum

from spn.algorithms.Inference import compute_global_type_weights, log_likelihood, mpe
from spn.algorithms.Posteriors import update_parametric_parameters_posterior, PriorDirichlet, PriorGamma, \
    PriorNormal, PriorNormalInverseGamma, PriorBeta
from spn.algorithms.Sampling import init_spn_sampling, sample_induced_trees, sample_spn_weights, \
    validate_row_partitioning, sample_Ws
from spn.structure.Base import Leaf, get_nodes_by_type, Sum, assign_ids, rebuild_scopes_bottom_up, max_node_id
from spn.structure.StatisticalTypes import MetaType, Type
from spn.structure.leaves.parametric.Inference import add_parametric_inference_support
from spn.structure.leaves.parametric.Parametric import *
from spn.io.Text import spn_to_str_equation
from visualize import plot_distributions_fitting_data, visualize_data_partition, reorder_data_partitions, plot_mixture_components_fitting_data, plot_mixtures_fitting_multilevel
from spn.algorithms.Statistics import get_structure_stats_dict
from spn.algorithms.LearningWrappers import learn_hspn
from explain import print_perf_dict
from utils import running_avg_numba

#
# TODOD: change it to be nana
MISS_VALUE = -1

RANDOM_SEED = 17

OMEGA_UNINF_PRIOR = 10

LEAF_OMEGA_UNINF_PRIOR = 0.1

CAT_UNIF_PRIOR = 1

W_UNINF_PRIOR = 100

#
# TODO: fill this when more parametric forms are available
DEFAULT_TYPE_PARAM_MAP = {
    'spicky-prior-1': {MetaType.REAL: OrderedDict({
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

        })},

    'wider-prior-1': {MetaType.REAL: OrderedDict({
        Type.REAL: OrderedDict({Gaussian: {'params': {'mean': 0, 'stdev': 1},
                                           'prior': PriorNormalInverseGamma(m_0=1, V_0=10, a_0=10, b_0=10)}}),
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

        })},

    'wider-prior-2': {MetaType.REAL: OrderedDict({
        Type.REAL: OrderedDict({Gaussian: {'params': {'mean': 0, 'stdev': 1},
                                           'prior': PriorNormalInverseGamma(m_0=1, V_0=10, a_0=10, b_0=10)}}),
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
                                                 'prior': PriorBeta(a_0=2, b_0=2)},
                                     Poisson: {'params': {'mean': 2},
                                               'prior': PriorGamma(a_0=20, b_0=2)}}),

        })}

}

DEFAULT_PRIOR_MAP = {
    'prior-1': {Gaussian: PriorNormalInverseGamma(m_0=1, V_0=10, a_0=1, b_0=1),
                Gamma: PriorGamma(a_0=1, b_0=1),
                Geometric: PriorBeta(a_0=1, b_0=1),
                Poisson: PriorGamma(a_0=1, b_0=1),
                Categorical: PriorDirichlet(alphas_0=None),
                LogNormal: PriorNormal(mu_0=1, tau_0=1),
                Bernoulli: PriorBeta(a_0=1, b_0=1)},
}

# def approximate_density(dist_node, bins=100):

#     if dist_node.meta_type == MetaType.DISCRETE:
#         k = dist_node.k
#         x = np.arange(k)
#     elif dist_node.meta_type == MetaType.REAL:
#         x = np.linspace(dist_node.ppf(0.01),
#                         dist_node.ppf(0.99), bins)
#     # y = dist_node.p(x, **dist_node.params)
#     y = dist_node.p(x.reshape(x.shape[0], -1))
#     return x, y


# def plot_distributions_fitting_data(data, dist_nodes, bins=100,
#                                     show_fig=False, save_fig=None, cmap=None):

#     import matplotlib
#     if not show_fig:
#         matplotlib.use('Agg')
#     import matplotlib.pyplot as plt
#     from matplotlib.backends.backend_pdf import PdfPages

#     n_nodes = len(dist_nodes)
#     max_id = np.max([l.id for l in dist_nodes])

#     fig, ax = plt.subplots(1, 1)
#     area = 0
#     l_hists = [None for l in dist_nodes]
#     l_bins = [None for l in dist_nodes]
#     for j, l in enumerate(dist_nodes):
#         pdf_x, pdf_y = approximate_density(l, bins=bins)
#         if l.meta_type == MetaType.DISCRETE:
#             ax.bar(pdf_x, pdf_y[:, 0], label="leaf {}: {}".format(l.id,
#                                                                   l.name),
#                    color=plt.cm.jet(l.id / max_id))
#         elif l.meta_type == MetaType.REAL:
#             ax.plot(pdf_x, pdf_y, label="leaf {}: {}".format(l.id,
#                                                              l.name),
#                     color=plt.cm.jet(l.id / max_id))
#         #
#         # drawing also the data, coloured by the membership
#         if len(l.row_ids) > 0:
#             hist, _bins = np.histogram(data[l.row_ids, :], bins=bins)
#             area += (np.diff(_bins) * hist).sum()
#             l_hists[j] = hist
#             l_bins[j] = _bins
#     for j, l in enumerate(dist_nodes):
#         if len(l.row_ids) > 0:
#             l_hists[j] = l_hists[j] / area
#             ax.bar(l_bins[j][:-1] + np.diff(l_bins[j]) / 2,
#                    l_hists[j], align='center',
#                    #color=plt.cm.jet((j + 1) / n_nodes)
#                    color=plt.cm.jet(l.id / max_id)
#                    )

#     ax.legend()

#     if show_fig:
#         plt.show()

#     if save_fig:
#         pp = PdfPages(save_fig)
#         pp.savefig(fig)
#         pp.close()

#     plt.close()


# def plot_likelihoods(ll_hist, leaves_ll_hist,
#                      param_leaves_ll_hist, ll_counts,
#                      fig_size=(12, 8),
#                      show_fig=False, save_fig=None, cmap=None):
#     import matplotlib
#     if not show_fig:
#         matplotlib.use('Agg')
#     import matplotlib.pyplot as plt
#     from matplotlib.backends.backend_pdf import PdfPages

#     n_type_leaves = len(leaves_ll_hist)
#     n_param_leaves = len(param_leaves_ll_hist)
#     max_id = np.max(list(leaves_ll_hist.keys()) + list(param_leaves_ll_hist.keys()))

#     fig, ax = plt.subplots(figsize=fig_size)
#     ax.plot(ll_counts, ll_hist, label='avg SPN LL', alpha=0.6)

#     for j, (l_id, leaf_ll_hist) in enumerate(leaves_ll_hist.items()):
#         ax.plot(ll_counts, leaf_ll_hist, label='avg type leaf {} LL'.format(l_id), alpha=0.6,
#                 # color=plt.cm.jet((j + 1) / n_type_leaves)
#                 color=plt.cm.jet(l_id / max_id)
#                 )

#     for j, (pl_id, param_leaf_ll_hist) in enumerate(param_leaves_ll_hist.items()):
#         ax.plot(ll_counts, param_leaf_ll_hist, label='avg param leaf {} LL'.format(pl_id), alpha=0.6,
#                 # color=plt.cm.jet((j + 1) / n_param_leaves)
#                 color=plt.cm.jet(pl_id / max_id))
#     # ax.plot(hll_hist, label='avg HLL')
#     ax.legend()

#     if show_fig:
#         plt.show()

#     if save_fig:
#         pp = PdfPages(save_fig + 'll-history.pdf')
#         pp.savefig(fig)
#         pp.close()

def dump_samples_to_pickle(samples, output_path, out_file_name, key, count_key='id'):
    pic_path = os.path.join(output_path, out_file_name)
    counts = np.array([s[count_key] for s in samples if key in s])
    k_hist = [s[key] for s in samples if key in s]
    with gzip.open(pic_path, 'wb') as f:
        res = {count_key: counts,
               key: k_hist}
        pickle.dump(res, f)
        print('Dumped {} to {}'.format(count_key, pic_path))


def dump_best_sample_to_pickle(samples, output_path, out_file_name,
                               key, best_id, count_key='id'):
    pic_path = os.path.join(output_path, out_file_name)
    best_sample = [s for s in samples if s[count_key] == best_id]
    assert len(best_sample) == 1
    best_sample = best_sample[0]
    with gzip.open(pic_path, 'wb') as f:
        res = {count_key: best_sample[count_key],
               key: best_sample[key]}
        pickle.dump(res, f)
        print('Dumped best {} to {}'.format(count_key, pic_path))


ISLV_CODE_MAP = {1: MetaType.REAL,
                 2: MetaType.REAL,
                 3: MetaType.BINARY,
                 4: MetaType.DISCRETE}


def islv_type_codes_to_meta_types(codes):
    return np.array([ISLV_CODE_MAP[c] for c in codes])


def islv_cardinalities_to_domains(data, cards, meta_types):
    domains = []
    for i, d in enumerate(cards):
        if meta_types[i] == MetaType.DISCRETE:
            domains.append(np.arange(d))
        elif meta_types[i] == MetaType.BINARY:
            domains.append(np.array([2]))
        elif meta_types[i] == MetaType.REAL:
            domains.append(np.array([np.min(data[:, i]), np.max(data[:, i])]))
        else:
            raise ValueError('Unrecognized MetaType {}'.format(meta_types[i]))
    return domains


def translating_discrete_data(X, meta_types):
    N, D = X.shape
    in_mv = ((X == -1) | np.isnan(X))
    for d in range(D):
        if meta_types[d] == MetaType.DISCRETE:
            for n in range(N):
                if ~in_mv[n, d]:
                    X[n, d] = X[n, d] - 1
            # X[:, d] = X[:, d] - 1
            assert np.all(X[:, d][~in_mv[:, d]] >= 0)

    return X


def load_islv_mat(data_path, return_orig=False):

    import scipy.io

    data_dict = scipy.io.loadmat(data_path)
    print('\nLoaded {}'.format(data_path))

    X_orig = data_dict['X']
    X = np.copy(X_orig).astype(np.float64)
    print('\twith shape: {}'.format(X.shape))

    C = data_dict['T'].flatten().astype(np.int64)
    print('\tmeta-types:{}'.format(C))

    R = data_dict['R'].flatten().astype(np.int64)
    print('\tmaximal discrete cardinality: {}'.format(R))

    meta_types = islv_type_codes_to_meta_types(C)
    print('\tmeta types', meta_types)

    domains = islv_cardinalities_to_domains(X, R, meta_types)
    print('\tdomains', domains)

    X = translating_discrete_data(X, meta_types)
    print('\ttranslated discrete features (starting from 0)')

    if return_orig:
        return X, meta_types, domains, X_orig, C, R

    return X.astype(np.float64), meta_types, domains


def load_islv_miss(data_path, shape):

    import scipy.io

    data_dict = scipy.io.loadmat(data_path)
    print('\nLoaded missing data from {}'.format(data_path))

    X_miss_ids = data_dict['miss']
    X_miss_ids = np.unravel_index(X_miss_ids.flatten().astype(np.int64) - 1,
                                  shape, order='F')
    X_miss = np.zeros(shape, dtype=bool)
    X_miss[X_miss_ids] = True
    return X_miss


def preprocess_positive_real_data(X, meta_types, eps=1e-6):
    """
    Adding a small positive epsilon to continuous data to avoid
    numerical inaccuracies
    """

    D = X.shape[1]

    assert meta_types.shape[0] == D

    for d in range(D):
        # to avoid numerical errors in positive real data
        if meta_types[d] == MetaType.REAL:
            X[np.isclose(X[:, d], 0), d] = eps


def init_weights(spn, meta_types, alpha_prior=100, rand_gen=None):
    """
    Initing W matrix according to type.
    NOTE: cannot put Dirichlet hyperparameters to 0 (strangely it worked on Isabel's code)

    input:
    T : uint array
    meta-type array (size D), encoding
    {
      1: real (w positive: all real | positive | interval)
      2: real (w/o positive: all real | interval)
      3: binary data
      4: discrete (non-binary: categorical | ordinal | count)
    }

    output:
    W : float matrix
        weight array (size Dx3 where 3 is the number of types per meta-types)
    """

    mixture_leaves = get_nodes_by_type(spn, TypeMixture)
    # first check that we have the same meta type for every children of the mixture in that dimension
    # and check that all the types are the same and in the same order
    leaf_classtype_D = {}
    for mt in mixture_leaves:
        d = mt.scope[0]
        meta_type = meta_types[d]
        for c in mt.children:
            assert meta_type == c.type.meta_type, 'parent meta-type: {} child meta-type: {}'.format(meta_type,
                                                                                                    c.type.meta_type)

        if d not in leaf_classtype_D:
            leaf_classtype_D[d] = [type(c) for c in mt.children]  # same class type

        assert len(leaf_classtype_D[d]) == len(mt.children)

        for i, t in enumerate(leaf_classtype_D[d]):
            assert type(mt.children[i]) == t

    W_priors = {}
    W_counts = {}
    W = {}
    for d, children_classes in leaf_classtype_D.items():
        if meta_types[d] == MetaType.BINARY:
            continue
        n_types = len(children_classes)
        W_priors[d] = np.array([alpha_prior for i in range(n_types)])
        W_counts[d] = np.zeros((n_types))  # do from spn?
        W[d] = np.zeros((n_types))

    for mt in mixture_leaves:
        mt.weights = W[mt.scope[0]]

    sample_Ws(W, meta_types, W_counts, W_priors, rand_gen)

    return W, W_priors


def retrieve_best_sample(samples, key='valid-lls'):
    """
    Retrieve the sample with the highest log-likelihood associated from a list of samples.
    Each sample is a dict (see sp_infer_data_types_ha1)

    """
    avg_lls = [s[key].mean() for s in samples]
    best_sample_id = np.argmax(avg_lls)
    return samples[best_sample_id]


def extend_matrix_one_value_per_row(X):
    N, D = X.shape

    ext_X = []
    for n in range(N):
        for d in range(D):
            x_n_d = X[n, d]
            if not np.isnan(x_n_d):
                r = np.zeros(D, dtype=X.dtype)
                r[:] = np.nan
                r[d] = x_n_d
                ext_X.append(r)

    ext_X = np.array(ext_X)
    N_nans = (~np.isnan(X)).sum()

    assert ext_X.shape[0] == N_nans
    assert ext_X.shape[1] == D
    return ext_X


from sklearn.metrics import mean_squared_error, mean_squared_log_error, mean_absolute_error, zero_one_loss

SCORE_MAP = {MetaType.REAL: mean_squared_error,
             MetaType.DISCRETE: mean_squared_error}


# def root_mean_squared_error(X_true, X_pred):
#     return np.sqrt(mean_squared_error(X_true, X_pred))


# def mean_norm_error(err_f, X_true, X_pred):
#     X_true_mean = X_true.mean()
#     return err_f(X_true, X_pred) / X_true_mean


# def std_err_norm_error(err_f, X_true, X_pred):
#     X_stddev = X_true.std()
#     X_std_err = X_stddev / np.sqrt(len(X_true))
#     return err_f(X_true, X_pred) / X_std_err


# def range_norm_error(err_f, X_true, X_pred):
#     X_range = X_true.max() - X_true.min()
#     return err_f(X_true, X_pred) / X_range


# mean_norm_root_mean_squared_error = functools.partial(mean_norm_error,
#                                                       root_mean_squared_error)
# std_err_norm_root_mean_squared_error = functools.partial(std_err_norm_error,
#                                                          root_mean_squared_error)
# range_norm_root_mean_squared_error = functools.partial(range_norm_error,
#                                                        root_mean_squared_error)

# METRICS_DICT = {'RMSE': root_mean_squared_error,
#                 'M-RMSE': mean_norm_root_mean_squared_error,
#                 'R-RMSE': range_norm_root_mean_squared_error,
#                 'S-RMSE': std_err_norm_root_mean_squared_error,
#                 'MSE': mean_squared_error,
#                 'MSLE': mean_squared_log_error,
#                 'MAE': mean_absolute_error,
#                 'ACC': zero_one_loss}

# SCORE_LIST_MAP = {MetaType.REAL: ['MSE', 'MAE', 'RMSE', 'M-RMSE', 'R-RMSE', 'S-RMSE'],
#                   MetaType.DISCRETE: ['MSE', 'MAE', 'ACC', 'MSLE', 'RMSE', 'M-RMSE', 'R-RMSE', 'S-RMSE']}

from explain import SCORE_LIST_MAP, METRICS_DICT


def compute_predictions(X_true, X_pred, meta_types, scores=SCORE_MAP):
    N, D = X_true.shape
    assert X_pred.shape[0] == N, X_pred.shape
    assert X_pred.shape[1] == D, X_pred.shape

    preds = np.zeros((N, D), dtype=X_true.dtype)
    for n in range(N):
        for d in range(D):
            x_true = X_true[n, d]
            x_pred = X_pred[n, d]
            if not np.isnan(x_true):
                assert not np.isnan(x_pred), x_pred
                preds[n, d] = scores[meta_types[d]](x_true, x_pred)

    return preds


def compute_predictions_dict(X_true, X_pred, meta_types,
                             X_min, X_max, X_mean, X_std,
                             #, scores=SCORE_MAP
                             ):
    N, D = X_true.shape
    assert X_pred.shape[0] == N, X_pred.shape
    assert X_pred.shape[1] == D, X_pred.shape

    perf_dict = defaultdict(dict)

    miss_vals = np.any(np.isnan(X_true))

    for d in range(D):

        sids = None
        #
        # dealing with missing values?
        if miss_vals:
            sids = ~np.isnan(X_true[:, d])
        else:
            sids = np.arange(X_true.shape[0])

        # print('SIDS', d, sids, sids.sum())

        if meta_types[d] == MetaType.DISCRETE:
            true_vals = X_true.astype(int)
            pred_vals = X_pred.astype(int)
        else:
            true_vals = X_true
            pred_vals = X_pred

        for m in SCORE_LIST_MAP[meta_types[d]]:

            score_func = METRICS_DICT[m]
            # print(X_true[sids, d], X_pred[sids, d])
            try:
                s = score_func(true_vals[sids, d], pred_vals[sids, d],
                               X_min[d], X_max[d], X_mean[d], X_std[d])
                perf_dict[d][m] = s
            except:
                pass

    return perf_dict


def compute_predictions_dict_miss(X_true, X_pred, X_mask, meta_types,
                                  X_min, X_max, X_mean, X_std,
                                  # scores=SCORE_MAP
                                  ):
    N, D = X_true.shape
    assert X_pred.shape[0] == N, X_pred.shape
    assert X_pred.shape[1] == D, X_pred.shape
    assert X_min.shape[0] == D
    assert X_max.shape[0] == D
    assert X_mean.shape[0] == D
    assert X_std.shape[0] == D

    perf_dict = defaultdict(dict)

    for d in range(D):

        sids = None
        #
        # dealing with missing values?

        sids = ~np.isnan(X_true[:, d]) & X_mask[:, d]

        if sids.sum() == 0:
            print('SIDS', d, sids, sids.sum(), (~np.isnan(
                X_true[:, d])).sum(), X_mask[:, d].sum(), N, D)
            print('zero')
            0 / 0

        for m in SCORE_LIST_MAP[meta_types[d]]:

            score_func = METRICS_DICT[m]
            # print(X_true[sids, d], X_pred[sids, d])
            s = score_func(X_true[sids, d], X_pred[sids, d],
                           X_min[d], X_max[d], X_mean[d], X_std[d])
            perf_dict[d][m] = s

    return perf_dict


def compute_score_W(W_true, W_pred, score=mean_squared_error):

    assert len(W_true) == len(W_pred)
    Ds = sorted(W_true.keys())
    preds = defaultdict(list)
    for d in Ds:
        wd_true = W_true[d]
        wd_pred = W_pred[d]
        # eassert len(wd_true) == len(wd_pred), '{} {}'.format(wd_true, wd_pred)

        for t in wd_pred.keys():
            #
            # if there is no corresponding type in the ground truth, then its weight is zero
            if t not in wd_true:
                w_true = 0.0
            else:
                w_true = wd_true[t]

            w_pred = wd_pred[t]

            s = score(np.array([w_true]),
                      np.array([w_pred]))
            preds[d].append(s)

        for t in wd_true.keys():
            #
            # if there is no corresponding type in the ground truth, then its weight is zero
            if t not in wd_pred:
                w_pred = 0.0
                w_true = wd_true[t]

                s = score(np.array([w_true]),
                          np.array([w_pred]))
                preds[d].append(s)

        preds[d] = np.array(preds[d]).mean()

    return np.array([preds[d] for d in sorted(preds.keys())])


def init_param_weights(type_param_map, init_weights="uniform", rand_gen=None, prior=OMEGA_UNINF_PRIOR):
    param_weights_map = {}
    for m_t, m_t_map in type_param_map.items():
        print('MT', m_t, m_t_map)
        n_param_forms = 0
        type_list_map = {}
        for type, type_map in m_t_map.items():
            n_param_forms += len(type_map)
            type_list_map.update(type_map)
        w = None
        if init_weights == 'uniform':
            w = np.ones(n_param_forms)
        elif init_weights == 'random':
            w = rand_gen.dirichlet(alpha=[prior for j in range(n_param_forms)])
        w = w / w.sum()
        #
        # order here does not really matter
        assert len(type_list_map) == len(w), type_list_map
        p_weights = {p: p_w for p, p_w in zip(type_list_map.keys(), w)}
        param_weights_map[m_t] = p_weights

    return param_weights_map


def abda_gibbs_inference(spn,
                         X_splits,
                         X_miss,
                         X_eval,
                         meta_types,
                         rand_gen,
                         n_iters=1000,
                         W_prior=100,
                         burn_in=4000,
                         save_ll_history=0,
                         save_samples=1,
                         plot_iter=0,
                         show_figs=True,
                         parametric_leaf_priors=None,
                         omega_prior='learnspn',
                         # leaf_omega_prior='uniform',
                         omega_unif_prior=OMEGA_UNINF_PRIOR,
                         leaf_omega_unif_prior=LEAF_OMEGA_UNINF_PRIOR,
                         scores=SCORE_MAP,
                         global_true_W=None,
                         save_all_params=False,
                         output_path=None):
    """

    ---
    Simple gibbs sampling scheme

    """

    #
    # selecting split to train and evaluating on
    X = None
    # X_eval = None

    X_train, X_valid, X_test = X_splits
    split_eval = None
    if X_miss is None:
        X = X_train
        logging.info('Training on original train')
        if X_valid is None:
            split_eval = 'valid'
            logging.info('Evaluating on original train')
        else:
            split_eval = 'train'
            logging.info('Evaluating on valid')
        save_model_params = True
    else:

        logging.info('Training on train split with MISSING VALUES')
        # X = np.copy(X_train)
        # # in_miss = np.isnan(X)
        # X[X_miss] = np.nan
        # X_eval = np.copy(X_train)
        # # miss_vals = np.isnan(X_miss)
        # X_eval[~X_miss] = np.nan
        # X_train = X
        save_model_params = False
        X = X_train
        X_eval_ext = extend_matrix_one_value_per_row(X_eval)

    N, D = X.shape
    logging.info('Processing data with {} shape'.format(X.shape))
    preprocess_positive_real_data(X, meta_types)

    # #
    # # Statistics
    # #
    # ll_history = []
    # ll_count_history = []
    # leaves_ll_history = defaultdict(list)
    # sum_samplehits_history = defaultdict(list)
    # leaf_samplehits_counter = defaultdict(Counter)
    # param_leaves_ll_history = defaultdict(list)
    # param_leaf_samplehits_counter = defaultdict(Counter)
    # perf_dict_history = []

    ############################################################################################
    #
    # INITING RVs
    #

    #
    # initing maps instances -> SPN leaves and edge_counts (Z_n)
    init_spn_sampling(spn)

    #
    # initing globall weights
    # NOTE: they can be empty if we have not shared and constrained leaves
    W, W_priors = init_weights(spn, meta_types, alpha_prior=W_prior, rand_gen=rand_gen)

    #
    # INIT S (just zeros)
    S = np.zeros((N, D), dtype=np.int32)

    #
    # collecting samples from the posterior
    samples = []

    #
    # collecting leaves and their associated parametric forms
    spn_type_leaves = get_nodes_by_type(spn, TypeLeaf)
    spn_parametric_leaves = get_nodes_by_type(spn, Parametric)

    part_path = None
    if output_path:
        part_path = os.path.join(output_path, 'partitions')
        os.makedirs(part_path, exist_ok=True)

    #
    # keeping track of best sample from the posterior, based on model likelihood
    best_stats = {}
    best_val_ll = -np.inf
    best_model = None
    best_iter = -1

    #
    # getting stats for relative errors
    X_min = np.nanmin(X, axis=0)
    X_max = np.nanmax(X, axis=0)
    X_mean = np.nanmean(X, axis=0)
    # X_std = np.sqrt(np.nansum(X - X_mean, axis=0))
    X_std = scipy.stats.sem(X, axis=0, nan_policy='omit')
    logging.info('X MIN {}'.format(X_min))
    logging.info('X MAX {}'.format(X_max))
    logging.info('X MEAN {}'.format(X_mean))
    logging.info('X STD {}'.format(X_std))

    ############################################################################################
    #
    # Gibbs sampler
    #
    for it in range(n_iters):

        #
        # A sample may contain:
        #    - 'id': the iteration id
        #    - 'partition': a np array of size NxD in which each entry is associated a leaf id
        #    - 'type-P': a np array of size NxD where each entry is the type/parametric form associated to it
        #    - 'Zs': a map sum-node id -> [children.row_ids]
        #    - 'Etas': a map parametric-leaf id -> parameters
        #    - 'Omegas: the map of the SPN upper structure weights {sum-id: weights}
        #    - '*ll*': containing keys for the lls {'train-lls', 'valid-lls', 'test-lls'}
        #    - 'Ws': the constrained type weights, if present
        #    - 'global-W' the dictionary containing the computed global w for parametric forms
        #    - 'global-type-W' the dictionary containing the computed global types
        #    - 'time': time taken for one iteration
        sample = {}
        sample['id'] = it

        iter_start_t = perf_counter()

        #
        # plotting
        #
        if burn_in and it >= burn_in and plot_iter and it % plot_iter == 0:

            fig_output_path = None
            if output_path:
                fig_output_path = os.path.join(output_path,
                                               'fit',
                                               'fit@iter-{}'.format(it + 1))
                os.makedirs(fig_output_path, exist_ok=True)
            #
            # plotting on a per-fat-leaf basis
            for t_leaf in spn_type_leaves:

                # print('LEN x', t_leaf.id, t_leaf.name, len(t_leaf.row_ids))
                # for l in t_leaf.children:
                #     print('\tLEN x', l.id, l.name, len(l.row_ids))

                d = t_leaf.scope[0]

                leaf_fig_output_path = None
                if output_path:
                    leaf_fig_output_path = os.path.join(fig_output_path, 'd-{}-leaf-{}'.format(d,
                                                                                               t_leaf.id))
                plot_distributions_fitting_data(X,
                                                # spn_parametric_leaves,
                                                dist_nodes=t_leaf.children,
                                                f_id=d,
                                                type_leaf_id=t_leaf.id,
                                                bins=100,
                                                weight_scaled=t_leaf.weights,
                                                show_fig=show_figs,
                                                save_fig=leaf_fig_output_path)

    ##########################################################################################
    #
    # Sample Zs (and Ss) in the SPN
    #
        ll_start_t = perf_counter()
        map_rows_cols_to_node_id, lls = sample_induced_trees(spn, X, rand_gen)
        type_P = get_type_partitioning_leaves(spn, map_rows_cols_to_node_id)

        #
        # visualizing partition?
        if burn_in and it >= burn_in and plot_iter and it % plot_iter == 0:
            reord_ids = reorder_data_partitions(map_rows_cols_to_node_id)
            title_str = "leaf partitioning at iter {}".format(it + 1)
            part_out_path = None
            if output_path:
                part_out_path = os.path.join(part_path, 'leaf.partitioning@it{}'.format(it + 1))
            #
            # inv leaf map, for printing partitionings
            inv_leaf_map = {l.id: spn_to_str_equation(l) + " id: " + str(l.id)  # l.__class__.__name__
                            for l in get_nodes_by_type(spn, Leaf) if len(l.row_ids) > 0}
            visualize_data_partition(map_rows_cols_to_node_id[reord_ids],
                                     color_map_ids=inv_leaf_map,
                                     title=title_str,
                                     output=part_out_path, show_fig=show_figs)

            title_str = "type partitioning at iter {}".format(it + 1)
            part_out_path = None
            if output_path:
                part_out_path = os.path.join(part_path, 'type.partitioning@it{}'.format(it + 1))

            visualize_data_partition(type_P[reord_ids],
                                     color_map_ids=INV_TYPE_PARAM_MAP,
                                     title=title_str,
                                     output=part_out_path, show_fig=show_figs)

        ll_end_t = perf_counter()
        avg_ll = lls[:, spn.id].mean()
        logging.info('\tSampled Zs and Ss in the spn in {} secs (avg LL: {})'.format(ll_end_t - ll_start_t,
                                                                                     avg_ll))
        #
        # storing leaf partition and Zs
        if burn_in and it >= burn_in and save_all_params:
            sample['partition'] = map_rows_cols_to_node_id
            sample['type-P'] = type_P
            sample['Zs'] = {s.id: [c.row_ids for c in s.children]
                            for s in get_nodes_by_type(spn, Sum)}

        #
        # checking partitioning is correct
        validate_row_partitioning(spn, np.arange(X.shape[0]))

        ##################################################################################################
        #
        # sample \etas from parametric leaves in the SPN
        #
        ll_start_t = perf_counter()
        for l in spn_parametric_leaves:
            update_parametric_parameters_posterior(l, X, rand_gen, parametric_leaf_priors[l])

        if burn_in and it >= burn_in and save_model_params:
            sample['Etas'] = {l.id: dict(l.params) for l in spn_parametric_leaves}

        ll_end_t = perf_counter()
        logging.info('\tSampled params from parametric leaves in {} secs'.format(
            ll_end_t - ll_start_t))
        #
        # printing leaf information
        # for t_leaf in spn_type_leaves:
        #     logging.debug('\n\t\tType Leaf ({}) d: {} id: {}'.format(t_leaf.__class__.__name__,
        #                                                              t_leaf.scope[0],
        #                                                              t_leaf.id))
        #     for l in t_leaf.children:
        #         logging.debug('\t\t\tparam leaf ({})\tid: {} params:\t{}'.format(l.__class__.__name__,
        #                                                                          l.id,
        #                                                                          l.params))

        #################################################################################################
        #
        # sample W (for GLOBAL and constrained leaves)
        #
        if W:
            S_counts = {}
            for mixture_leaf in spn_type_leaves:
                d = mixture_leaf.scope[0]
                if not d in S_counts:
                    # print(W_priors)
                    S_counts[d] = np.zeros_like(W_priors[d])

                S_counts[d] += mixture_leaf.edge_counts

            sample_Ws(W, meta_types, S_counts, W_priors, rand_gen)

            sample['Ws'] = W

            ll_end_t = perf_counter()

            logging.info('\n\tComputed sampling W in {} secs'.format(ll_end_t - ll_start_t))
        logging.info('\tCONSTRAINED type weights: {}'.format(W))

        ###############################################################################################
        #
        # sample Omegas
        ll_start_t = perf_counter()
        sample_spn_weights(spn, rand_gen, omega_prior, omega_unif_prior, leaf_omega_unif_prior)
        ll_end_t = perf_counter()
        logging.info('\tComputed sampling Omegas in {} secs'.format(ll_end_t - ll_start_t))

        if burn_in and it >= burn_in and save_model_params:
            sample['Omegas'] = {s.id: np.array(s.weights) for s in get_nodes_by_type(spn, Sum)}

        # #
        # # saving statistics
        # if save_ll_history and it % save_ll_history == 0:
        #     for s in get_nodes_by_type(spn, Sum):
        #         sum_samplehits_history[s.id].append(np.copy(s.edge_counts))

        ##############################################################################################
        #
        # evaluate performances
        #
        ##########
        #
        # compute global W
        #
        w_start_t = perf_counter()
        global_W = compute_global_type_weights(spn, aggr_type=False)
        global_type_W = compute_global_type_weights(spn, aggr_type=True)
        w_end_t = perf_counter()
        g_w_score = None
        if global_true_W:
            g_w_score = compute_score_W(global_true_W, global_W)
            logging.info('\tComputed global W (score: {}) in {} secs'.format(g_w_score.mean(),
                                                                             w_end_t - w_start_t))
            logging.info('\tComputed global W by eval:\n\t\t{} (type: {}) in {} secs'.format(global_W,
                                                                                             global_type_W,
                                                                                             w_end_t - w_start_t))
        sample['global-W'] = global_W
        sample['global-type-W'] = global_type_W
        sample['global-W-score'] = g_w_score

        ##########
        #
        # Computing sample likelihood
        if save_ll_history and it % save_ll_history == 0:
            logging.info('\nComputing model sample likelihood:')
            for x_split, split_name in zip([X_train, X_valid, X_test],
                                           ['train', 'valid', 'test']):
                if x_split is not None:
                    split_data_lls = log_likelihood(spn, x_split)

                    # m_id = max_node_id(spn)
                    # lls = np.zeros((x_split.shape[0], m_id + 1))
                    # split_data_lls_m = log_likelihood(spn, x_split, llls_matrix=lls)
                    # assert_array_almost_equal(split_data_lls, split_data_lls_m)

                    assert split_data_lls.shape[0] == x_split.shape[0]
                    logging.info('\t{} data LL mean:{} min:{} max:{}'.format(split_name,
                                                                             split_data_lls.mean(),
                                                                             split_data_lls.min(),
                                                                             split_data_lls.max()))

                    # max_id = np.argmax(split_data_lls)
                    # print('MAX ID', max_id, split_data_lls[max_id])
                    # for l in spn_parametric_leaves:
                    #     if max_id in l.row_ids:
                    #         print('\tinside leaf', l.id, l.__class__.__name__, l.params)
                    #         print(lls[max_id, l.id], np.argmax(lls[max_id]), np.max(lls[max_id]))

                    #
                    # just storing all lls for all splits
                    # TODO: check if this is too memory intensive
                    sample['{}-lls'.format(split_name)] = split_data_lls

                    #
                    # update best model?
                    if it >= burn_in and split_name == split_eval:
                        avg_val_ll = split_data_lls.mean()
                        print('EVAL', avg_val_ll)
                        if avg_val_ll > best_val_ll:
                            best_val_ll = avg_val_ll
                            best_iter = it
                            #
                            # overwriting the spn
                            best_spn_output_path = os.path.join(out_path, 'best.spn.model.pkl')
                            store_start_t = perf_counter()
                            with open(best_spn_output_path, 'wb') as f:
                                pickle.dump(spn, f)
                            store_end_t = perf_counter()
                            logging.info('Stored best spn to {} (in {} secs)'.format(best_spn_output_path,
                                                                                     store_end_t - store_start_t))

            # for t in spn_type_leaves:
            #     leaves_ll_history[t.id].append(lls[:, t.id].mean())
            #     for ins in t.row_ids:
            #         leaf_samplehits_counter[t.id][ins] += 1

            # for l in spn_parametric_leaves:
            #     param_leaves_ll_history[l.id].append(lls[:, l.id].mean())
            #     for ins in l.row_ids:
            #         param_leaf_samplehits_counter[l.id][ins] += 1

        ############
        #
        # Dealing with missing values
        # --ll
        if burn_in and it >= burn_in and X_miss is not None:
            # print(X_eval_ext.shape)
            ll_start_t = perf_counter()
            mv_lls = log_likelihood(spn, X_eval_ext)
            # assert mv_lls.shape[0] == X_miss.sum()
            ll_end_t = perf_counter()
            assert mv_lls.shape[0] == (~np.isnan(X_eval)).sum()
            logging.info('\t computed missing value LL mean: {} min: {} max:{} (in {} secs)'.format(mv_lls.mean(),
                                                                                                    mv_lls.min(),
                                                                                                    mv_lls.max(),
                                                                                                    ll_end_t - ll_start_t))
            sample['mv-lls'] = mv_lls

            #
            # predictions
            ll_start_t = perf_counter()
            mv_mpe_ass, mv_mpe_lls = mpe(spn, X)
            X_eval_mask = np.isnan(X_eval)
            # forcing masking for those datasets with inherent missing values
            mv_mpe_ass[X_eval_mask] = np.nan
            assert_array_equal(np.isnan(X_eval), np.isnan(mv_mpe_ass))
            mv_preds = compute_predictions_dict(X_eval, mv_mpe_ass, meta_types,
                                                X_min, X_max, X_mean, X_std,
                                                # scores
                                                )
            ll_end_t = perf_counter()
            logging.info('\t computed missing value PREDS in {} secs'.format(ll_end_t - ll_start_t))
            print_perf_dict(mv_preds)

            ll_start_t = perf_counter()
            X_mep_eval_ext = extend_matrix_one_value_per_row(mv_mpe_ass)

            mv_mpe_lls = log_likelihood(spn, X_mep_eval_ext)
            ll_end_t = perf_counter()
            # assert mv_mpe_lls.shape[0] == (~np.isnan(X_eval)).sum()
            # assert mv_lls.shape[0] == mv_mpe_lls.shape[0]
            logging.info('\t computed mv imputation LL mean: {} min: {} max:{} (in {} secs)'.format(mv_mpe_lls.mean(),
                                                                                                    mv_mpe_lls.min(),
                                                                                                    mv_mpe_lls.max(),
                                                                                                    ll_end_t - ll_start_t))

            sample['mv-preds'] = mv_mpe_ass
            sample['mv-preds-lls'] = mv_mpe_lls
            sample['mv-preds-scores'] = mv_preds

        # #
        # # FIXME: this has to be updated for the new process
        # # saving predictions on features
        # if save_perf_history and it % save_perf_history == 0:
        #     perf_start_t = perf_counter()

        #     perf_dict = eval_predict_data(X, C.astype(np.uint8), R.astype(np.int64), S,
        #                                   (Yreal, Yint, Ypos, Ycat, Yord, Ycount),
        #                                   Wint,
        #                                   theta, theta_L, theta_H,
        #                                   maxX, minX, meanX,
        #                                   # pos_continuous_metrics=pos_cont_perf_metrics,
        #                                   discrete_metrics=disc_perf_metrics,
        #                                   continuous_metrics=cont_perf_metrics)
        #     perf_end_t = perf_counter()
        #     print('\tComputed prediction performances in {} secs'.format(perf_end_t - perf_start_t))
        #     print_perf_dict(perf_dict)
        #     perf_dict_history.append(perf_dict)

        iter_end_t = perf_counter()
        print('Done iteration {}/{} in {}\n\n\n------------------'.format(it + 1, n_iters,
                                                                          iter_end_t - iter_start_t),
              # end="\r"
              )
        sample['time'] = iter_end_t - iter_start_t

        if save_samples and it % save_samples == 0:
            samples.append(sample)

    best_stats['id'] = best_iter
    best_stats['lls'] = best_val_ll

    return samples, best_stats


def get_parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str,
                        help='Path to dataset folder')
    parser.add_argument('--miss', type=str, nargs='?',
                        default=None,
                        help='Path to missing data mask')
    parser.add_argument('-o', '--output', type=str,
                        default='./exp/',
                        help='Output path to exp result')
    #
    # LearnMSPN parameters
    # parser.add_argument('-r', '--row-split', type=str, nargs='?',
    #                     default='rdc-kmeans',
    #                     help='Cluster method to apply on rows')
    # parser.add_argument('-c', '--col-split', type=str, nargs='?',
    #                     default='rdc',
    #                     help='(In)dependency test to apply to columns')
    # parser.add_argument('--row-split-args', type=str, nargs='?',
    #                     help='Additional row split method parameters in the form of a list' +
    #                          ' "[name1=val1,..,namek=valk]"')
    # parser.add_argument('--col-split-args', type=str, nargs='?',
    #                     help='Additional col split method parameters in the form of a list' +
    #                          ' "[name1=val1,..,namek=valk]"')
    parser.add_argument('--col-split-threshold', type=float,
                        default=0.8,
                        help='Threshold for column split test')
    parser.add_argument('--seed', type=int, nargs='?',
                        default=1337,
                        help='Seed for the random generator')
    parser.add_argument('-m', '--min-inst-slice', type=int, nargs='?',
                        default=50,
                        help='Min number of instances in a slice to split by cols')
    parser.add_argument('--leaf-type', type=str, nargs='?',
                        default='pm',
                        help='Leaf type (tm|pm|tm-pm)')
    parser.add_argument('--type-param-map', type=str,
                        default='spicky-prior-1',
                        help='Which default param map to instantiate leaves')
    parser.add_argument('--param-init', type=str,
                        default='default',
                        help='How to initialize parameters of parametric forms (prior|mle|default)')
    parser.add_argument('--param-weight-init', type=str,
                        default='uniform',
                        help='How to initialize weights of mixture of parametric forms/sypes (uniform|random)')
    # parser.add_argument('--leaf-weights', type=str, nargs='?',
    #                     default='uniform',
    #                     help='')
    # parser.add_argument('-a', '--alpha', type=float, nargs='?',
    #                     default=1.0,
    #                     help='Smoothing factor for leaf probability estimation')
    # parser.add_argument('--product-first', action='store_true',
    #                     help='Whether to split first on the columns')
    parser.add_argument('--save-model', action='store_true',
                        help='Whether to store the model file as a pickle file')
    parser.add_argument('--gzip', action='store_true',
                        help='Whether to compress the model pickle file')
    #
    # Gibbs sampler's parameters
    #
    parser.add_argument('-i', '--n-iters', type=int,
                        default=100,
                        help='Number of iterations of Gibbs sampler')
    parser.add_argument('-b', '--burn-in', type=int,
                        default=4000,
                        help='Number of iterations to discard samples')
    parser.add_argument('--ll-history', type=int,
                        default=1,
                        help='Whether to save the history of average model LL after a number of iterations')
    parser.add_argument('--plot-iter', type=int,
                        default=10,
                        help='Whether to plot iterations')
    parser.add_argument('--save-samples', type=int,
                        default=1,
                        help='Whether to save collected samples')
    parser.add_argument('--perf-history', type=int,
                        default=1,
                        help='Whether to save the history of model prections')
    parser.add_argument('--fig-size', type=int, nargs='+',
                        default=(10, 7),
                        help='A tuple for the explanation fig size ')
    # parser.add_argument('--show-explain', action='store_true',
    #                     help='Whether to show by screen the explanaitions')
    parser.add_argument('--show-plots', action='store_true',
                        help='Whether to show by screen the plots')
    parser.add_argument('--ravg-buffer', type=int, nargs='+',
                        default=[100],
                        help='Running average buffer size for computing lls')
    # parser.add_argument('--no-sample-u-omega', type=int,
    #                     default=-1,
    #                     help='Number of iterates before which to disable sampling Us and Omegas for the SPN')
    parser.add_argument('--omega-prior', type=str,
                        default='uniform',
                        help='Which prior to use for Dirichlets for Omegas (None|uniform|learnspn)')
    # parser.add_argument('--leaf-omega-prior', type=str,
    #                     default='uniform',
    #                     help='Which prior to use for Dirichlets for Omegas (None|uniform|learnspn)')
    parser.add_argument('--omega-unif-prior', type=float,
                        default=OMEGA_UNINF_PRIOR,
                        help='Default uniform prior (pseudo counts) for Omega Dirichlets (default 10)')
    parser.add_argument('--leaf-omega-unif-prior', type=float,
                        default=LEAF_OMEGA_UNINF_PRIOR,
                        help='Default uniform prior (pseudo counts) for leaf Omega Dirichlets (default 0.1)')
    parser.add_argument('--cat-unif-prior', type=float,
                        default=CAT_UNIF_PRIOR,
                        help='Default uniform prior (pseudo counts) for Categorical Dirichlets (default 1)')
    parser.add_argument('--w-unif-prior', type=float,
                        default=W_UNINF_PRIOR,
                        help='Default uniform prior (pseudo counts) for W Dirichlets (default 100)')
    parser.add_argument('--exp-id', type=str,
                        default=None,
                        help='Dataset output suffix')
    parser.add_argument('--dummy-id', type=str,
                        default=None,
                        help='Not used parameter')
    parser.add_argument('-v', '--verbose', type=int, nargs='?',
                        default=1,
                        help='Verbosity level')

    return parser


if __name__ == '__main__':

    #########################################
    # creating the opt parser

    #
    # parsing the args
    args = get_parse_args().parse_args()

    #
    # setting verbosity level
    if args.verbose == 1:
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    elif args.verbose == 2:
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    #
    # an additional seed fix, is this unnecessary?
    rand_gen = np.random.RandomState(args.seed)

    # row_split_args = None
    # if args.row_split_args is not None:
    #     row_key_value_pairs = args.row_split_args.translate(
    #         {ord('['): '', ord(']'): ''}).split(',')
    #     row_split_args = {key.strip(): value.strip() for key, value in
    #                       [pair.strip().split('=')
    #                        for pair in row_key_value_pairs]}
    # else:
    #     row_split_args = {}
    # logging.info('Row split method parameters:  {}'.format(row_split_args))

    # col_split_args = None
    # if args.col_split_args is not None:
    #     col_key_value_pairs = args.col_split_args.translate(
    #         {ord('['): '', ord(']'): ''}).split(',')
    #     col_split_args = {key.strip(): value.strip() for key, value in
    #                       [pair.strip().split('=')
    #                        for pair in col_key_value_pairs]}
    # else:
    #     col_split_args = {}
    # logging.info('Col split method parameters:  {}'.format(col_split_args))

    logging.info("Starting with arguments:\n%s", args)

    #
    # creating output dirs if they do not exist
    dataset_name = os.path.basename(args.dataset)
    date_string = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    if args.exp_id:
        out_path = os.path.join(args.output, args.exp_id)
    else:
        out_path = os.path.join(args.output,  dataset_name, date_string)
    os.makedirs(out_path, exist_ok=True)

    out_log_path = os.path.join(out_path, 'exp.log')
    logging.info('Opening logging file in {}'.format(out_log_path))

    # #
    # # loading isabel data
    # data_dict, feature_names, feature_types, domains = load_mat_dict(dataset_name,
    #                                                                  data_dir=args.data_dir)

    #
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
        # miss_data_path = os.path.join(args.dataset, 'miss.data')

        print('Looking for pickle at dir... {}'.format(args.miss))
        miss_data_path = args.miss

        try:
            load_start_t = perf_counter()
            with open(miss_data_path, 'rb') as f:
                X_miss = pickle.load(f)
            load_end_t = perf_counter()
        except:
            print('FAILED to load pickle, trying matlab .mat file')
            load_start_t = perf_counter()
            X_miss = load_islv_miss(miss_data_path, shape=X_train.shape)
            load_end_t = perf_counter()

        logging.info('Loaded missing data mask {} from {} (in {} secs)'.format(X_miss.shape,
                                                                               miss_data_path,
                                                                               load_end_t - load_start_t))
        assert X_miss.shape == X_train.shape

        X_eval = np.copy(X_train)
        # miss_vals = np.isnan(X_miss)
        X_eval[~X_miss] = np.nan

        # masking the train
        X_train_orig = np.copy(X_train)
        X_train[X_miss] = np.nan

        assert_array_equal(np.isnan(X_train_orig) | X_miss, np.isnan(X_train))

        logging.info('\n\nDEALING with {} missing values '.format(np.isnan(X_train).sum()))

    #
    # sturcture learning hyperparameters
    #
    # DEFAULT_ROW_SPLIT_ARGS['seed'] = {'default': args.seed, 'type': int}

    #
    # getting the splitting functions
    # row_split_func = ROW_SPLIT_METHODS[args.row_split]
    # row_split_arg_names = inspect.getargspec(row_split_func)
    # print('row args', row_split_arg_names[0])

    # r_args = {}
    # for r_arg in row_split_arg_names[0]:
    #     r_val = None
    #     if r_arg not in row_split_args:
    #         r_val = DEFAULT_ROW_SPLIT_ARGS[r_arg]['default']
    #     else:
    #         r_val = row_split_args[r_arg]
    #     r_args[r_arg] = DEFAULT_ROW_SPLIT_ARGS[r_arg]['type'](r_val)

    # print('\n\n\nR ARGS', r_args)
    # row_split_method = row_split_func(**r_args)

    # col_split_func = COL_SPLIT_METHODS[args.col_split]
    # col_split_arg_names = inspect.getargspec(col_split_func)
    # print('col args', col_split_arg_names[0])

    # c_args = {}
    # for c_arg in col_split_arg_names[0]:
    #     c_val = None
    #     if c_arg not in col_split_args:
    #         c_val = DEFAULT_COL_SPLIT_ARGS[c_arg]['default']
    #     else:
    #         c_val = col_split_args[c_arg]
    #     c_args[c_arg] = DEFAULT_COL_SPLIT_ARGS[c_arg]['type'](c_val)

    # print('\n\n\nC ARGS', c_args)
    # col_split_method = col_split_func(**c_args)

    table_header = '\t'.join(['dataset',
                              'fold',
                              # 'row-split',
                              # 'row-split-args',
                              # 'col-split',
                              # 'col-split-args',
                              'min-inst-split',
                              'alpha',
                              # 'leaf',
                              # 'prior_weight',
                              # 'bootstraps',
                              # 'avg-bootstraps',
                              # 'prod-first',
                              'train-avg-ll',
                              # 'valid-avg-ll',
                              # 'test-avg-ll',
                              'learning-time',
                              'eval-time',
                              'nodes',
                              'edges',
                              'layers',
                              'prod-nodes',
                              'sum-nodes',
                              'leaves',
                              'spn-json-path',
                              'lls-files'
                              ])
    table_header += '\n'

    with open(out_log_path, 'w') as out_file:
        out_file.write(table_header)

        # for f, train in enumerate(fold_splits):
        # logging.info('\n\n######## FOLD {}/{} ###########\n'.format(f + 1, len(fold_splits)))
        # valid_str = None if valid is None else valid.shape
        logging.info('train:{}'.format(X_train.shape))

        # train = whole_data[train_index]
        # test = whole_data[test_index]

        #
        # retrieving stats and meta-types back, if needed
        if meta_types is None:
            stats_map = {}
            data_stats_path = os.path.join(args.dataset, 'data.stats')
            with open(data_stats_path, 'rb') as f:
                stats_map = pickle.load(f)
            meta_types = stats_map['meta-types']
            domains = stats_map['domains']
            g_true_W = stats_map.get('full-type-W')

        #
        # call the new learnSPN routine here
        #

        #
        # creating a context, then learning
        #
        ds_context = Context(meta_types=meta_types)
        ds_context.domains = domains
        type_param_map = DEFAULT_TYPE_PARAM_MAP[args.type_param_map]

        #
        # set dirichlet prior update as something a proxy
        type_param_map[MetaType.DISCRETE][Type.CATEGORICAL][Categorical]['prior'].alphas_0 = args.cat_unif_prior

        ds_context.param_form_map = type_param_map
        # prior_map = DEFAULT_PRIOR_MAP[args.prior_map]
        # ds_context.prior_map = prior_map
        init_weights_map = init_param_weights(type_param_map, args.param_weight_init, rand_gen)
        logging.info('Starting with inited param weights: {}'.format(init_weights_map))

        ds_context.init_weights_map = init_weights_map
        ds_context.leaf_type = args.leaf_type
        ds_context.param_init = args.param_init
        ds_context.priors = {}

        learn_start_t = perf_counter()
        spn = learn_hspn(X_train,
                         ds_context,
                         min_instances_slice=args.min_inst_slice,
                         threshold=args.col_split_threshold,
                         linear=False,
                         # ohe=True,
                         memory=None,
                         rand_gen=rand_gen)

        rebuild_scopes_bottom_up(spn)
        assign_ids(spn)
        learn_end_t = perf_counter()

        stats = get_structure_stats_dict(spn)
        learning_time = learn_end_t - learn_start_t
        logging.info('\n\nLearned spn in {} secs\n\t with stats:\n\t{}'.format(learning_time,
                                                                               stats))

        add_typed_leaves_text_support()
        add_parametric_inference_support()
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
        # FIXME: are priors returned by structure learning?
        #
        parametric_leaf_priors = ds_context.priors
        logging.info('\nComputed priors:\n\t{}'.format(parametric_leaf_priors))

        # spn = SPN.LearnStructure(train,
        #                          featureNames=feature_names,
        #                          # families=families,
        #                          domains=domains,
        #                          featureTypes=feature_types,
        #                          min_instances_slice=args.min_inst_slice,
        #                          # bin_width=args.tail_width,
        #                          alpha=args.alpha,
        #                          kest=args.k,
        #                          C=C,
        #                          R=R,
        #                          s2B=args.s2b, s2Z=args.s2z,
        #                          # isotonic=args.isotonic,
        #                          # pw_bootstrap=args.bootstraps,
        #                          # avg_pw_boostrap=args.average_bootstraps,
        #                          row_split_method=row_split_method,
        #                          col_split_method=col_split_method,
        #                          cluster_first=cluster_first,
        #                          # kernel_family=args.kernel_family,
        #                          # kernel_bandwidth=args.kernel_bandwidth,
        #                          # kernel_metric=args.kernel_metric,
        #                          # prior_weight=args.prior_weight,
        #                          rand_seed=args.seed
        #                          )

        exp_str = dataset_name
        exp_str += '\t{}'.format(f)
        # exp_str += '\t{}'.format(args.row_split)
        # exp_str += '\t{}'.format(args.row_split_args)
        # exp_str += '\t{}'.format(args.col_split)
        # exp_str += '\t{}'.format(args.col_split_args)
        exp_str += '\t{}'.format(args.min_inst_slice)
        # exp_str += '\t{}'.format(args.alpha)
        # exp_str += '\t{}'.format(args.leaf)
        # exp_str += '\t{}'.format(args.prior_weight)
        # exp_str += '\t{}'.format(args.bootstraps)
        # exp_str += '\t{}'.format(args.average_bootstraps)
        # exp_str += '\t{}'.format(args.product_first)
        # exp_str += '\t{}'.format(train_avg_ll)
        exp_str += '\t{}'.format(-np.inf)
        # exp_str += '\t{}'.format(valid_avg_ll)
        # exp_str += '\t{}'.format(test_avg_ll)
        exp_str += '\t{}'.format(learning_time)
        # exp_str += '\t{}'.format(eval_time)
        exp_str += '\t{}'.format(-np.inf)
        exp_str += '\t{}'.format(stats['nodes'])
        exp_str += '\t{}'.format(stats['edges'])
        exp_str += '\t{}'.format(stats['layers'])
        exp_str += '\t{}'.format(stats['prod'])
        exp_str += '\t{}'.format(stats['sum'])
        exp_str += '\t{}'.format(stats['leaf'])
        # exp_str += '\t{}'.format(spn_json_path)
        # lls_file_path = os.path.join(output_path, 'lls-{}'.format(f))
        # exp_str += '\t{}'.format(lls_file_path)
        exp_str += '\n'
        out_file.write(exp_str)

        # saving lls to numpy files
        # np.save('{}.train'.format(lls_file_path), train_lls)

    #
    #

    os.makedirs(os.path.join(out_path, 'fit'), exist_ok=True)

    infer_start_t = perf_counter()
    samples, best_stats = abda_gibbs_inference(spn,
                                               X_splits=(X_train, X_valid, X_test),
                                               X_miss=X_miss,
                                               X_eval=X_eval,
                                               meta_types=meta_types,
                                               rand_gen=rand_gen,
                                               n_iters=args.n_iters,
                                               burn_in=args.burn_in,
                                               W_prior=args.w_unif_prior,
                                               parametric_leaf_priors=parametric_leaf_priors,
                                               save_samples=args.save_samples,
                                               save_ll_history=args.ll_history,
                                               omega_prior=args.omega_prior,
                                               # leaf_omega_prior=args.leaf_omega_prior,
                                               plot_iter=args.plot_iter,
                                               show_figs=args.show_plots,
                                               output_path=out_path,
                                               omega_unif_prior=args.omega_unif_prior,
                                               leaf_omega_unif_prior=args.leaf_omega_unif_prior,
                                               scores=SCORE_MAP,
                                               global_true_W=g_true_W)
    infer_end_t = perf_counter()
    print('Done in {}'.format(infer_end_t - infer_start_t))

    best_iter = best_stats['id']

    #
    # dumping results individually
    if X_miss is not None:
        #
        # transductive setting
        dump_samples_to_pickle(samples, out_path, out_file_name='lls.pklz',
                               key='lls', count_key='id')
        dump_samples_to_pickle(samples, out_path, out_file_name='mv-lls.pklz',
                               key='mv-lls', count_key='id')
        dump_samples_to_pickle(samples, out_path, out_file_name='mv-preds.pklz',
                               key='mv-preds', count_key='id')
        dump_samples_to_pickle(samples, out_path, out_file_name='mv-preds-lls.pklz',
                               key='mv-preds-lls', count_key='id')
        dump_samples_to_pickle(samples, out_path, out_file_name='mv-preds-scores.pklz',
                               key='mv-preds-scores', count_key='id')
        dump_samples_to_pickle(samples, out_path, out_file_name='etas.pklz',
                               key='Etas', count_key='id')
        dump_samples_to_pickle(samples, out_path, out_file_name='omegas.pklz',
                               key='Omegas', count_key='id')
    else:
        pass
        # #
        # # inductive setting
        # assert best_iter != -1
        # print('Found best iter', best_iter)
        # dump_samples_to_pickle(samples, out_path, out_file_name='train-lls.pklz',
        #                        key='train-lls', count_key='id')
        # dump_samples_to_pickle(samples, out_path, out_file_name='valid-lls.pklz',
        #                        key='valid-lls', count_key='id')
        # dump_samples_to_pickle(samples, out_path, out_file_name='test-lls.pklz',
        #                        key='test-lls', count_key='id')

        # dump_best_sample_to_pickle(samples, out_path, out_file_name='best-train-lls.pklz',
        #                            key='train-lls', best_id=best_iter, count_key='id')
        # dump_best_sample_to_pickle(samples, out_path, out_file_name='best-valid-lls.pklz',
        #                            key='valid-lls', best_id=best_iter, count_key='id')
        # dump_best_sample_to_pickle(samples, out_path, out_file_name='best-test-lls.pklz',
        #                            key='test-lls', best_id=best_iter, count_key='id')

    dump_samples_to_pickle(samples, out_path, out_file_name='global-W.pklz',
                           key='global-W', count_key='id')
    dump_samples_to_pickle(samples, out_path, out_file_name='global-type-W.pklz',
                           key='global-type-W', count_key='id')

    # #
    # # dropping everything to a pickle
    # result_pickle_path = os.path.join(out_path, 'result-dump.pklz')
    # res = {'sl-time': learning_time,
    #        'samples': samples}
    # with gzip.open(result_pickle_path, 'wb') as f:
    #     pickle.dump(res, f)

    time_pickle_path = os.path.join(out_path, 'time.pklz')
    res = {'sl-time': learning_time, }
    with gzip.open(time_pickle_path, 'wb') as f:
        pickle.dump(res, f)

    #
    # storing the spn after training on file
    spn_output_path = os.path.join(out_path, 'spn.model@{}iters.pkl'.format(args.n_iters))
    store_start_t = perf_counter()
    with open(spn_output_path, 'wb') as f:
        pickle.dump(spn, f)
    store_end_t = perf_counter()
    logging.info('Stored spn to {} (in {} secs)'.format(spn_output_path,
                                                        store_end_t - store_start_t))

    #
    # print fitting
    fit_mix_path = os.path.join(out_path, 'vis-fit@iter{}'.format(args.n_iters))
    os.makedirs(fit_mix_path, exist_ok=True)
    plot_mixture_components_fitting_data(spn,
                                         X_train,
                                         bins=100,
                                         show_fig=args.show_plots,
                                         save_fig=fit_mix_path,
                                         cmap=None)
    plot_mixtures_fitting_multilevel(spn,
                                     X_train,
                                     meta_types,
                                     bins=100,
                                     show_fig=args.show_plots,
                                     save_fig=fit_mix_path,
                                     cmap=None)

    #
    # plot perf (ll, imputations, \dots...)
    ll_counts = np.array([s['id'] for s in samples if 'train-lls' in s])
    train_ll_hist, valid_ll_hist, test_ll_hist = None, None, None

    train_ll_hist = np.array([s['train-lls'][:, 0] for s in samples if 'train-lls' in s])
    print('train shape', train_ll_hist.shape)
    valid_ll_hist = np.array([s['valid-lls'][:, 0] for s in samples if 'valid-lls' in s])
    test_ll_hist = np.array([s['test-lls'][:, 0] for s in samples if 'test-lls' in s])

    mv_counts = np.array([s['id'] for s in samples if 'mv-lls' in s])
    mv_train_ll_hist = np.array([s['mv-lls'][:, 0] for s in samples if 'mv-lls' in s])

    train_avg_ll_hist = train_ll_hist.mean(axis=1) if len(train_ll_hist) > 0 else None
    valid_avg_ll_hist = valid_ll_hist.mean(axis=1) if len(valid_ll_hist) > 0 else None
    test_avg_ll_hist = test_ll_hist.mean(axis=1) if len(test_ll_hist) > 0 else None
    mv_train_avg_ll_hist = mv_train_ll_hist.mean(axis=1) if len(mv_train_ll_hist) > 0 else None

    pp_counts = np.array([s['id'] for s in samples if 'mv-preds-scores' in s])
    perf_history = [s['mv-preds-scores'] for s in samples if 'mv-preds-scores' in s]
    #
    # plotting likelihoods
    if args.ll_history:

        import matplotlib
        if not args.show_plots:
            matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages

        # ll_hist = np.array(ll_hist).T
        # hll_hist = np.array(hll_hist).T
        # print('LL hist shape', ll_hist.shape)

        # avg_ll = ll_hist.mean(axis=0)
        # avg_hll = hll_hist.mean(axis=0)

        for avg_ll_hist, ll_hist, split_name in zip([train_avg_ll_hist,
                                                     valid_avg_ll_hist, test_avg_ll_hist],
                                                    [train_ll_hist,
                                                     valid_ll_hist,
                                                     test_ll_hist],
                                                    ['train', 'valid', 'test']):
            if avg_ll_hist is not None and len(avg_ll_hist) > 0:

                fig, ax = plt.subplots(figsize=args.fig_size)
                ax.plot(ll_counts, avg_ll_hist, label='sample avg LL')
                # ax.plot(ll_counts, avg_ll, label='sample avg LL')
                # ax.plot(ll_counts, avg_hll, label='sample avg HLL')
                for b in args.ravg_buffer:
                    ravg_ll_hist = running_avg_numba(ll_hist.T, b)
                    # ravg_hll_hist = running_avg_numba(hll_hist, b)
                    ravg_ll = ravg_ll_hist.mean(axis=0)
                    print(ravg_ll.shape, ravg_ll_hist.shape)
                    # ravg_hll = ravg_hll_hist.mean(axis=0)
                    ax.plot(ll_counts, ravg_ll, label='last {} samples avg LL'.format(b))
                    # ax.plot(ll_counts, ravg_hll, label='last {} samples avg HLL'.format(b))
                ax.legend()

                ll_hist_output = os.path.join(out_path, '{}-ll-history.pdf'.format(split_name))
                pp = PdfPages(ll_hist_output)
                pp.savefig(fig)
                pp.close()
                print('Saved LL history to {}'.format(ll_hist_output))

                if args.show_plots:
                    plt.show()

        if mv_train_avg_ll_hist is not None and len(mv_train_avg_ll_hist) > 0:

            fig, ax = plt.subplots(figsize=args.fig_size)
            ax.plot(mv_counts, mv_train_avg_ll_hist, label='mv avg LL')
            # ax.plot(ll_counts, avg_ll, label='sample avg LL')
            # ax.plot(ll_counts, avg_hll, label='sample avg HLL')
            for b in args.ravg_buffer:
                ravg_ll_hist = running_avg_numba(mv_train_ll_hist.T, b)
                # ravg_hll_hist = running_avg_numba(hll_hist, b)
                ravg_ll = ravg_ll_hist.mean(axis=0)
                print(ravg_ll.shape, ravg_ll_hist.shape)
                # ravg_hll = ravg_hll_hist.mean(axis=0)
                ax.plot(mv_counts, ravg_ll, label='last {} samples avg LL'.format(b))
                # ax.plot(ll_counts, ravg_hll, label='last {} samples avg HLL'.format(b))
            ax.legend()

            mv_ll_hist_output = os.path.join(out_path, 'miss-ll-history.pdf')
            pp = PdfPages(mv_ll_hist_output)
            pp.savefig(fig)
            pp.close()
            print('Saved MV LL history to {}'.format(mv_ll_hist_output))

            if args.show_plots:
                plt.show()

    # if args.ll_history:

    #     import matplotlib

    #     if not args.show_plots:
    #         matplotlib.use('Agg')
    #     import matplotlib.pyplot as plt
    #     from matplotlib.backends.backend_pdf import PdfPages

    #     fig, ax = plt.subplots(figsize=args.fig_size)
    #     for split in ['train', 'valid', 'test']
    #     ax.plot(ll_counts, ll_hist, label='avg LL', alpha=0.6)
    #     # ax.plot(ll_counts, pll_hist, label='avg PLL')
    #     for l_id, leaf_ll_hist in leaves_ll_hist.items():
    #         ax.plot(ll_counts, leaf_ll_hist, label='avg leaf {} LL'.format(l_id), alpha=0.6)
    #     # ax.plot(hll_hist, label='avg HLL')
    #     ax.legend()

    #     ll_hist_output = os.path.join(output_path, 'll-history.pdf')
    #     pp = PdfPages(ll_hist_output)
    #     pp.savefig(fig)
    #     pp.close()
    #     print('Saved LL history to {}'.format(ll_hist_output))

    #     if args.show_plots:
    #         plt.show()

    #     # fig, ax = plt.subplots(figsize=args.fig_size)
    #     # ax.plot(ll_counts, pll_hist, label='avg PLL', alpha=0.6)
    #     # for l_id, leaf_pll_hist in leaf_plls_hist.items():
    #     #     ax.plot(ll_counts, leaf_pll_hist, label='avg leaf {} PLL'.format(l_id), alpha=0.6)
    #     # # ax.plot(hll_hist, label='avg HLL')
    #     # ax.legend()

    #     # pll_hist_output = os.path.join(output_path, 'pll-history.pdf')
    #     # pp = PdfPages(pll_hist_output)
    #     # pp.savefig(fig)
    #     # pp.close()
    #     # print('Saved PLL history to {}'.format(pll_hist_output))

    #     # if args.show_plots:
    #     #     plt.show()

    #     fig, ax = plt.subplots(figsize=args.fig_size)
    #     ax.set_yscale("log", nonposy='clip')
    #     for s_id, samplehits_hist in sum_samplehits_hist.items():
    #         sh_matrix = np.array(samplehits_hist)
    #         print('SH matrix', sh_matrix)
    #         for h in range(sh_matrix.shape[1]):
    #             ax.plot(ll_counts, sh_matrix[:, h],
    #                     label='#hits U_{} = {}'.format(s_id, h), alpha=0.6)
    #     # ax.plot(hll_hist, label='avg HLL')
    #     ax.legend()

    #     sh_hist_output = os.path.join(output_path, 'samplehits-history.pdf')
    #     pp = PdfPages(sh_hist_output)
    #     pp.savefig(fig)
    #     pp.close()
    #     print('Saved sample hits history to {}'.format(sh_hist_output))

    #     if args.show_plots:
    #         plt.show()

    #     for l_id, counter in leaf_samplehits_count.items():
    #         fig, ax = plt.subplots(figsize=args.fig_size)
    #         labels, values = zip(*counter.items())
    #         print('LABELs', labels, 'VALS', values)
    #         sorted_ids = np.argsort(labels)
    #         values = np.array(values)[sorted_ids]
    #         labels = np.array(labels)[sorted_ids]
    #         ax.scatter(labels, values,  # normed=True,
    #                    label='#hits leaf_{}'.format(l_id))

    #         ax.legend()

    #         count_hist_output = os.path.join(output_path,
    #                                          'samplehits-count-leaf-{}.pdf'.format(l_id))
    #         pp = PdfPages(count_hist_output)
    #         pp.savefig(fig)
    #         pp.close()
    #         print('Saved sample counter for leaf {}history to {}'.format(l_id, count_hist_output))

    #         if args.show_plots:
    #             plt.show()

    # #
    # # plotting predictions
    if args.perf_history and len(perf_history) > 0:

        perf_path = os.path.join(out_path, 'perf@{}iter'.format(args.n_iters))
        os.makedirs(perf_path, exist_ok=True)

        import matplotlib

        if not args.show_plots:
            matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages

        for d in range(X_train.shape[1]):

            metrics_d = perf_history[0][d].keys()
            print('Metrics for feature {}: {}'.format(d, metrics_d))

            for m in metrics_d:

                perf_d = [p[d][m] for p in perf_history]
                fig, ax = plt.subplots(figsize=args.fig_size)
                ax.plot(pp_counts, perf_d, label=m)
                ax.legend()

                perf_d_output = os.path.join(perf_path, '{}-f{}-perf.pdf'.format(m, d))
                pp = PdfPages(perf_d_output)
                pp.savefig(fig)
                pp.close()
                print('Saved perf {} for feature {} to {}'.format(m, d, perf_d_output))

                if args.show_plots:
                    plt.show()

                #
                # explaining data
                # FIXME: work on data explanaition

                # Wint = 2
                # explain_path = os.path.join(output_path, 'explain@{}iter'.format(args.iters))
                # os.makedirs(explain_path, exist_ok=True)

                # explain_data(X, C, R,
                #              (Yreal, Yint, Ypos,  Ycat, Yord, Ycount), Wint,
                #              theta, theta_L, theta_H,
                #              maxX, minX, meanX,
                #              fig_size=args.fig_size,
                #              output_path=explain_path,
                #              show=args.show_explain)
