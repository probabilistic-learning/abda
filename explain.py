"""
Routines to plot data explanations out of the samplers

@author: antonio vergari

"""
import os

import numpy as np
from utils import fre,  f_pos, f_int
from utils import f_cat, f_ord, f_count


def show_save_explanaition(x, f_rec, fig_size=(10, 7), marker_size=70,
                           output=None, show=False):

    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    sample_range = list(range(len(x)))

    fig, ax = plt.subplots(figsize=fig_size)

    ax.scatter(sample_range, x, alpha=0.5, label='x', marker='x', s=marker_size)
    ax.scatter(sample_range, f_rec, alpha=0.5, label='f(y)', marker='+', s=marker_size + 20)
    ax.legend()

    if output is not None:
        pp = PdfPages(output + '.pdf')
        pp.savefig(fig)
        pp.close()

    if show:
        plt.show()


def explain_real_data(x, y, w, mu,
                      fig_size=(10, 7),  output=None, show=False):
    """
    Vaidates the transformation f_{real}(y) over some data X of known real type

    ---
    input:

    x : float array
        feature array (size N) of floating point numbers representing a real valued attribute
    y : float array
        pseudo-observation array (size N), to be transformed into real valued observations
    w : float
        scaling factor
    mu: float
        translation factor
    """

    f_rec = fre(y, w, mu)

    # print(f_rec.shape, y.shape)

    show_save_explanaition(x, f_rec, fig_size=fig_size,
                           output=output, show=show)


def explain_positive_data(x, y, w, fig_size=(10, 7), output=None, show=False):
    """
    Vaidates the transformation f_{pos}(y) over some data X of known real positive type

    ---
    input:

    x : float array
        feature array (size N) of floating point numbers representing a real positive valued attribute
    y : float array
        pseudo-observation array (size N), to be transformed into real positive valued observations
    w : float
        scaling factor
    """

    f_rec = f_pos(y, w)

    # print(f_rec.shape, y.shape)
    show_save_explanaition(x, f_rec, fig_size=fig_size,
                           output=output, show=show)


def explain_interval_data(x, y, w, theta_L, theta_H, fig_size=(10, 7), output=None, show=False):
    """
    Vaidates the transformation f_{int}(y) over some data X of known real interval type

    ---
    input:

    x : float array
        feature array (size N) of floating point numbers representing a real interval valued attribute
    y : float array
        pseudo-observation array (size N), to be transformed into real interval valued observations
    w : float
        scaling factor
    theta_L : float
        low interval bound
    theta_H : float
        high interval bound
    """

    f_rec = f_int(y, w, theta_L, theta_H)

    # print(f_rec.shape, y.shape)
    show_save_explanaition(x, f_rec, fig_size=fig_size,
                           output=output, show=show)


def explain_categorical_data(x, y, Rd, fig_size=(10, 7), output=None, show=False):
    """
    Vaidates the transformation f_{cat}(y) over some data X of known categorical type

    ---
    input:

    x : uint array
        feature array (size N) of uint numbers representing a categorical attribute
    y : float array
        pseudo-observation array (size RxN), to be transformed into categorical observations
    """

    f_rec = f_cat(y, Rd)

    # print(f_rec.shape, y.shape)
    show_save_explanaition(x, f_rec, fig_size=fig_size,
                           output=output, show=show)


def explain_ordinal_data(x, y, theta, Rd, fig_size=(10, 7), output=None, show=False):
    """
    Vaidates the transformation f_{ord}(y) over some data X of known ordinal type

    ---
    input:

    x : uint array
        feature array (size N) of uint numbers representing a ordinal attribute
    y : float array
        pseudo-observation array (size RxN), to be transformed into ordinal observations
    """

    f_rec = f_ord(y, theta, Rd)

    # print(f_rec.shape, y.shape)
    show_save_explanaition(x, f_rec, fig_size=fig_size,
                           output=output, show=show)


def explain_count_data(x, y, w, fig_size=(10, 7), output=None, show=False):
    """
    Vaidates the transformation f_{count}(y) over some data X of known count type

    ---
    input:

    x : uint array
        feature array (size N) of uint numbers representing a count attribute
    y : float array
        pseudo-observation array (size RxN), to be transformed into count observations
    """

    f_rec = f_count(y,  w)

    # print(f_rec.shape, y.shape)
    show_save_explanaition(x, f_rec, fig_size=fig_size,
                           output=output, show=show)


def explain_data(X, C, R,
                 Ys, Wint,
                 theta, theta_L, theta_H,
                 maxX, minX, meanX,
                 fig_size=(10, 7),
                 output_path=None,
                 show=True):

    N, D = X.shape

    Yreal, Yint, Ypos, Ycat, Yord, Ycount = Ys

    for j, d in enumerate(range(D)):

        print('explaining feature {}/{}'.format(j + 1, D))

        print('Explaining feature {}/{}'.format(d + 1, D))

        o_path = os.path.join(output_path, 'd{}'.format(d))

        if C[d] == 1:
            fo_path = '{}-real'.format(o_path)
            explain_real_data(X[:, d], Yreal[d, :], (maxX[d] - meanX[d]) / 2,
                              meanX[d],
                              fig_size=fig_size, output=fo_path, show=show)

            fo_path = '{}-int'.format(o_path)
            explain_interval_data(X[:, d], Yint[d, :], 1 / Wint, theta_L[d], theta_H[d],
                                  fig_size=fig_size, output=fo_path, show=show)

            fo_path = '{}-pos'.format(o_path)
            explain_positive_data(X[:, d], Ypos[d, :],  maxX[d] / 2,
                                  fig_size=fig_size, output=fo_path, show=show)

        elif C[d] == 2:
            fo_path = '{}-real'.format(o_path)
            explain_real_data(X[:, d], Yreal[d, :], (maxX[d] - meanX[d]) / 2,
                              meanX[d],
                              fig_size=fig_size, output=fo_path, show=show)

            fo_path = '{}-int'.format(o_path)
            explain_interval_data(X[:, d], Yint[d, :], 1 / Wint, theta_L[d], theta_H[d],
                                  fig_size=fig_size, output=fo_path, show=show)

        elif C[d] == 3:
            pass
        elif C[d] == 4:
            fo_path = '{}-cat'.format(o_path)
            explain_categorical_data(X[:, d], Ycat[d, :, :], R[d],
                                     fig_size=fig_size, output=fo_path, show=show)

            fo_path = '{}-ord'.format(o_path)
            explain_ordinal_data(X[:, d], Yord[d, :], theta[d], R[d],
                                 fig_size=fig_size, output=fo_path, show=show)

            fo_path = '{}-count'.format(o_path)
            explain_count_data(X[:, d], Ycount[d, :],  maxX[d] / 2,
                               fig_size=fig_size, output=fo_path, show=show)


from functools import partial
from numpy.linalg import norm
from sklearn.metrics import mean_squared_error, mean_squared_log_error, mean_absolute_error, zero_one_loss


def root_mean_squared_error(X_true, X_pred):
    return np.sqrt(mean_squared_error(X_true, X_pred))


def mean_norm_error(err_f, X_true, X_pred, X_min, X_max, X_mean, X_std):
    # X_true_mean = X_true.mean()
    # return err_f(X_true, X_pred) / X_true_mean
    # return err_f(X_true, X_pred) / X_mean
    return err_f(X_true, X_pred)


def std_err_norm_error(err_f, X_true, X_pred, X_min, X_max, X_mean, X_std):
    # X_stddev = X_true.std()
    # X_std_err = X_stddev / np.sqrt(len(X_true))
    # return err_f(X_true, X_pred) / X_std_err
    X_std_err = np.sqrt(np.sum(np.power(X_mean - X_true, 2)))
    # return err_f(X_true, X_pred) / X_std_err
    return err_f(X_true, X_pred)


def range_norm_error(err_f, X_true, X_pred, X_min, X_max, X_mean, X_std):
    # X_range = X_true.max() - X_true.min()
    X_range = X_max - X_min
    # return err_f(X_true, X_pred) / X_range
    return err_f(X_true, X_pred)


def rmse(X_true, X_pred, X_min, X_max, X_mean, X_std):
    return root_mean_squared_error(X_true, X_pred)


def m_rmse(X_true, X_pred, X_min, X_max, X_mean, X_std):
    return root_mean_squared_error(X_true, X_pred) / X_mean


def r_rmse(X_true, X_pred, X_min, X_max, X_mean, X_std):
    return root_mean_squared_error(X_true, X_pred) / (X_max - X_min)


def s_rmse(X_true, X_pred, X_min, X_max, X_mean, X_std):
    return root_mean_squared_error(X_true, X_pred) / (X_std)


def mse(X_true, X_pred, X_min, X_max, X_mean, X_std):
    return mean_squared_error(X_true, X_pred)


def msle(X_true, X_pred, X_min, X_max, X_mean, X_std):
    return mean_squared_log_error(X_true, X_pred)


def mae(X_true, X_pred, X_min, X_max, X_mean, X_std):
    return mean_absolute_error(X_true, X_pred)


def zero_one(X_true, X_pred, X_min, X_max, X_mean, X_std):
    return zero_one_loss(X_true, X_pred)


mean_norm_root_mean_squared_error = partial(mean_norm_error,
                                            err_f=rmse)
std_err_norm_root_mean_squared_error = partial(std_err_norm_error,
                                               err_f=rmse)
range_norm_root_mean_squared_error = partial(range_norm_error,
                                             err_f=rmse)


METRICS_DICT = {'RMSE': rmse,
                # 'M-RMSE': mean_norm_root_mean_squared_error,
                # 'R-RMSE': range_norm_root_mean_squared_error,
                # 'S-RMSE': std_err_norm_root_mean_squared_error,
                'M-RMSE': m_rmse,
                'R-RMSE': r_rmse,
                'S-RMSE': s_rmse,
                'MSE': mse,
                'MSLE': msle,
                'MAE': mae,
                'ACC': zero_one}

from spn.structure.StatisticalTypes import MetaType
SCORE_LIST_MAP = {MetaType.REAL: ['MSE', 'MAE', 'RMSE', 'M-RMSE', 'R-RMSE', 'S-RMSE'],
                  MetaType.DISCRETE: ['MSE', 'MAE', 'ACC', 'MSLE', 'RMSE', 'M-RMSE', 'R-RMSE', 'S-RMSE']}


import numba
from numba import jit, uint8, int64, int32, float64, optional, boolean
from numba.types import Tuple

# @numba.jit(nopython=True)


@numba.jit(float64[:, :](
    # X
    float64[:, :],
    #   C           R
    uint8[:], int64[:],
    # S
    int32[:, :],
    # Ys
    Tuple((float64[:, :], float64[:, :], float64[:, :],
           float64[:, :, :], float64[:, :], float64[:, :])),
    # Wint
    int64,
    # theta, theta_L, theta_H,
    float64[:, :], float64[:], float64[:],
    # maxX, meanX, minX,
    float64[:], float64[:], float64[:]),
    # nopython=True
)
def predict_data(X, C, R, S,
                 Ys, Wint,
                 theta, theta_L, theta_H,
                 maxX, minX, meanX,
                 # select_best=True
                 ):

    N, D = X.shape

    X_hat = np.zeros((N, D))
    X_hat[:, :] = np.inf

    #
    # doing this likelihood wise
    L = 6
    XL_hat = np.zeros((N, D, L))
    XL_hat[:, :, :] = np.nan

    Yreal, Yint, Ypos, Ycat, Yord, Ycount = Ys

    print(np.isnan(Yreal).sum(),
          np.isnan(Yint).sum(),
          np.isnan(Ypos).sum(),
          np.isnan(Ycat).sum(),
          np.isnan(Yord).sum(),
          np.isnan(Ycount).sum())

    miss_vals = np.any(np.isnan(X))
    for d in range(D):

        sids = None
        #
        # dealing with missing values?
        if miss_vals:
            sids = np.isnan(X[:, d])
        else:
            sids = np.arange(X.shape[0])
            # sids = np.ones(X.shape[0])

        x_hat = None
        if C[d] == 1:

            #
            # real
            XL_hat[sids, d, 0] = fre(Yreal[d, sids], (maxX[d] - meanX[d]) / 2,
                                     meanX[d])

            #
            # int
            XL_hat[sids, d, 1] = f_int(Yint[d, sids], 1 / Wint, theta_L[d], theta_H[d])

            #
            # pos
            XL_hat[sids, d, 2] = f_pos(Ypos[d, sids],  maxX[d] / 2)

        elif C[d] == 2:

            #
            # real
            XL_hat[sids, d, 0] = fre(Yreal[d, sids], (maxX[d] - meanX[d]) / 2,
                                     meanX[d])

            #
            # int
            XL_hat[sids, d, 1] = f_int(Yint[d, sids], 1 / Wint, theta_L[d], theta_H[d])

        elif C[d] == 3:
            pass

        elif C[d] == 4:
            #
            # cat
            # print(Ycat[d, :, sids].shape, Ycat.shape)
            XL_hat[sids, d, 3] = f_cat(Ycat[d, :, sids].T, R[d])

            #
            # ord
            XL_hat[sids, d, 4] = f_ord(Yord[d, sids], theta[d], R[d])

            #
            # count
            XL_hat[sids, d, 5] = f_count(Ycount[d, sids],  maxX[d] / 2)

    print('XLHAT nan', np.isnan(XL_hat).sum())
    print('XHAT nan', np.isnan(X_hat).sum())
    #
    # computing only predictions according to the currest best type in S?
    # if not select_best:
    #     return XL_hat
    # else:
    for n in range(N):
        for d in range(D):
            if C[d] == 1:
                if S[n, d] == 1:
                    X_hat[n, d] = XL_hat[n, d, 0]
                elif S[n, d] == 2:
                    X_hat[n, d] = XL_hat[n, d, 1]
                elif S[n, d] == 4:
                    X_hat[n, d] = XL_hat[n, d, 2]
                # else:
                #     raise ValueError(
                #         'Unrecognized type S {} for feature {} (C={})'.format(S[n, d], d, C[d]))
            elif C[d] == 2:
                if S[n, d] == 1:
                    X_hat[n, d] = XL_hat[n, d, 0]
                elif S[n, d] == 2:
                    X_hat[n, d] = XL_hat[n, d, 1]
                # else:
                #     raise ValueError(
                #         'Unrecognized type S {} for feature {} (C={})'.format(S[n, d], d, C[d]))
            elif C[d] == 3:
                pass
            elif C[d] == 4:
                if S[n, d] == 1:
                    X_hat[n, d] = XL_hat[n, d, 3]
                elif S[n, d] == 2:
                    X_hat[n, d] = XL_hat[n, d, 4]
                elif S[n, d] == 3:
                    X_hat[n, d] = XL_hat[n, d, 5]
                # else:
                #     raise ValueError(
                #         'Unrecognized type S {} for feature {} (C={})'.format(S[n, d], d, C[d]))

    if miss_vals:
        assert np.isnan(X).sum() == (~np.isnan(X_hat)).sum()  # , "{} {}".format(np.isnan(X).sum(),
        #                  (~np.isnan(X_hat)).sum())
    else:
        assert np.isnan(X_hat).sum() == 0

    return X_hat


from collections import defaultdict


def print_perf_dict(perf_dict, feature_ids=None, metrics=['MSE', 'MSLE', 'MAE', 'ACC']):

    if feature_ids is None:
        feature_ids = perf_dict.keys()

    for d in feature_ids:
        print('{}\t{}'.format(d, '\t'.join('{}:{}'.format(m, str(perf_dict[d][m]))
                                           for m in sorted(metrics) if m in perf_dict[d])))


def eval_predict_data(X, X_orig, C, R, S,
                      Ys, Wint,
                      theta, theta_L, theta_H,
                      maxX, minX, meanX, stdX,
                      continuous_metrics=['MSE', 'MAE'],
                      # pos_continuous_metrics=['MSE', 'MSLE', 'MAE'],
                      discrete_metrics=['MSE', 'MSLE', 'MAE', 'ACC']):

    X_hat = predict_data(X, C, R, S,
                         Ys, Wint,
                         theta, theta_L, theta_H,
                         maxX, minX, meanX)

    N, D = X.shape

    perf_dict = defaultdict(dict)
    # perf_dict = {}

    miss_vals = np.any(np.isnan(X))

    X_true = None
    if miss_vals:
        X_true = X_orig
    else:
        X_true = X

    for d in range(D):

        # perf_dict[d] = {}

        sids = None
        #
        # dealing with missing values?
        if miss_vals:
            sids = (np.isnan(X[:, d]) & ~np.isnan(X_orig[:, d]))
        else:
            sids = np.arange(X.shape[0])

        metrics = None
        if C[d] == 3:
            continue
        elif C[d] == 1 or C[d] == 2:
            metrics = continuous_metrics
        # elif C[d] == 2:
        #     metrics = continuous_metrics
        elif C[d] == 4:
            metrics = discrete_metrics

        for m in metrics:

            # print('m', m, d, 'C', C[d], metrics, 'S', S[:, d], 'XH', X_hat[:, d], 'X', X[:, d])
            score_func = METRICS_DICT[m]
            s = score_func(X_true[sids, d], X_hat[sids, d], minX[d], maxX[d], meanX[d], stdX[d])
            perf_dict[d][m] = s

    return X_hat, perf_dict


def compute_perf_dict_miss(X, X_orig, X_hat, X_mask, C,
                           X_min, X_max, X_mean, X_std,
                           continuous_metrics=['MSE', 'MAE'],
                           # pos_continuous_metrics=['MSE', 'MSLE', 'MAE'],
                           discrete_metrics=['MSE', 'MSLE', 'MAE', 'ACC']):

    N, D = X.shape

    assert X_hat.shape[0] == N, X_hat.shape
    assert X_hat.shape[1] == D, X_hat.shape
    assert X_min.shape[0] == D
    assert X_max.shape[0] == D
    assert X_mean.shape[0] == D
    assert X_std.shape[0] == D

    perf_dict = defaultdict(dict)
    # perf_dict = {}

    X_true = X_orig

    for d in range(D):

        # perf_dict[d] = {}

        sids = None
        #
        # dealing with missing values?
        sids = (np.isnan(X[:, d]) & ~np.isnan(X_orig[:, d]) & X_mask[:, d])

        # print(sids.sum(), np.isnan(X[:, d]) & ~np.isnan(X_orig[:, d]))
        metrics = None
        if C[d] == 3:
            continue
        elif C[d] == 1 or C[d] == 2:
            metrics = continuous_metrics
        # elif C[d] == 2:
        #     metrics = continuous_metrics
        elif C[d] == 4:
            metrics = discrete_metrics

        for m in metrics:

            # print('m', m, d, 'C', C[d], metrics)
            score_func = METRICS_DICT[m]
            s = score_func(X_true[sids, d], X_hat[sids, d], X_min[d], X_max[d], X_mean[d], X_std[d])
            perf_dict[d][m] = s

    return perf_dict


def strong_and(p, r):
    return max(r + p - 1, 0)


def weak_and(p, r):
    return min(r, p)


def prod_and(p, r):
    return p * r


from numpy.testing import assert_array_almost_equal


def compute_confusion_matrix(R, P, hard=False, soft_rule='prod', unnorm=False):
    """
    Computes the confution matrix against two collected samples
    R (reference truth) with shape NxC where N is the number of samples and C the number of classes
    and P (predictions) with same shape.
    Both soft and computations considered

    see [1]

    [1] - http://softclassval.r-forge.r-project.org/blob/Beleites-20111020-RKI.pdf
    """

    import itertools
    from sklearn.metrics import confusion_matrix

    N, C = R.shape
    assert P.shape == R.shape, P.shape

    if not unnorm:
        assert_array_almost_equal(R.sum(axis=1), np.ones(N))
        assert_array_almost_equal(P.sum(axis=1), np.ones(N))

    if hard:
        #
        # from soft predictions to hard counts
        R_best = np.argmax(R, axis=1)
        Rb = np.zeros((N, C), dtype=np.int64)
        Rb[:, R_best] = 1
        R = Rb

        P_best = np.argmax(P, axis=1)
        Pb = np.zeros((N, C), dtype=np.int64)
        Pb[:, P_best] = 1
        P = Pb

        assert R_best.shape[0] == N
        assert P_best.shape[0] == N

        return confusion_matrix(R_best, P_best)

    #
    # computing matrix
    conf_mat = np.zeros((N, C, C))

    aggr_func = None
    if soft_rule == 'strong-AND':
        aggr_func = strong_and
    elif soft_rule == 'weak-AND':
        aggr_func = weak_and
    elif soft_rule == 'prod':
        aggr_func = prod_and

    for i, j in itertools.product(range(C), range(C)):
        print(i, j)

        for n in range(N):
            conf_mat[n, i, j] = aggr_func(R[n, i],  P[n, j])

    #
    # aggregate over samples
    conf_mat = conf_mat.sum(axis=0)

    assert conf_mat.shape[0] == C, conf_mat.shape
    assert conf_mat.shape[1] == C, conf_mat.shape

    return conf_mat
