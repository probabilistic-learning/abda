"""
Statistical Type Discovery by Latent Variable Feature Modeling

@author: antonio vergari

--------
General translation dictionary and naming scheme:

D : uint
    number of features/ random variables (RVs)
N : uint
    number of instances
T : uint array
    meta-type array (size D), encoding
    {
      1: real (w positive: all real | positive | interval)
      2: real (w/o positive: all real | interval)
      3: binary data
      4: discrete (non-binary: categorical | ordinal | count)
    }
R : uint array
    max-domain-cardinality array (size D) for discrete (meta-type 4) RVs

Z : float matrix
    latent space representation (size NxK)

B : float matrix
    latent space assignments (size KxD)

W : float matrix
    weight array (size Dx4 where 4 is the number of types per meta-types)
    TODO: refactor this into 3

"""

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
from numpy.testing import assert_array_almost_equal
import scipy.io
from scipy.special import logsumexp

import numba

from utils import is_missing_value
from utils import fre_1, f_1, fint_1, df_1
from utils import inverse
from utils import xpdf_re, xpdf_pos, xpdf_int
from utils import mvnrnd, mnrnd, truncnormrnd
# from tfspn.utils import phi
from utils import phi_erf_numba as phi
from utils import cat_pdf_numba, cat_pdf_numba_prealloc, logsumexp_3D_numba
from explain import explain_data, print_perf_dict, eval_predict_data
from utils import running_avg_numba
from spn.structure.StatisticalTypes import MetaType
#
# TODOD: change it to be nana
MISS_VALUE = np.nan

RANDOM_SEED = 0

INT_EXP = .04

LOG_ZERO = -50

U_MAX = 500


def dump_samples_to_pickle(samples, output_path, out_file_name, key, count_key='id'):
    pic_path = os.path.join(output_path, out_file_name)
    counts = np.array([s[count_key] for s in samples if key in s])
    k_hist = [s[key] for s in samples if key in s]
    with gzip.open(pic_path, 'wb') as f:
        res = {count_key: counts,
               key: k_hist}
        pickle.dump(res, f)
        print('Dumped {} to {}'.format(count_key, pic_path))


def extend_matrix_one_value_per_row_nonzero(X, M, M_orig):
    N, D = X.shape

    assert X.shape[0] == M.shape[1], '{} {}'.format(X.shape, M.shape)
    assert X.shape[1] == M.shape[0], '{} {}'.format(X.shape, M.shape)
    assert M.shape == M_orig.shape

    ext_X = []
    for n in range(N):
        for d in range(D):
            x_n_d = X[n, d]
            # if x_n_d != 0.0:
            if np.isnan(M[d, n]) and not np.isnan(M_orig[d, n]):
                ext_X.append(x_n_d)

    ext_X = np.array(ext_X)
    # r_z, c_z = np.nonzero(X)
    assert ext_X.shape[0] == (np.isnan(M) & ~np.isnan(M_orig)).sum()
    return ext_X


def init_weights(T):
    """
    Initing W matrix according to type.
    NOTE: cannot put Dirichlet hyperparameters to 0 (strangely it worked on Isabel's code)

    TODO: remove the useless columns and make the matrix Dx3 (no dir RV)

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
        weight array (size Dx4 where 4 is the number of types per meta-types)
        TODO: refactor this into 3
    """

    D = T.shape[0]
    W = np.zeros((D, 4))

    #
    # initing W by setting a Dirichlet's hyperparameters according to a RV meta-type
    for d in range(D):

        if T[d] == 1:
            W[d] = np.array([100, 100, 1, 100])

        elif T[d] == 2:
            W[d] = np.array([100, 100, 1, 1])

        elif T[d] == 3:
            #
            # we could skip this since W has been inited by zero
            # W[d] = np.array([0, 0, 0, 0])a
            # pass
            W[d] += 1

        elif T[d] == 4:
            W[d] = np.array([100, 100, 100, 1])

    return W


def preprocess_positive_real_data(X, T, eps=1e-6):
    """
    Adding a small positive epsilon to continuous data to avoid
    numerical inaccuracies
    """

    D = X.shape[1]

    assert T.shape[0] == D, T.shape

    for d in range(D):
        #
        # to avoid numerical errors in positive real data
        if T[d] == 1 or T[d] == 2:
            X[np.isclose(X[:, d], 0), d] = eps

    return X


from numba.types import Tuple, pyobject
from numba import jit, uint8, int64, int32, float64, optional, boolean


@numba.jit(Tuple((float64[:, :], float64[:, :]))(
    # X              kest
    float64[:, :], int64,
    #   C           R
    uint8[:], int64[:],
    # muZ
    float64[:],
    # SZ
    float64[:, :],
    #  s2Z,   s2Y,
    float64, float64,
    # aux
    float64[:],
    # Ps
    optional(float64[:, :, :]),
    optional(float64[:, :, :]),
    optional(float64[:, :, :]),
    optional(float64[:, :, :]), optional(float64[:, :, :, :]),
    optional(float64[:, :, :]), optional(float64[:, :, :]),
    # Bs
    optional(float64[:, :]),
    optional(float64[:, :]),
    optional(float64[:, :]),
    optional(float64[:, :]), optional(float64[:, :, :]),
    optional(float64[:, :]), optional(float64[:, :]),
    # Ys
    float64[:, :],
    float64[:, :],
    float64[:, :],
    float64[:, :], float64[:, :, :],
    float64[:, :], float64[:, :],
    # EYE
    float64[:, :]), nopython=True)
def sampling_Z_numba(X, Kest, C, R,
                     muZ,
                     SZ, s2Z, s2Y,
                     aux,
                     Preal, Ppos, Pint, Pbin, Pcat, Pord, Pcount,
                     Breal, Bpos, Bint, Bbin, Bcat, Bord, Bcount,
                     Yreal, Ypos, Yint, Ybin, Ycat, Yord, Ycount,
                     EYE,
                     # rand_gen
                     ):
    """
    """

    N, D = X.shape

    auxM = np.zeros((N, Kest))

    for n in range(N):

        # muZ[:] = 0
        # SZ[:, :] = 0
        # aux[:] = 0

        muZ.fill(0)
        SZ.fill(0)
        aux.fill(0)

        for d in range(D):

            if C[d] == 1:

                SZ += Preal[d]
                aux = np.copy(Breal[d])
                aux = aux * Yreal[d, n]
                muZ += aux

                SZ += Pint[d]
                aux = np.copy(Bint[d])
                aux = aux * Yint[d, n]
                muZ += aux

                SZ += Ppos[d]
                aux = np.copy(Bpos[d])
                aux = aux * Ypos[d, n]
                muZ += aux

            elif C[d] == 2:

                SZ += Preal[d]
                aux = np.copy(Breal[d])
                aux = aux * Yreal[d, n]
                muZ += aux

                SZ += Pint[d]
                aux = np.copy(Bint[d])
                aux = aux * Yint[d, n]
                muZ += aux

            elif C[d] == 3:

                SZ += Pbin[d]
                aux = np.copy(Bbin[d])
                aux = aux * Ybin[d, n]
                muZ += aux

            elif C[d] == 4:

                SZ += Pord[d]
                aux = np.copy(Bord[d])
                aux = aux * Yord[d, n]
                muZ += aux

                SZ += Pcount[d]
                aux = np.copy(Bcount[d])
                aux = aux * Ycount[d, n]
                muZ += aux

                #
                # TODO: refactor from 'r' to 'xnd' to avoid confusion
                # also, casting is necessary
                r = int(X[n, d])
                auxY = -np.inf

                if r == -1 or np.isnan(r):
                    for r2 in range(R[d]):
                        if Ycat[d, r2, n] > auxY:
                            r = r2
                else:
                    r -= 1
                    SZ += Pcat[d, r]
                    if r > 0:
                        aux = np.copy(Bcat[d, r])
                        aux = aux * Ycat[d, r, n]
                        muZ += aux

        SZ = 1 / s2Z * np.dot(EYE, EYE) + 1 / s2Y * SZ
        SZ = inverse(SZ)
        aux = 1 / s2Y * np.dot(SZ, muZ)
        # Z[:, n] = mvnrnd(aux, SZ, rand_gen)
        auxM[n, :] = aux

    # return Z
    return auxM, SZ


# @numba.jit
# def sampling_Bs(X, Z, d, Kest, C, s2Y, s2B, SB, aux,
#                 Breal, Yreal,
#                 Bint, Yint,
#                 Bpos, Ypos,
#                 Bbin, Ybin,
#                 Bcat, Ycat,
#                 Bord, Yord,
#                 Bcount, Ycount, rand_gen):
#     """
#     """

#     N, D = X.shape

#     if C[d] == 1:

#         SB = 1 / s2Y * np.dot(Z, Z.T) + 1 / s2B * SB
#         SB = inverse(SB)

#         muB = 1 / s2Y * np.dot(Z, Yreal[d])
#         np.dot(SB, muB, out=aux)
#         Breal[d] = mvnrnd(aux, SB, rand_gen)

#         muB = 1 / s2Y * np.dot(Z, Ypos[d])
#         np.dot(SB, muB, out=aux)
#         Bpos[d] = mvnrnd(aux, SB, rand_gen)

#         muB = 1 / s2Y * np.dot(Z, Yint[d])
#         np.dot(SB, muB, out=aux)
#         Bint[d] = mvnrnd(aux, SB, rand_gen)

#     elif C[d] == 2:
#         SB = 1 / s2Y * np.dot(Z, Z.T) + 1 / s2B * SB
#         SB = inverse(SB)

#         muB = 1 / s2Y * np.dot(Z, Yreal[d])
#         np.dot(SB, muB, out=aux)
#         Breal[d] = mvnrnd(aux, SB, rand_gen)

#         muB = 1 / s2Y * np.dot(Z, Yint[d])
#         np.dot(SB, muB, out=aux)
#         Bint[d] = mvnrnd(aux, SB, rand_gen)

#     elif C[d] == 3:
#         SB = 1 / s2Y * np.dot(Z, Z.T) + 1 / s2B * SB
#         SB = inverse(SB)

#         muB = 1 / s2Y * np.dot(Z, Ybin[d])
#         np.dot(SB, muB, out=aux)
#         Bbin[d] = mvnrnd(aux, SB, rand_gen)

#     elif C[d] == 4:
#         SB = 1 / s2Y * np.dot(Z, Z.T) + 1 / s2B * SB
#         SB = inverse(SB)

#         for r in range(1, R[d]):
#             muB = 1 / s2Y * np.dot(Z, Ycat[d, r])
#             np.dot(SB, muB, out=aux)
#             Bcat[d, r] = mvnrnd(aux, SB, rand_gen)

#         muB = 1 / s2Y * np.dot(Z, Yord[d])
#         np.dot(SB, muB, out=aux)
#         Bord[d] = mvnrnd(aux, SB, rand_gen)

#         muB = 1 / s2Y * np.dot(Z, Ycount[d])
#         np.dot(SB, muB, out=aux)
#         Bcount[d] = mvnrnd(aux, SB, rand_gen)

#     return Breal, Bint, Bpos, Bbin, Bcat, Bord, Bcount


# @numba.jit
@numba.jit(Tuple((float64[:, :],
                  float64[:, :],
                  float64[:, :],
                  float64[:, :], float64[:, :, :],
                  float64[:, :], float64[:, :]))(
    # X            Z              d kest
    float64[:, :], float64[:, :], int64, int64,
    #   C           R      S
    uint8[:], int64[:], int32[:, :],
    #  s2Y, s2B, s2U,
    float64, float64, float64,
    # sY, sYd, s2theta,
    float64, float64, float64,
    # aux
    float64[:],
    # Bs
    optional(float64[:, :]),
    optional(float64[:, :]),
    optional(float64[:, :]),
    optional(float64[:, :]), optional(float64[:, :, :]),
    optional(float64[:, :]), optional(float64[:, :]),
    # Ys
    float64[:, :],
    float64[:, :],
    float64[:, :],
    float64[:, :], float64[:, :, :],
    float64[:, :], float64[:, :],
    # maxX, meanX, minX,
    float64[:], float64[:], float64[:],
    # theta, theta_L, theta_H,
    float64[:, :], float64[:], float64[:],
    # Wint
    int64, pyobject
), nopython=False)
def sampling_Ys_numba(X, Z, d,
                      Kest,
                      C, R, S,
                      s2Y, s2B, s2U,
                      # SB,
                      sY, sYd, s2theta,
                      aux,
                      Breal, Bpos, Bint, Bbin, Bcat, Bord, Bcount,
                      Yreal, Ypos, Yint, Ybin, Ycat, Yord, Ycount,
                      maxX, meanX, minX,
                      theta, theta_L, theta_H,
                      Wint,
                      rand_gen):
    """
    """

    N, D = X.shape

    xnd = 0.0

    for d in range(D):
        if C[d] == 1:

            for n in range(N):

                xnd = X[n, d]

                if S[n, d] == 1:

                    Zn = Z[:, n]
                    muy = np.dot(Zn, Breal[d])

                    if xnd == -1 or np.isnan(xnd):
                        Yreal[d, n] = muy + rand_gen.normal(loc=0, scale=sY)
                    else:
                        Yreal[d, n] = (fre_1(xnd, 2 / (maxX[d] - meanX[d]), meanX[d]) / s2U + muy / s2Y) / (
                            1 / s2Y + 1 / s2U) + rand_gen.normal(loc=0, scale=np.sqrt(1 / (1 / s2Y + 1 / s2U)))

                    muy = np.dot(Zn, Bint[d])
                    Yint[d, n] = muy + rand_gen.normal(loc=0, scale=sY)

                    muy = np.dot(Zn, Bpos[d])
                    Ypos[d, n] = muy + rand_gen.normal(loc=0, scale=sY)

                elif S[n, d] == 2:

                    Zn = Z[:, n]
                    muy = np.dot(Zn, Bint[d])

                    if xnd == -1 or np.isnan(xnd):
                        Yint[d, n] = muy + rand_gen.normal(loc=0, scale=sY)
                    else:
                        Yint[d, n] = (fint_1(xnd, Wint, theta_L[d], theta_H[d]) / s2U + muy / s2Y) / (
                            1 / s2Y + 1 / s2U) + rand_gen.normal(loc=0, scale=np.sqrt(1 / (1 / s2Y + 1 / s2U)))

                    muy = np.dot(Zn, Breal[d])
                    Yreal[d, n] = muy + rand_gen.normal(loc=0, scale=sY)

                    muy = np.dot(Zn, Bpos[d])
                    Ypos[d, n] = muy + rand_gen.normal(loc=0, scale=sY)

                #
                # NOTE on Isabel's code this correponds to S[n, d] == 4
                elif S[n, d] == 4:
                    # elif S[n, d] == 3:
                    Zn = Z[:, n]
                    muy = np.dot(Zn, Bpos[d])

                    if xnd == -1 or np.isnan(xnd):
                        Ypos[d, n] = muy + rand_gen.normal(loc=0, scale=sY)
                    else:
                        Ypos[d, n] = (f_1(xnd, 2 / maxX[d]) / s2U + muy / s2Y) / (1 / s2Y +
                                                                                  1 / s2U) + rand_gen.normal(loc=0, scale=np.sqrt(1 / (1 / s2Y + 1 / s2U)))

                    muy = np.dot(Zn, Breal[d])
                    Yreal[d, n] = muy + rand_gen.normal(loc=0, scale=sY)

                    muy = np.dot(Zn, Bint[d])
                    Yint[d, n] = muy + rand_gen.normal(loc=0, scale=sY)

        #
        # just like before, C[d] == 1 but with no pos
        elif C[d] == 2:

            for n in range(N):

                xnd = X[n, d]

                if S[n, d] == 1:

                    Zn = Z[:, n]
                    muy = np.dot(Zn, Breal[d])

                    if xnd == -1 or np.isnan(xnd):
                        Yreal[d, n] = muy + rand_gen.normal(loc=0, scale=sY)
                    else:
                        Yreal[d, n] = (fre_1(xnd, 2 / (maxX[d] - meanX[d]), meanX[d]) / s2U + muy / s2Y) / (
                            1 / s2Y + 1 / s2U) + rand_gen.normal(loc=0, scale=np.sqrt(1 / (1 / s2Y + 1 / s2U)))

                    muy = np.dot(Zn, Bint[d])
                    Yint[d, n] = muy + rand_gen.normal(loc=0, scale=sY)

                elif S[n, d] == 2:

                    Zn = Z[:, n]
                    muy = np.dot(Zn, Bint[d])

                    if xnd == -1 or np.isnan(xnd):
                        Yint[d, n] = muy + rand_gen.normal(loc=0, scale=sY)
                    else:
                        Yint[d, n] = (fint_1(xnd, Wint, theta_L[d], theta_H[d]) / s2U + muy / s2Y) / (
                            1 / s2Y + 1 / s2U) + rand_gen.normal(loc=0, scale=np.sqrt(1 / (1 / s2Y + 1 / s2U)))

                    muy = np.dot(Zn, Breal[d])
                    Yreal[d, n] = muy + rand_gen.normal(loc=0, scale=sY)

        elif C[d] == 3:
            for n in range(N):

                Zn = Z[:, n]
                muy = np.dot(Zn, Bbin[d])

                if xnd == -1 or np.isnan(xnd):
                    Ybin[d, n] = muy + rand_gen.normal(loc=0, scale=sY)
                elif xnd == 1:
                    xnd = int(X[n, d])
                    Ybin[d, n] = truncnormrnd(muy, sY, - np.inf, 0, rand_gen=rand_gen)
                elif xnd == 2:
                    xnd = int(X[n, d])
                    Ybin[d, n] = truncnormrnd(muy, sY, 0, np.inf, rand_gen=rand_gen)

        elif C[d] == 4:

            muyCat = np.zeros(R[d])
            Ymax = np.zeros(R[d])
            Ymin = np.zeros(R[d])

            Ymin[:] = np.inf
            Ymax[:] = -np.inf

            for n in range(N):

                xnd = X[n, d]

                Zn = Z[:, n]

                if S[n, d] == 1:

                    # np.dot(Zn, Bcat[d].T, out=muyCat
                    muyCat = np.dot(Zn, Bcat[d].T)
                    if xnd == -1 or np.isnan(xnd):
                        for r in range(R[d]):
                            Ycat[d, r, n] = muyCat[r] + rand_gen.normal(loc=0, scale=sYd)
                    else:

                        xnd = int(X[n, d])
                        xr = int(xnd) - 1
                        xrr = int(xnd) - 2
                        Ycat[d, xr, n] = truncnormrnd(
                            muyCat[xr], sYd, 0, np.inf, rand_gen=rand_gen)
                        #
                        # QUESTION: is this for errorchecking/debugging?
                        if (np.isinf(Ycat[d, xr, n])):
                            print("n={}, d={},xnd={}, muy={}, y={}".format(n, d, xr,
                                                                           muyCat[xr], Ycat[d, xr, n]))
                        for r in range(R[d]):
                            if r != xr:
                                Ycat[d, r, n] = truncnormrnd(
                                    muyCat[r], sYd, -np.inf, Ycat[d, xr, n], rand_gen=rand_gen)

                    muy = np.dot(Zn, Bord[d])
                    Yord[d, n] = muy + rand_gen.normal(loc=0, scale=sYd)

                    muy = np.dot(Zn, Bcount[d])
                    Ycount[d, n] = muy + rand_gen.normal(loc=0, scale=sYd)

                elif S[n, d] == 2:

                    muy = np.dot(Zn, Bord[d])

                    if xnd == -1 or np.isnan(xnd):
                        Yord[d, n] = muy + rand_gen.normal(loc=0, scale=sYd)

                    elif xnd == 1:
                        xnd = int(X[n, d])
                        xr = int(xnd) - 1
                        xrr = int(xnd) - 2
                        Yord[d, n] = truncnormrnd(
                            muy, sYd, - np.inf, theta[d, xr], rand_gen=rand_gen)

                    elif xnd == R[d]:
                        xnd = int(X[n, d])
                        xr = int(xnd) - 1
                        xrr = int(xnd) - 2
                        Yord[d, n] = truncnormrnd(
                            muy, sYd, theta[d, xrr], np.inf, rand_gen=rand_gen)

                    else:
                        xnd = int(X[n, d])
                        xr = int(xnd) - 1
                        xrr = int(xnd) - 2
                        Yord[d, n] = truncnormrnd(
                            muy, sYd, theta[d, xrr], theta[d, xr], rand_gen=rand_gen)

                        # print(n, d, xr, X[n, d], Yord.shape, Ymax.shape)
                        if Yord[d, n] > Ymax[xr]:
                            Ymax[xr] = Yord[d, n]

                        if Yord[d, n] < Ymin[xrr]:
                            Ymin[xrr] = Yord[d, n]

                    muyCat = np.dot(Zn, Bcat[d].T)
                    for r in range(R[d]):
                        Ycat[d, r, n] = muyCat[r] + rand_gen.normal(loc=0, scale=sYd)

                    muy = np.dot(Zn, Bcount[d])
                    Ycount[d, n] = muy + rand_gen.normal(loc=0, scale=sYd)

                #
                # count
                elif S[n, d] == 3:

                    muy = np.dot(Zn, Bcount[d])
                    if xnd == -1 or np.isnan(xnd):
                        Ycount[d, n] = muy + rand_gen.normal(loc=0, scale=sYd)
                    else:

                        Ycount[d, n] = truncnormrnd(muy, sYd, f_1(
                            xnd, 2 / maxX[d]), f_1(xnd + 1, 2 / maxX[d]), rand_gen=rand_gen)

                    # print('shapes', Zn.shape, Bcat[d].T.shape, Bcat[d].shape)
                    muyCat = np.dot(Zn, Bcat[d].T)
                    for r in range(R[d]):
                        Ycat[d, r, n] = muyCat[r] + rand_gen.normal(loc=0, scale=sYd)

                    muy = np.dot(Zn, Bord[d])

                    Yord[d, n] = muy + rand_gen.normal(loc=0, scale=sYd)

            #
            # resetting, just in case
            muy = None
            muyCat = None

            #
            # sampling thetas
            for r in range(1, R[d] - 1):

                xlo, xhi = None, None

                if theta[d, r - 1] > Ymax[r]:
                    xlo = theta[d, r - 1]
                else:
                    xlo = Ymax[r]

                if theta[d, r + 1] < Ymin[r]:
                    xhi = theta[d, r + 1]
                else:
                    xhi = Ymin[r]

                if r == R[d] - 2:
                    theta[d, r] = truncnormrnd(0, s2theta, xlo, np.inf, rand_gen=rand_gen)
                else:
                    theta[d, r] = truncnormrnd(0, s2theta, xlo, xhi, rand_gen=rand_gen)

            Ymax = None
            Ymin = None

    return Yreal, Yint, Ypos, Ybin, Ycat, Yord, Ycount


@numba.njit()
def est_mean_Ys(X, X_orig, Z, C, R, maxR,
                # s2Y, s2B, s2U, SB, sY, sYd, s2theta, aux,
                Breal,  # Yreal,
                Bint,  # Yint,
                Bpos,  # Ypos,
                Bbin,  # Ybin,
                Bcat,  # Ycat,
                Bord,  # Yord,
                Bcount,  # Ycount,
                # maxX, meanX, minX,
                # theta, theta_L, theta_H,
                # Wint,                rand_gen
                ):
    """
    """

    N, D = X.shape

    Yreal_m = np.zeros((D, N))
    Ypos_m = np.zeros((D, N))
    Yint_m = np.zeros((D, N))
    Ybin_m = np.zeros((D, N))
    Ycat_m = np.zeros((D, maxR, N))
    Yord_m = np.zeros((D, N))
    Ycount_m = np.zeros((D, N))

    Yreal_m.fill(np.nan)
    Ypos_m.fill(np.nan)
    Yint_m.fill(np.nan)
    Ybin_m.fill(np.nan)
    Ycat_m.fill(np.nan)
    Yord_m.fill(np.nan)
    Ycount_m.fill(np.nan)

    for d in range(D):

        for n in range(N):

            xnd = X[n, d]
            tnd = X_orig[n, d]
            Zn = Z[:, n]

            if C[d] == 1:

                # if np.isnan(xnd) and not np.isnan(tnd):
                muy = np.dot(Zn, Breal[d])
                Yreal_m[d, n] = muy

                muy = np.dot(Zn, Bint[d])
                Yint_m[d, n] = muy

                muy = np.dot(Zn, Bpos[d])
                Ypos_m[d, n] = muy

            #
            # just like before, C[d] == 1 but with no pos
            elif C[d] == 2:

                # if np.isnan(xnd) and not np.isnan(tnd):
                muy = np.dot(Zn, Breal[d])
                Yreal_m[d, n] = muy

                muy = np.dot(Zn, Bint[d])
                Yint_m[d, n] = muy

            elif C[d] == 3:

                # if np.isnan(xnd) and not np.isnan(tnd):
                muy = np.dot(Zn, Bbin[d])
                Ybin_m[d, n] = muy

            elif C[d] == 4:

                muyCat = np.zeros(R[d])

                muyCat = np.dot(Zn, Bcat[d].T)

                for r in range(R[d]):
                    Ycat_m[d, r, n] = muyCat[r]

                muy = np.dot(Zn, Bord[d])
                Yord_m[d, n] = muy

                muy = np.dot(Zn, Bcount[d])
                Ycount_m[d, n] = muy

    return Yreal_m, Yint_m, Ypos_m, Ybin_m, Ycat_m, Yord_m, Ycount_m


# @numba.jit(float64[:, :](
#     # X            # X_orig
#     float64[:, :], float64[:, :],
#     # Z
#     float64[:, :],
#     # W
#     float64[:, :],
#     # alpha_W
#     float64[:, :],
#     # Kest,
#     int64,
#     #   C           R
#     uint8[:], int64[:],
#     # S
#     int32[:, :],
#     # s2Y, sYd, s2U,
#     float64, float64, float64,
#     # U
#     float64[:, :, :],
#     # Bs
#     optional(float64[:, :]),
#     optional(float64[:, :]),
#     optional(float64[:, :]),
#     optional(float64[:, :]), optional(float64[:, :, :]),
#     optional(float64[:, :]), optional(float64[:, :]),
#     # maxX, meanX, minX,
#     float64[:], float64[:], float64[:],
#     # theta, theta_L, theta_H,
#     float64[:, :], float64[:], float64[:],
#     # Wint n_samples_cat
#     int64, int64
#     # boolean, boolean,
# ), nopython=True)
def compute_miss_log_likelihood_numba(X,
                                      X_orig,
                                      Z, W, alpha_W, Kest,
                                      C, R, S,
                                      s2Y, sYd, s2U,
                                      U,
                                      Breal,
                                      Bint,
                                      Bpos,
                                      Bbin,
                                      Bcat,
                                      Bord,
                                      Bcount,
                                      maxX, meanX, minX,
                                      theta, theta_L, theta_H,
                                      Wint,
                                      n_samples_cat,
                                      # rand_gen,
                                      ):
    N, D = X.shape
    #
    # computing likelihoods
    L = np.zeros((D, N))

    for n in range(N):
        for d in range(D):

            L[d, n] = 0
            #
            # missing value?
            xnd = X[n, d]
            # if xnd == MISS_VALUE:
            if np.isnan(xnd):

                tnd = X_orig[n, d]

                # if tnd != MISS_VALUE:
                if not np.isnan(tnd):
                    #     print('tnd', tnd)

                    L[d, n] = 0
                    Baux = np.zeros(Kest)
                    Zn = Z[:, n]

                    if C[d] == 1 or C[d] == 2:

                        if S[n, d] == 1:
                            aux = np.dot(Zn, Breal[d])
                            L[d, n] = xpdf_re(tnd, 2 / (maxX[d] - meanX[d]),
                                              meanX[d], aux, s2Y, s2U)
                            # if np.isnan(L[d, n]):
                            #     print(d, n, L[d, n])
                            #     0 / 0

                        elif S[n, d] == 2:
                            aux = np.dot(Zn, Bint[d])
                            L[d, n] = xpdf_int(tnd, Wint, theta_L[d], theta_H[d], aux, s2Y, s2U)
                            # if np.isnan(L[d, n]):
                            #     print(d, n, tnd, Wint, theta_L[d], theta_H[d], aux, s2Y, s2U)
                            #     0 / 0

                        #
                        #
                        # FIXME: should this be dir? shouldn't it be pos?
                        elif S[n, d] == 3:
                            pass
                        elif S[n, d] == 4:
                            # Bd_view = gsl_matrix_submatrix (Bdir[d],0,0, Kest,1);
                            # matrix_multiply(&Zn_view.matrix,&Bd_view.matrix,aux,1,0,CblasTrans,CblasNoTrans);
                            #  LIK[d][n]= xpdf_dir(fre_1(tnd,2/maxX[d],0),theta_dir[d],gsl_matrix_get (aux, 0, 0),s2Y,s2u);
                            aux = np.dot(Zn, Bpos[d])
                            # print('l', tnd, 2 / maxX[d], aux, s2Y, s2U, )
                            # print('f1', f_1(tnd, 2 / maxX[d]))
                            # print('df1', df_1(tnd, 2 / maxX[d]))
                            L[d, n] = xpdf_pos(tnd, 2 / maxX[d], aux, s2Y, s2U)
                            # if np.isnan(L[d, n]):
                            #     print(d, n, L[d, n])
                            #     0 / 0

                        aux = None
                        Baux = None

                    elif C[d] == 4:
                        if S[n, d] == 1:
                            r = int(X_orig[n, d]) - 1

                            # prodC = np.ones(100)
                            # u = rand_gen.normal(loc=0, scale=sYd, size=100)
                            u = U[n, d, :n_samples_cat]

                            # for r2 in range(R[d]):
                            #     if r2 != r:
                            #         Baux = np.copy(Bcat[d, r])
                            #         Baux -= Bcat[d, r2]
                            #         aux = np.dot(Zn, Baux)

                            #         for ii in range(100):
                            #             # prodC[ii] = prodC[ii] * \
                            #             #     stats.norm.cdf(x=u[ii] + aux, scale=1)
                            #             prodC[ii] = prodC[ii] * \
                            #                 phi(x=u[ii] + aux)

                            # sumC = prodC.sum()
                            # L[d, n] = np.log(sumC / 100)
                            L[d, n] = cat_pdf_numba(d, r, Bcat, Zn, u, 100, R[d])
                            # if np.isnan(L[d, n]):
                            #     print(d, n, L[d, n])
                            #     0 / 0

                        elif S[n, d] == 2:
                            aux = np.dot(Zn, Bord[d])

                            if tnd == 1:
                                # L[d, n] = np.log(stats.norm.cdf(
                                #     x=theta[d, int(tnd) - 1] - aux, scale=1))
                                L[d, n] = np.log(phi(
                                    x=theta[d, int(tnd) - 1] - aux))
                                # if np.isnan(L[d, n]):
                                #     print(d, n, L[d, n])
                                #     0 / 0
                            elif tnd == R[d]:
                                # L[d, n] = np.log(
                                #     1 - stats.norm.cdf(x=theta[d, int(tnd) - 2] - aux, scale=1))
                                L[d, n] = np.log(
                                    1 - phi(x=theta[d, int(tnd) - 2] - aux))
                                # if not np.isfinite(L[d, n]):
                                #     print(d, n, L[d, n])
                                #     0 / 0
                            else:
                                # L[d, n] = np.log(stats.norm.cdf(
                                #     theta[d, int(tnd) - 1] - aux, scale=1) - stats.norm.cdf(x=theta[d, int(tnd) - 2] - aux, scale=1))
                                L[d, n] = np.log(phi(
                                    theta[d, int(tnd) - 1] - aux) - phi(x=theta[d, int(tnd) - 2] - aux))
                                # if np.isnan(L[d, n]):
                                #     print(d, n, L[d, n])
                                #     0 / 0

                        elif S[n, d] == 3:
                            aux = np.dot(Zn, Bcount[d])
                            # L[d, n] = np.log(stats.norm.cdf(
                            #     x=f_1(tnd + 1, 2 / maxX[d]) - aux, scale=1) - stats.norm.cdf(x=f_1(tnd, 2 / maxX[d]) - aux, scale=1))
                            L[d, n] = np.log(phi(
                                x=f_1(tnd + 1, 2 / maxX[d]) - aux) - phi(x=f_1(tnd, 2 / maxX[d]) - aux))
                            # if np.isnan(L[d, n]):
                            #     print(d, n, L[d, n])
                            #     0 / 0

                        aux = None
                        Baux = None

                    #
                    # deal with numerical inaccuracies:
                    if np.isinf(L[d, n]):
                        L[d, n] = LOG_ZERO

    return L


# @numba.jit(nopython=True)
# @numba.jit
@numba.jit(Tuple((float64[:, :, :], float64[:, :, :]))(
    # X             # instance_ids feature_ids
    float64[:, :], int64[:], int64[:],
    # Z
    float64[:, :],
    # W
    float64[:, :],
    # alpha_W
    float64[:, :],
    # Kest,
    int64,
    #   C           R
    uint8[:], int64[:],
    # S
    int32[:, :],
    # s2Y, sYd, s2U,
    float64, float64, float64,
    # U
    float64[:, :, :],
    # Bs
    optional(float64[:, :]),
    optional(float64[:, :]),
    optional(float64[:, :]),
    optional(float64[:, :]), optional(float64[:, :, :]),
    optional(float64[:, :]), optional(float64[:, :]),
    # maxX, meanX, minX,
    float64[:], float64[:], float64[:],
    # theta, theta_L, theta_H,
    float64[:, :], float64[:], float64[:],
    # Wint n_samples_cat
    int64, int64,
    # boolean, boolean,
), nopython=True)
def compute_log_likelihood_numba_u(X,
                                   instance_ids, feature_ids,
                                   Z,
                                   W,
                                   alpha_W, Kest,
                                   C, R, S,
                                   s2Y, sYd, s2U,
                                   # u,
                                   U,
                                   Breal,
                                   Bint,
                                   Bpos,
                                   Bbin,
                                   Bcat,
                                   Bord,
                                   Bcount,
                                   maxX, meanX, minX,
                                   theta, theta_L, theta_H,
                                   Wint,
                                   n_samples_cat,
                                   # n_samples_cat=100,
                                   # rand_gen,
                                   # compute_hard_ll=True,
                                   # compute_entry_wise=False,
                                   ):
    """
    Computing likelihoods
    """

    # if rand_gen is None:
    #     rand_gen = np.random.RandomState(RANDOM_SEED)

    N, D = X.shape
    n_instances = None

    #
    # adding one dimension for likelihood types
    # (0:real, 1:int, 2:pos, 3:cat, 4:ord, 5:count)
    # FIXME: just semplify everything having L = 3 and refactor all the code
    L = 6
    #
    # storing log-likelihoods in a matrix DxNxL
    # TODO: make this more memory efficient
    LL = np.zeros((D, N, L))

    for n, d in zip(instance_ids, feature_ids):

        h = None
        LL[d, n, :] = 0

        xnd = X[n, d]

        if ~np.isnan(xnd):
            Baux = np.zeros(Kest)
            Zn = Z[:, n]

            if C[d] == 1:

                #
                # real
                aux = np.dot(Zn, Breal[d])
                LL[d, n, 0] = xpdf_re(xnd, 2 / (maxX[d] - meanX[d]),
                                      meanX[d], aux, s2Y, s2U)

                #
                # int
                aux = np.dot(Zn, Bint[d])
                LL[d, n, 1] = xpdf_int(xnd, Wint, theta_L[d], theta_H[d], aux, s2Y, s2U)

                #
                # pos
                aux = np.dot(Zn, Bpos[d])
                LL[d, n, 2] = xpdf_pos(xnd, 2 / maxX[d], aux, s2Y, s2U)

                if S[n, d] == 1:
                    h = 0
                elif S[n, d] == 2:
                    h = 1
                elif S[n, d] == 3:
                    raise NotImplementedError('Circuluar data are not supposed to be used')

                elif S[n, d] == 4:
                    h = 2

            elif C[d] == 2:

                #
                # real
                aux = np.dot(Zn, Breal[d])
                LL[d, n, 0] = xpdf_re(xnd, 2 / (maxX[d] - meanX[d]),
                                      meanX[d], aux, s2Y, s2U)

                #
                # int
                aux = np.dot(Zn, Bint[d])
                LL[d, n, 1] = xpdf_int(xnd, Wint, theta_L[d], theta_H[d], aux, s2Y, s2U)

                if S[n, d] == 1:
                    h = 0
                elif S[n, d] == 2:
                    h = 1
                elif S[n, d] == 3:
                    raise NotImplementedError('Circuluar data are not supposed to be used')

                elif S[n, d] == 4:
                    raise NotImplementedError('This feature is expected not to be positive')

            elif C[d] == 3:
                continue

            elif C[d] == 4:

                xnd = int(xnd)
                r = int(xnd - 1)
                rr = int(xnd - 2)

                # u_1 = rand_gen.choice(np.arange(u.shape[0]), size=100, replace=False)
                # u = rand_gen.normal(loc=0, scale=sYd, size=100)
                u = U[n, d, :n_samples_cat]

                #
                # cat
                # prodC = np.ones(n_samples_cat)
                # # u = rand_gen.normal(loc=0, scale=sYd, size=n_samples_cat)
                # # u = np.array([np.random.normal(loc=0, scale=sYd)
                # #               for j in range(n_samples_cat)])

                # for r2 in range(R[d]):
                #     r2 = int(r2)
                #     r = int(r)
                #     if r2 != r:
                #         Baux = np.copy(Bcat[d, r])
                #         Baux -= Bcat[d, r2]
                #         aux = np.dot(Zn, Baux)

                #         for ii in range(n_samples_cat):
                #             # prodC[ii] = prodC[ii] * \
                #             #     stats.norm.cdf(x=u[ii] + aux, scale=1)
                #             prodC[ii] = prodC[ii] * \
                #                 phi(x=u[ii] + aux)

                # sumC = prodC.sum()
                # LL[d, n, 3] = np.log(sumC / 100)
                ll = cat_pdf_numba(d, r, Bcat, Zn, u, n_samples_cat, R[d])
                LL[d, n, 3] = ll
                if (np.isinf(ll)):
                    print('C4 cat', d, n, ll)

                #
                # ord
                aux = np.dot(Zn, Bord[d])

                if xnd == 1:
                    LL[d, n, 4] = np.log(phi(x=theta[d, r] - aux))

                elif xnd == R[d]:
                    LL[d, n, 4] = np.log(1 - phi(x=theta[d, rr] - aux))

                else:
                    LL[d, n, 4] = np.log(phi(theta[d, r] - aux) - phi(x=theta[d, rr] - aux))

                #
                # count
                aux = np.dot(Zn, Bcount[d])
                LL[d, n, 5] = np.log(phi(x=f_1(xnd + 1, 2 / maxX[d]) - aux) -
                                     phi(x=f_1(xnd, 2 / maxX[d]) - aux))

                if S[n, d] == 1:
                    h = 3

                elif S[n, d] == 2:
                    h = 4

                elif S[n, d] == 3:
                    h = 5

    #
    # reshaping weight matrix
    WM = np.zeros((D, N, L))

    for d in range(D):
        if C[d] == 1:
            WM[d, :, 0] = W[d, 0]
            WM[d, :, 1] = W[d, 1]
            WM[d, :, 2] = W[d, 3]
        elif C[d] == 2:
            WM[d, :, 0] = W[d, 0]
            WM[d, :, 1] = W[d, 1]
        elif C[d] == 3:
            WM[d, :, :] = 1  # log(\sum{exp(0)*WM}) = 1
        elif C[d] == 4:
            WM[d, :, 3] = W[d, 0]
            WM[d, :, 4] = W[d, 1]
            WM[d, :, 5] = W[d, 2]

    return LL, WM

    avg_ll = None


def compute_log_likelihood_numba(X,
                                 instance_ids, feature_ids,
                                 Z, W, alpha_W, Kest,
                                 C, R, S,
                                 s2Y, sYd, s2U,
                                 # u,
                                 Breal,
                                 Bint,
                                 Bpos,
                                 Bbin,
                                 Bcat,
                                 Bord,
                                 Bcount,
                                 maxX, meanX, minX,
                                 theta, theta_L, theta_H,
                                 Wint,
                                 n_samples_cat,
                                 # n_samples_cat=100,
                                 rand_gen,
                                 compute_hard_ll=True,
                                 compute_entry_wise=False,
                                 ):
    """
    Computing likelihoods
    """

    N, D = X.shape
    n_instances = None

    #
    # adding one dimension for likelihood types
    # (0:real, 1:int, 2:pos, 3:cat, 4:ord, 5:count)
    # FIXME: just semplify everything having L = 3 and refactor all the code
    L = 6
    #
    # storing log-likelihoods in a matrix DxNxL
    # TODO: make this more memory efficient
    LL = np.zeros((D, N, L))

    HLL = None
    if compute_hard_ll:
        HLL = np.zeros((D, N))

    for n, d in zip(instance_ids, feature_ids):

        h = None
        LL[d, n, :] = 0

        xnd = X[n, d]

        if ~np.isnan(xnd):
            Baux = np.zeros(Kest)
            Zn = Z[:, n]

            if C[d] == 1:

                #
                # real
                aux = np.dot(Zn, Breal[d])
                LL[d, n, 0] = xpdf_re(xnd, 2 / (maxX[d] - meanX[d]),
                                      meanX[d], aux, s2Y, s2U)

                #
                # int
                aux = np.dot(Zn, Bint[d])
                LL[d, n, 1] = xpdf_int(xnd, Wint, theta_L[d], theta_H[d], aux, s2Y, s2U)

                #
                # pos
                aux = np.dot(Zn, Bpos[d])
                LL[d, n, 2] = xpdf_pos(xnd, 2 / maxX[d], aux, s2Y, s2U)

                if S[n, d] == 1:
                    h = 0
                elif S[n, d] == 2:
                    h = 1
                elif S[n, d] == 3:
                    raise NotImplementedError('Circuluar data are not supposed to be used')

                elif S[n, d] == 4:
                    h = 2

            elif C[d] == 2:

                #
                # real
                aux = np.dot(Zn, Breal[d])
                LL[d, n, 0] = xpdf_re(xnd, 2 / (maxX[d] - meanX[d]),
                                      meanX[d], aux, s2Y, s2U)

                #
                # int
                aux = np.dot(Zn, Bint[d])
                LL[d, n, 1] = xpdf_int(xnd, Wint, theta_L[d], theta_H[d], aux, s2Y, s2U)

                if S[n, d] == 1:
                    h = 0
                elif S[n, d] == 2:
                    h = 1
                elif S[n, d] == 3:
                    raise NotImplementedError('Circuluar data are not supposed to be used')

                elif S[n, d] == 4:
                    raise NotImplementedError('This feature is expected not to be positive')

            elif C[d] == 3:
                continue

            elif C[d] == 4:

                xnd = int(xnd)
                r = xnd - 1
                rr = xnd - 2

                u = rand_gen.normal(loc=0, scale=sYd, size=100)

                ll = cat_pdf_numba(d, r, Bcat, Zn, u, n_samples_cat, R[d])
                LL[d, n, 3] = ll
                if (np.isinf(ll)):
                    print('C4 cat', d, n, ll)

                #
                # ord
                aux = np.dot(Zn, Bord[d])

                if xnd == 1:
                    LL[d, n, 4] = np.log(phi(x=theta[d, r] - aux))

                elif xnd == R[d]:
                    LL[d, n, 4] = np.log(1 - phi(x=theta[d, rr] - aux))

                else:
                    LL[d, n, 4] = np.log(phi(theta[d, r] - aux) - phi(x=theta[d, rr] - aux))

                #
                # count
                aux = np.dot(Zn, Bcount[d])
                LL[d, n, 5] = np.log(phi(x=f_1(xnd + 1, 2 / maxX[d]) - aux) -
                                     phi(x=f_1(xnd, 2 / maxX[d]) - aux))

                if S[n, d] == 1:
                    h = 3

                elif S[n, d] == 2:
                    h = 4

                elif S[n, d] == 3:
                    h = 5

            if compute_hard_ll:
                HLL[d, n] = LL[d, n, h]

                aux = None
                Baux = None

    #
    # reshaping weight matrix
    WM = np.zeros((D, N, L))

    for d in range(D):
        if C[d] == 1:
            WM[d, :, 0] = W[d, 0]
            WM[d, :, 1] = W[d, 1]
            WM[d, :, 2] = W[d, 3]
        elif C[d] == 2:
            WM[d, :, 0] = W[d, 0]
            WM[d, :, 1] = W[d, 1]
        elif C[d] == 3:
            WM[d, :, :] = 1  # log(\sum{exp(0)*WM}) = 1
        elif C[d] == 4:
            WM[d, :, 3] = W[d, 0]
            WM[d, :, 4] = W[d, 1]
            WM[d, :, 5] = W[d, 2]

    avg_ll = None
    #
    # making a mixture of valid likelihood models

    # MLL = logsumexp_numba(LL, b=WM)
    MLL = logsumexp(LL, axis=-1, b=WM)
    if np.any(np.isinf(MLL)):
        print('MLL with infs!!!!!\n',
              # MLL[np.isinf(
              # MLL)], LL[np.isinf(MLL)],
              np.count_nonzero(WM, axis=-1), np.isinf(MLL).sum())
        # 0 / 0

    if compute_entry_wise:
        if compute_hard_ll:
            return MLL, HLL

        else:
            return MLL

    LLs = MLL.sum(axis=0)
    assert LLs.shape[0] == N

    if compute_hard_ll:
        HLLs = HLL.sum(axis=0)
        assert HLLs.shape[0] == N

        return LLs, HLLs

    else:
        return LLs


# @numba.jit(nopython=False)
def compute_log_likelihood(X,
                           instance_ids, feature_ids,
                           Z, W, alpha_W, Kest,
                           C, R, S,
                           s2Y, sYd, s2U,
                           Breal,
                           Bint,
                           Bpos,
                           Bbin,
                           Bcat,
                           Bord,
                           Bcount,
                           maxX, meanX, minX,
                           theta, theta_L, theta_H,
                           Wint,
                           n_samples_cat=100,
                           compute_hard_ll=True,
                           rand_gen=None):
    """
    Computing likelihoods
    """

    if rand_gen is None:
        rand_gen = np.random.RandomState(RANDOM_SEED)

    N, D = X.shape
    n_instances = None

    #
    # if no feature_ids are not specified, we take all of them, assuming instance ids is
    # a sequence of unique ids
    if feature_ids is None:
        # feature_ids = np.array([i for i in range(D)], dtype=int)
        _instance_ids = []
        feature_ids = []
        n_instances = len(instance_ids)
        for n in instance_ids:
            for d in range(D):
                _instance_ids.append(n)
                feature_ids.append(d)
        instance_ids = _instance_ids

    else:
        assert len(feature_ids) == len(instance_ids), \
            'Different ids for instances {} and features {}'.format(instance_ids, feature_ids)
        #
        # FIXME: this is not always correct, here we assume we are taking into account still all features in D
        n_instances = len(feature_ids) // D
    # print('Considering # instances', n_instances, '\t\t\t\t\t')

    #
    # adding one dimension for likelihood types
    # (0:real, 1:int, 2:pos, 3:cat, 4:ord, 5:count)
    # FIXME: just semplify everything having L = 3 and refactor all the code
    L = 6
    #
    # storing log-likelihoods in a matrix DxNxL
    # TODO: make this more memory efficient
    LL = np.zeros((D, N, L))

    HLL = None
    if compute_hard_ll:
        HLL = np.zeros((D, N))

    for n, d in zip(instance_ids, feature_ids):

        h = None
        LL[d, n, :] = 0

        xnd = X[n, d]

        Baux = np.zeros(Kest)
        Zn = Z[:, n]

        if C[d] == 1:

            #
            # real
            aux = np.dot(Zn, Breal[d])
            LL[d, n, 0] = xpdf_re(xnd, 2 / (maxX[d] - meanX[d]),
                                  meanX[d], aux, s2Y, s2U)

            #
            # int
            aux = np.dot(Zn, Bint[d])
            LL[d, n, 1] = xpdf_int(xnd, Wint, theta_L[d], theta_H[d], aux, s2Y, s2U)

            #
            # pos
            aux = np.dot(Zn, Bpos[d])
            LL[d, n, 2] = xpdf_pos(xnd, 2 / maxX[d], aux, s2Y, s2U)

            if S[n, d] == 1:
                h = 0
            elif S[n, d] == 2:
                h = 1
            elif S[n, d] == 3:
                raise NotImplementedError('Circuluar data are not supposed to be used')

            elif S[n, d] == 4:
                h = 2

        elif C[d] == 2:

            #
            # real
            aux = np.dot(Zn, Breal[d])
            LL[d, n, 0] = xpdf_re(xnd, 2 / (maxX[d] - meanX[d]),
                                  meanX[d], aux, s2Y, s2U)

            #
            # int
            aux = np.dot(Zn, Bint[d])
            LL[d, n, 1] = xpdf_int(xnd, Wint, theta_L[d], theta_H[d], aux, s2Y, s2U)

            if S[n, d] == 1:
                h = 0
            elif S[n, d] == 2:
                h = 1
            elif S[n, d] == 3:
                raise NotImplementedError('Circuluar data are not supposed to be used')

            elif S[n, d] == 4:
                raise NotImplementedError('This feature is expected not to be positive')

        elif C[d] == 3:
            continue

        elif C[d] == 4:

            xnd = int(xnd)
            r = xnd - 1
            rr = xnd - 2

            #
            # cat
            prodC = np.ones(n_samples_cat)
            u = rand_gen.normal(loc=0, scale=sYd, size=n_samples_cat)

            for r2 in range(R[d]):
                if r2 != r:
                    Baux = np.copy(Bcat[d, r])
                    Baux -= Bcat[d, r2]
                    aux = np.dot(Zn, Baux)

                    for ii in range(n_samples_cat):
                        # prodC[ii] = prodC[ii] * \
                        #     stats.norm.cdf(x=u[ii] + aux, scale=1)
                        prodC[ii] = prodC[ii] * \
                            phi(x=u[ii] + aux)

            sumC = prodC.sum()
            LL[d, n, 3] = np.log(sumC / n_samples_cat)

            #
            # ord
            aux = np.dot(Zn, Bord[d])

            if xnd == 1:
                LL[d, n, 4] = np.log(phi(x=theta[d, r] - aux))

            elif xnd == R[d]:
                LL[d, n, 4] = np.log(1 - phi(x=theta[d, rr] - aux))

            else:
                LL[d, n, 4] = np.log(phi(theta[d, r] - aux) - phi(x=theta[d, rr] - aux))

            #
            # count
            aux = np.dot(Zn, Bcount[d])
            LL[d, n, 5] = np.log(phi(x=f_1(xnd + 1, 2 / maxX[d]) - aux) -
                                 phi(x=f_1(xnd, 2 / maxX[d]) - aux))

            if S[n, d] == 1:
                h = 3

            elif S[n, d] == 2:
                h = 4

            elif S[n, d] == 3:
                h = 5

        if compute_hard_ll:
            # print(LL[d, n, h].shape, h, n, d, LL[d, n, h])
            HLL[d, n] = LL[d, n, h]
            if np.isnan(LL[d, n, h]) or np.isinf(LL[d, n, h]):
                print('nan', d, n, h, xnd, maxX[d], C[d], '  --  ')

            aux = None
            Baux = None

    #
    # reshaping weight matrix
    WM = np.zeros((D, N, L))

    for d in range(D):
        if C[d] == 1:
            WM[d, :, 0] = W[d, 0]
            WM[d, :, 1] = W[d, 1]
            WM[d, :, 2] = W[d, 3]
        elif C[d] == 2:
            WM[d, :, 0] = W[d, 0]
            WM[d, :, 1] = W[d, 1]
        elif C[d] == 3:
            WM[d, :, :] = 1  # log(\sum{exp(0)*WM}) = 1
        elif C[d] == 4:
            WM[d, :, 3] = W[d, 0]
            WM[d, :, 4] = W[d, 1]
            WM[d, :, 5] = W[d, 2]

    avg_ll = None
    #
    # making a mixture of valid likelihood models

    MLL = logsumexp(LL, axis=-1, b=WM)
    assert MLL.shape == HLL.shape, "Log sum exp went wrong, {} {} {}".format(
        MLL.shape, LL.shape, HLL.shape)

    avg_ll = MLL.sum() / n_instances

    if compute_hard_ll:
        avg_hll = HLL.sum() / n_instances
        return LL, avg_ll, HLL, avg_hll

    else:
        return LL, avg_ll


def infer_data_types(X, C, R,
                     s2Z=1, s2B=1, s2Y=1, s2U=0.001, s2theta=1,
                     n_iters=10000,
                     burn_in=4000,
                     max_K=None,
                     X_orig=None,
                     save_ll_history=100,
                     save_perf_history=1,
                     disc_perf_metrics=['MSE', 'MSLE', 'MAE', 'ACC'],
                     cont_perf_metrics=['MSE', 'MAE'],
                     int_exp=INT_EXP,
                     # pos_cont_perf_metrics=['MSE', 'MSLE', 'MAE'],
                     save_all_params=False,
                     rand_gen=None):
    """
    This is a porting of Isabel's original code

    ---
    input:

    X : float matrix
        data matrix (size NxD) of (homogeneous) floating point numbers
        Missing data, if present, are encoded by MISS_VALUE (default -1)
    C : uint array
        meta-type array (size D), encoding
        {
          1: real (w positive: all real | positive | interval)
          2: real (w/o positive: all real | interval)
          3: binary data
          4: discrete (non-binary: categorical | ordinal | count)
        }
        TODO: refactor C to T

    R : uint array
        max-domain-cardinality array (size D) for discrete (meta-type 4) RVs

    s2Z : float
        variance for the Zs (default=1)

    s2B : float
        variance for the Bs (default=1)

    s2Y : float
        variance for the Ys (default=1)

    s2U : float
        variance for the Us (default=0.001)

    s2theta : float
        variance for the thetas (default=1)

    n_iters : uint
        number of Gibbs iterations (default=1000)

    max_K : uint
        number of maximum latent dimensions

    ll_history: bool
        whether to compute and store the model lls at each iterate

    X_orig : float matrix
        (optional) original data matrix (size NxD) with no missing values
    ---
    output:

    K_est: uint
        estimated best value for K

    W_est: float matrix
        estimated data types mixture weights (size Dx4)
         where 4 is the number of types per meta-types
        TODO: refactor this into 3

    countErr: float array
        count errors

    L : float matrix
        computed likelihood


    """

    X = preprocess_positive_real_data(X, C)
    # X_orig = preprocess_positive_real_data(X_orig, C)

    # init_W = init_weights(C)
    alpha_W = init_weights(C)

    N, D = X.shape

    instance_ids = []
    feature_ids = []
    for n in range(N):
        for d in range(D):
            instance_ids.append(n)
            feature_ids.append(d)

    if X_orig is not None:
        assert X_orig.shape[0] == N and X_orig.shape[1] == D, X_orig.shape

    # if max_K is None:
    #     max_K = np.sqrt(D)
    #     logging.info('Setting max K to square root of D {}'.format(D))

    if rand_gen is None:
        rand_gen = np.random.RandomState(RANDOM_SEED)

    ll_history = []
    hll_history = []
    ll_count_history = []
    iter_time_history = []
    ll_time_history = []
    perf_dict_history = []

    #
    # from variance to stds
    su = np.sqrt(s2U)
    sY = np.sqrt(s2Y)
    sYd = float(sY)
    s2uy = (s2Y + s2U)
    suy = np.sqrt(s2Y + s2U)
    Wint = 2

    #
    # maximum cardinality for support of discrete RVs
    maxR = np.max(R)

    Kest, W_est, countErr, L = max_K, alpha_W, None, None

    #
    # initing additional variables
    countErrC = None
    W = np.zeros((D, 4))

    #
    # computing max, min, count statistics
    #
    # maxX = np.zeros(D)
    # minX = np.zeros(D)
    # meanX = np.zeros(D)
    countX, sumX = 0, 0
    theta_L = np.zeros(D)
    theta_H = np.zeros(D)
    theta_dir = np.zeros(D)

    S = np.zeros((N, D), dtype=np.int32)

    maxX = np.nanmax(X, axis=0)
    minX = np.nanmin(X, axis=0)
    stdX = scipy.stats.sem(X, axis=0, nan_policy='omit')
    print('min', minX, 'max', maxX, 'sem', stdX)

    if np.any(np.isnan(X)):
        rangeX = (maxX - minX)
        maxX = maxX + rangeX * int_exp
        minX = minX - rangeX * int_exp
        for d in range(D):
            if C[d] == 4:
                minX[d] = np.ceil(minX[d])
                maxX[d] = np.ceil(maxX[d])

        minX = np.nanmin(X_orig, axis=0)
        maxX = np.nanmax(X_orig, axis=0)
        print('min after P', minX, 'max after P', maxX)
        print('min true', np.nanmin(X_orig, axis=0), 'max orig', np.nanmax(X_orig, axis=0))

    # maxX = np.max(X_orig, axis=0)
    # minX = np.min(X_orig, axis=0)

    countX = ((X != -1) & (~ np.isnan(X))).sum(axis=0)
    sumX = np.nansum(X, axis=0)
    meanX = sumX / countX

    epsilon = (maxX - minX) / 10000
    theta_dir = fre_1(np.maximum(np.abs(minX), abs(maxX)) + epsilon, 2 / maxX, 0)

    theta_L = minX - epsilon
    theta_H = maxX + epsilon

    for d in range(D):
        #
        # sampling from the Dirichlet
        # print(alpha_W[d])
        # p = rand_gen.dirichlet(alpha_W[d], size=1)

        # W[d, :] = p

        if C[d] == 1:
            W[d, [0, 1, 3]] = rand_gen.dirichlet(alpha_W[d, [0, 1, 3]], size=1)
        elif C[d] == 2:
            W[d, [0, 1]] = rand_gen.dirichlet(alpha_W[d, [0, 1]], size=1)
        elif C[d] == 3:
            pass
        elif C[d] == 4:
            W[d, [0, 1, 2]] = rand_gen.dirichlet(alpha_W[d, [0, 1, 2]], size=1)

    #
    # allocating matrices only once
    # in Isabel's code it was allocating only some rows, depending if the meta-type
    # was allowing certain types
    #
    #
    Z = np.zeros((Kest, N))
    Yreal = np.zeros((D, N))
    Ypos = np.zeros((D, N))
    Yint = np.zeros((D, N))
    Ydir = np.zeros((D, N))
    Ybin = np.zeros((D, N))
    Ycat = np.zeros((D, maxR, N))
    Yord = np.zeros((D, N))
    Ycount = np.zeros((D, N))

    Breal = np.zeros((D, Kest))
    Bpos = np.zeros((D, Kest))
    Bint = np.zeros((D, Kest))
    # Bdir = np.zeros((D, Kest))
    Bbin = np.zeros((D, Kest))
    Bcat = np.zeros((D, maxR, Kest))
    Bord = np.zeros((D, Kest))
    Bcount = np.zeros((D, Kest))

    theta = np.zeros((D, maxR))
    EYE = np.eye(Kest, Kest)

    #
    # collecting samples(Z, B, Y, S, W)
    samples = []

    #
    # initing Bs and Ys
    for d in range(D):

        muB = np.zeros(Kest)
        SB = np.eye(Kest, Kest) * s2B

        #
        # all real data
        if C[d] == 1:
            Breal[d] = mvnrnd(muB, SB, rand_gen)
            Bpos[d] = mvnrnd(muB, SB, rand_gen)
            Bint[d] = mvnrnd(muB, SB, rand_gen)

            p = W[d, :]

            for n in range(N):
                snd = mnrnd(p, rand_gen) + 1
                S[n, d] = snd
                xnd = X[n, d]

                if xnd == -1 or np.isnan(xnd):
                    Yreal[d, n] = rand_gen.normal(loc=0, scale=sY)
                    Yint[d, n] = rand_gen.normal(loc=0, scale=sY)
                    Ypos[d, n] = rand_gen.normal(loc=0, scale=sY)
                else:
                    Yreal[d, n] = fre_1(xnd, 2 / (maxX[d] - meanX[d]), meanX[d])
                    Yint[d, n] = fint_1(xnd, Wint, theta_L[d], theta_H[d])
                    Ypos[d, n] = f_1(xnd, 2 / maxX[d])

        #
        # real but not positive
        elif C[d] == 2:

            Breal[d] = mvnrnd(muB, SB, rand_gen)
            Bint[d] = mvnrnd(muB, SB, rand_gen)

            p = W[d, :]

            for n in range(N):
                snd = mnrnd(p, rand_gen) + 1
                S[n, d] = snd
                xnd = X[n, d]

                if xnd == -1 or np.isnan(xnd):
                    Yreal[d, n] = rand_gen.normal(loc=0, scale=sY)
                    if np.isnan(Yreal[d, n]):
                        print(Yreal[d, n], xnd)
                        0 / 0
                    Yint[d, n] = rand_gen.normal(loc=0, scale=sY)
                else:
                    Yreal[d, n] = fre_1(xnd, 2 / (maxX[d] - meanX[d]), meanX[d])
                    if np.isnan(Yreal[d, n]):
                        print(Yreal[d, n], xnd)
                        0 / 0
                    Yint[d, n] = fint_1(xnd, Wint, theta_L[d], theta_H[d])

        elif C[d] == 3:

            Bbin[d] = mvnrnd(muB, SB, rand_gen)

            for n in range(N):

                if xnd == -1 or np.isnan(xnd):
                    Ybin[d, n] = rand_gen.normal(loc=0, scale=sY)
                elif xnd == 1:
                    xnd = int(X[n, d])
                    Ybin[d, n] = truncnormrnd(0, sY, -np.inf, 0, rand_gen=rand_gen)
                elif xnd == 2:
                    xnd = int(X[n, d])
                    Ybin[d, n] = truncnormrnd(0, sY, 0, np.inf, rand_gen=rand_gen)

        elif C[d] == 4:

            Bord[d] = mvnrnd(muB, SB, rand_gen)
            Bcount[d] = mvnrnd(muB, SB, rand_gen)
            Bcat[d, 0, :] = -1

            theta[d, 0] = -sY

            p = W[d, :]

            for r in range(1, R[d]):
                Bcat[d, r] = mvnrnd(muB, SB, rand_gen)
                if r < R[d] - 1:
                    theta[d, r] = theta[d, r - 1] + (4 * sY / maxX[d]) * rand_gen.rand()

            for n in range(N):
                snd = mnrnd(p, rand_gen) + 1
                S[n, d] = snd
                xnd = X[n, d]

                if xnd == -1 or np.isnan(xnd):

                    Yord[d, n] = rand_gen.normal(loc=0, scale=sYd)
                    Ycount[d, n] = rand_gen.normal(loc=0, scale=sYd)

                    for r in range(R[d]):
                        Ycat[d, r, n] = rand_gen.normal(loc=0, scale=sYd)

                else:

                    xnd = int(xnd)
                    xr = int(xnd) - 1
                    xrr = int(xnd) - 2

                    Ycat[d, xr, n] = truncnormrnd(0, sYd, 0, np.inf, rand_gen=rand_gen)
                    for r in range(R[d]):
                        if r != xr:
                            Ycat[d, r, n] = truncnormrnd(
                                0, sYd, -np.inf, Ycat[d, xr, n], rand_gen=rand_gen)

                    if xnd == 1:
                        Yord[d, n] = truncnormrnd(0, sYd, -np.inf, theta[d, xr], rand_gen=rand_gen)
                    elif xnd == R[d]:
                        Yord[d, n] = truncnormrnd(0, sYd, theta[d, xrr], np.inf, rand_gen=rand_gen)
                    else:
                        Yord[d, n] = truncnormrnd(0, sYd, theta[d, xrr],
                                                  theta[d, xr], rand_gen=rand_gen)

                    Ycount[d, n] = f_1(xnd, 2 / maxX[d]) + rand_gen.normal(loc=0, scale=sYd)

        muB = None
        SB = None

    #
    # allocating just once
    #
    # TODO: refactor that to allocate only the memory needed for meta-types
    #
    Preal = np.zeros((D, Kest, Kest))
    Ppos = np.zeros((D, Kest, Kest))
    Pint = np.zeros((D, Kest, Kest))
    Pbin = np.zeros((D, Kest, Kest))
    Pcat = np.zeros((D, maxR, Kest, Kest))
    Pord = np.zeros((D, Kest, Kest))
    Pcount = np.zeros((D, Kest, Kest))

    countErr, countErrC = 0, 0

    #
    # precomputing random normal values
    u_size = (N, D, U_MAX)
    U = rand_gen.normal(loc=0, scale=sYd, size=u_size)

    #
    # Gibbs sampler
    for it in range(n_iters):

        #
        # A sample may contain:
        #    - 'id': the iteration id
        #    - 'Zs': a copy of the lv Z matrix (optional, if save_all_params is True)
        #    - 'Bs': a copy of the lv Bs as in a tuple (Breal, Bint, Bpos, Bbin, Bcat, Bord, Bcount, optional)
        #    - 'Ys': a copy of the pseudo obs Ys as in a tuple (Yreal, Yint, Ypos, Ybin, Ycat, Yord, Ycount, optional)
        #    - 'theta' : a copy of theta (optional)
        #    - 'S': a copy of the S matrix
        #    - 'W': a copy of the global type weights
        #    - 'lls': a vector of lls for the training samples (marginal lls when missing values)
        #    - 'mv-lls': a vector of the lls for the missing values
        #    - 'mv-preds': the imputed missing values
        #    - 'mv-preds-scores': a dictionary of some pre-computed measure on mv-preds, feature-wise
        #    - 'time': time taken for one iteration
        sample = {}
        sample['id'] = it

        iter_start_t = perf_counter()

        if countErr > 0 or countErrC > 0:
            break

        #
        # precompute b*b^T
        step_start_t = perf_counter()
        for d in range(D):
            if C[d] == 1:
                #
                # Question: can we skip this by operating only on the blas 'beta'?
                Preal[d, :, :] = 0
                Ppos[d, :, :] = 0
                Pint[d, :, :] = 0

                np.outer(Breal[d], Breal[d], out=Preal[d])
                np.outer(Bpos[d], Bpos[d], out=Ppos[d])
                np.outer(Bint[d], Bint[d], out=Pint[d])

            elif C[d] == 2:
                Preal[d, :, :] = 0
                Pint[d, :, :] = 0

                np.outer(Breal[d], Breal[d], out=Preal[d])
                np.outer(Bint[d], Bint[d], out=Pint[d])

            elif C[d] == 3:
                Pbin[d, :, :] = 0

                np.outer(Bbin[d], Bbin[d], out=Pbin[d])

            elif C[d] == 4:
                Pord[d, :, :] = 0
                Pcount[d, :, :] = 0
                Pcat[d, :, :, :] = 0

                np.outer(Bord[d], Bord[d], out=Pord[d])
                np.outer(Bcount[d], Bcount[d], out=Pcount[d])

                for r in range(maxR):
                    if r < R[d]:
                        np.outer(Bcat[d, r], Bcat[d, r], out=Pcat[d, r])

        step_end_t = perf_counter()
        logging.info('\tprecomputed bb^t in {} secs'.format(step_end_t - step_start_t))

        #
        # sampling Zs
        #
        muZ = np.zeros(Kest)
        SZ = np.zeros((Kest, Kest))
        aux = np.zeros(Kest)

        step_start_t = perf_counter()
        auxM, SZ = sampling_Z_numba(X, Kest, C.astype(np.uint8), R.astype(np.int64),
                                    muZ,
                                    SZ, s2Z, s2Y,
                                    aux,
                                    Preal, Ppos, Pint, Pbin, Pcat, Pord, Pcount,
                                    Breal, Bpos, Bint, Bbin, Bcat, Bord, Bcount,
                                    Yreal, Ypos, Yint, Ybin, Ycat, Yord, Ycount,
                                    EYE,
                                    # rand_gen
                                    )
        for n in range(X.shape[0]):
            Z[:, n] = mvnrnd(auxM[n, :], SZ, rand_gen)

        if burn_in and it >= burn_in and save_all_params:
            sample['Zs'] = np.copy(Z)

        step_end_t = perf_counter()
        logging.info('\tsampled Zs in {} secs'.format(step_end_t - step_start_t))

        #
        # force cleaning
        SZ = None
        muZ = None
        aux = None

        step_start_ts = [None for d in range(D)]
        step_mid_ts = [None for d in range(D)]
        step_smid_ts = [None for d in range(D)]
        step_end_ts = [None for d in range(D)]

        #
        # sampling Ys
        step_start_t = perf_counter()
        Yreal, Yint, Ypos, Ybin, Ycat, Yord, Ycount = \
            sampling_Ys_numba(X, Z, d,
                              Kest,
                              C, R, S,
                              s2Y, s2B, s2U,
                              # SB,
                              sY, sYd, s2theta,
                              aux,
                              Breal, Bpos, Bint, Bbin, Bcat, Bord, Bcount,
                              Yreal, Ypos, Yint, Ybin, Ycat, Yord, Ycount,
                              maxX, meanX, minX,
                              theta, theta_L, theta_H,
                              Wint,
                              rand_gen)
        step_end_t = perf_counter()
        logging.info('\tsampled Ys in {} secs'.format(step_end_t - step_start_t))

        time_b_ini = perf_counter()
        for d in range(D):

            #
            # sampling Bs
            #
            SB = np.eye(Kest, Kest)
            aux = np.zeros(Kest)

            #
            # TODO: move the computation of SB out of the loop over D
            if C[d] == 1:

                # print('SB', SB, s2B, s2Y, Z)
                SB = 1 / s2Y * np.dot(Z, Z.T) + 1 / s2B * SB
                # print('SB2', SB)
                SB = inverse(SB)

                muB = 1 / s2Y * np.dot(Z, Yreal[d])
                np.dot(SB, muB, out=aux)
                Breal[d] = mvnrnd(aux, SB, rand_gen)

                muB = 1 / s2Y * np.dot(Z, Ypos[d])
                np.dot(SB, muB, out=aux)
                Bpos[d] = mvnrnd(aux, SB, rand_gen)

                muB = 1 / s2Y * np.dot(Z, Yint[d])
                np.dot(SB, muB, out=aux)
                Bint[d] = mvnrnd(aux, SB, rand_gen)

            elif C[d] == 2:
                SB = 1 / s2Y * np.dot(Z, Z.T) + 1 / s2B * SB
                SB = inverse(SB)

                muB = 1 / s2Y * np.dot(Z, Yreal[d])
                np.dot(SB, muB, out=aux)
                Breal[d] = mvnrnd(aux, SB, rand_gen)

                muB = 1 / s2Y * np.dot(Z, Yint[d])
                np.dot(SB, muB, out=aux)
                Bint[d] = mvnrnd(aux, SB, rand_gen)

            elif C[d] == 3:
                SB = 1 / s2Y * np.dot(Z, Z.T) + 1 / s2B * SB
                SB = inverse(SB)

                muB = 1 / s2Y * np.dot(Z, Ybin[d])
                np.dot(SB, muB, out=aux)
                Bbin[d] = mvnrnd(aux, SB, rand_gen)

            elif C[d] == 4:
                SB = 1 / s2Y * np.dot(Z, Z.T) + 1 / s2B * SB
                SB = inverse(SB)

                for r in range(1, R[d]):
                    muB = 1 / s2Y * np.dot(Z, Ycat[d, r])
                    np.dot(SB, muB, out=aux)
                    Bcat[d, r] = mvnrnd(aux, SB, rand_gen)

                muB = 1 / s2Y * np.dot(Z, Yord[d])
                np.dot(SB, muB, out=aux)
                Bord[d] = mvnrnd(aux, SB, rand_gen)

                muB = 1 / s2Y * np.dot(Z, Ycount[d])
                np.dot(SB, muB, out=aux)
                Bcount[d] = mvnrnd(aux, SB, rand_gen)

            #
            #
            SB = None
            muB = None
            aux = None

        time_b_end = perf_counter()
        logging.info('\tsampled Bs in {}'.format(time_b_end - time_b_ini))

        shuffle_start_t = perf_counter()
        rand_gen.shuffle(U)
        shuffle_end_t = perf_counter()
        logging.info('\shuffling U  {}'.format(shuffle_end_t - shuffle_start_t))

        time_sw_ini = perf_counter()
        # sampling S and W

        countErr, countErrC, Pp = sample_S_W(Bcat, Bcount, Bint, Bord, Bpos, Breal, C, N, R, W, Wint, X, Z, U, countErr,
                                             countErrC, D,
                                             maxX, meanX, s2U, s2Y, theta, theta_H, theta_L)

        for d in range(D):
            paramW = np.copy(alpha_W[d, :])
            for n in range(N):
                S[n, d] = mnrnd(Pp[d, n, :], rand_gen) + 1
                paramW[S[n, d] - 1] += 1

            if C[d] == 1:
                W[d, [0, 1, 3]] = rand_gen.dirichlet(paramW[[0, 1, 3]], size=1)
            elif C[d] == 2:
                W[d, [0, 1]] = rand_gen.dirichlet(paramW[[0, 1]], size=1)
            elif C[d] == 4:
                W[d, [0, 1, 2]] = rand_gen.dirichlet(paramW[[0, 1, 2]], size=1)

        time_sw_end = perf_counter()

        logging.info('\tsampled Ss and Ws in {}'.format(time_sw_end - time_sw_ini))

        if burn_in and it >= burn_in and save_all_params:
            sample['Bs'] = (np.copy(Breal), np.copy(Bint), np.copy(Bpos),
                            np.copy(Bbin), np.copy(Bcat), np.copy(Bord),
                            np.copy(Bcount))
            sample['Ys'] = (np.copy(Yreal), np.copy(Yint), np.copy(Ypos),
                            np.copy(Ybin), np.copy(Ycat), np.copy(Yord),
                            np.copy(Ycount))
            sample['theta'] = np.copy(theta)
        sample['S'] = np.copy(S)
        sample['W'] = np.copy(W)

        if save_ll_history and it % save_ll_history == 0:

            # u = rand_gen.normal(loc=0, scale=sYd, size=1000)
            # rand_gen.shuffle(u)

            ll_start_t = perf_counter()
            rand_gen.shuffle(U)
            LLX, WM = compute_log_likelihood_numba_u(X,
                                                     np.array(instance_ids),
                                                     np.array(feature_ids),
                                                     Z, W, alpha_W, Kest,
                                                     C.astype(np.uint8),
                                                     R.astype(np.int64), S,
                                                     s2Y, sYd, s2U,
                                                     U,
                                                     Breal,
                                                     Bint,
                                                     Bpos,
                                                     Bbin,
                                                     Bcat,
                                                     Bord,
                                                     Bcount,
                                                     maxX, meanX, minX,
                                                     theta, theta_L, theta_H,
                                                     Wint,
                                                     n_samples_cat=100,
                                                     # rand_gen=rand_gen
                                                     )
            MLL = logsumexp(LLX, axis=-1, b=WM)
            if np.any(np.isinf(MLL)):
                print('MLL with infs!!!!!\n',
                      # MLL[np.isinf(
                      # MLL)], LL[np.isinf(MLL)],
                      np.count_nonzero(WM, axis=-1), np.isinf(MLL).sum())
                # 0 / 0

            #
            # multiplying features

            # if compute_entry_wise:
            #     if compute_hard_ll:
            #         return MLL, HLL

            #     else:
            #         return MLL

            LLs = MLL.sum(axis=0)
            assert LLs.shape[0] == N

            avg_ll = LLs.mean()
            ll_end_t = perf_counter()
            ll_time = ll_end_t - ll_start_t
            print('\tComputed ll NUMBA {}  in {} secs'.format(avg_ll,
                                                              ll_time))
            sample['lls'] = LLs

            # if compute_hard_ll:
            #     HLLs = HLL.sum(axis=0)
            #     assert HLLs.shape[0] == N

            #     return LLs, HLLs

            # else:
            #     return LLs

            ##################################################################################################
            # OLD LL
            # ll_start_t = perf_counter()
            # # LL, avg_ll, HLL, avg_hll = compute_log_likelihood_numba(X,
            # LL, HLL = compute_log_likelihood_numba(X,
            #                                        instance_ids, feature_ids,
            #                                        Z, W, alpha_W, Kest,
            #                                        C, R, S,
            #                                        s2Y, sYd, s2U,
            #                                        # u,
            #                                        # U,
            #                                        Breal,
            #                                        Bint,
            #                                        Bpos,
            #                                        Bbin,
            #                                        Bcat,
            #                                        Bord,
            #                                        Bcount,
            #                                        maxX, meanX, minX,
            #                                        theta, theta_L, theta_H,
            #                                        Wint,
            #                                        n_samples_cat=100,
            #                                        compute_hard_ll=True,
            #                                        rand_gen=rand_gen
            #                                        )
            # ll_end_t = perf_counter()
            # ll_time = ll_end_t - ll_start_t
            # avg_ll = LL.mean()
            # avg_hll = HLL.mean()
            # print('\tComputed ll {} and hll {} in {} secs'.format(avg_ll, avg_hll,
            #                                                       ll_time))
            # sample['lls'] = LL
            #################################################################################################

            # ll_history.append(avg_ll)
            # hll_history.append(avg_hll)
            # ll_history.append(LL)
            # hll_history.append(HLL)
            # ll_count_history.append(it)
            # ll_time_history.append(ll_time)

            # u = rand_gen.normal(loc=0, scale=sYd, size=100)
        if burn_in and it >= burn_in and save_perf_history and it % save_perf_history == 0:
            ll_start_t = perf_counter()
            rand_gen.shuffle(U)
            LX = compute_miss_log_likelihood_numba(X, X_orig, Z, W, alpha_W, Kest,
                                                   C.astype(np.uint8), R.astype(np.int64),
                                                   S,
                                                   s2Y, sYd, s2U,
                                                   U,
                                                   Breal, Bint, Bpos, Bbin, Bcat, Bord, Bcount,
                                                   maxX, meanX, minX,
                                                   theta, theta_L, theta_H,
                                                   Wint, n_samples_cat=100,
                                                   # rand_gen,
                                                   )
            mv_LX = extend_matrix_one_value_per_row_nonzero(LX, X, X_orig)
            ll_end_t = perf_counter()
            sample['mv-lls'] = mv_LX
            logging.info('\nMVX s {}'.format(mv_LX.shape))

            logging.info('AVG miss LX {} min {} max {} (in {} secs)'.format(mv_LX.mean(),
                                                                            mv_LX.min(),
                                                                            mv_LX.max(),
                                                                            ll_end_t - ll_start_t,)
                         # LX[np.nonzero(LX)].mean(),                  # LX[np.nonzero(LX)].shape, mv_LX.shape
                         )

            perf_start_t = perf_counter()

            Yreal_m, Yint_m, Ypos_m, Ybin_m, Ycat_m, Yord_m, Ycount_m = est_mean_Ys(
                X, X_orig, Z, C, R, maxR,  Breal, Bint, Bpos, Bbin, Bcat, Bord, Bcount)

            X_hat, perf_dict = eval_predict_data(X, X_orig, C.astype(np.uint8), R.astype(np.int64), S,
                                                 #(Yreal, Yint, Ypos, Ycat, Yord, Ycount),
                                                 (Yreal_m, Yint_m, Ypos_m, Ycat_m, Yord_m, Ycount_m),
                                                 Wint,
                                                 theta, theta_L, theta_H,
                                                 maxX, minX, meanX, stdX,
                                                 # pos_continuous_metrics=pos_cont_perf_metrics,
                                                 discrete_metrics=disc_perf_metrics,
                                                 continuous_metrics=cont_perf_metrics)
            perf_end_t = perf_counter()
            logging.info('\tComputed prediction performances in {} secs'.format(
                perf_end_t - perf_start_t))
            print_perf_dict(perf_dict)

            ll_start_t = perf_counter()

            X_hat = preprocess_positive_real_data(X_hat, C)
            MVLX = compute_miss_log_likelihood_numba(X, X_hat, Z, W, alpha_W, Kest,
                                                     C.astype(np.uint8), R.astype(np.int64),
                                                     S,
                                                     s2Y, sYd, s2U,
                                                     U,
                                                     Breal, Bint, Bpos, Bbin, Bcat, Bord, Bcount,
                                                     maxX, meanX, minX,
                                                     theta, theta_L, theta_H,
                                                     Wint, n_samples_cat=100,
                                                     # rand_gen,
                                                     )
            mv_MVLX = extend_matrix_one_value_per_row_nonzero(MVLX, X, X_hat)
            ll_end_t = perf_counter()

            sample['mv-preds'] = X_hat
            sample['mv-preds-lls'] = mv_MVLX
            sample['mv-preds-scores'] = perf_dict

            # assert mv_MVLX.shape[0] == mv_LX.shape[0]

            logging.info('AVG miss imputations LL {} min {} max {} (in {} secs)'.format(mv_MVLX.mean(),
                                                                                        mv_MVLX.min(),
                                                                                        mv_MVLX.max(),
                                                                                        ll_end_t - ll_start_t,))

        iter_end_t = perf_counter()
        iter_time = iter_end_t - iter_start_t
        print('\n\tDone iteration {}/{} in {}\n\n'.format(it + 1, n_iters,
                                                          iter_time)  # ,  end="\r"
              )
        sample['time'] = iter_time
        samples.append(sample)

    print("countErr {}".format(countErr))
    print("countErrC {}".format(countErrC))

    print("\nComputing Likelihood \n")
    #
    # computing likelihoods
    L = np.zeros((D, N))

    for n in range(N):
        for d in range(D):

            L[d, n] = 0
            #
            # missing value?
            xnd = X[n, d]
            # if xnd == MISS_VALUE:
            if np.isnan(xnd):

                tnd = X_orig[n, d]

                # if tnd != MISS_VALUE:
                if not np.isnan(tnd):
                    #     print('tnd', tnd)

                    L[d, n] = 0
                    Baux = np.zeros(Kest)
                    Zn = Z[:, n]

                    if C[d] == 1 or C[d] == 2:

                        if S[n, d] == 1:
                            aux = np.dot(Zn, Breal[d])
                            L[d, n] = xpdf_re(tnd, 2 / (maxX[d] - meanX[d]),
                                              meanX[d], aux, s2Y, s2U)
                            if np.isnan(L[d, n]):
                                print(d, n)
                                0 / 0

                        elif S[n, d] == 2:
                            aux = np.dot(Zn, Bint[d])
                            L[d, n] = xpdf_int(tnd, Wint, theta_L[d], theta_H[d], aux, s2Y, s2U)
                            if np.isnan(L[d, n]):
                                print(d, n, tnd, Wint, theta_L[d], theta_H[d], aux, s2Y, s2U)
                                0 / 0

                        #
                        #
                        # FIXME: should this be dir? shouldn't it be pos?
                        elif S[n, d] == 3:
                            pass
                        elif S[n, d] == 4:
                            # Bd_view = gsl_matrix_submatrix (Bdir[d],0,0, Kest,1);
                            # matrix_multiply(&Zn_view.matrix,&Bd_view.matrix,aux,1,0,CblasTrans,CblasNoTrans);
                            #  LIK[d][n]= xpdf_dir(fre_1(tnd,2/maxX[d],0),theta_dir[d],gsl_matrix_get (aux, 0, 0),s2Y,s2u);
                            aux = np.dot(Zn, Bpos[d])
                            L[d, n] = xpdf_pos(tnd,  2 / maxX[d], aux, s2Y, s2U)

                        aux = None
                        Baux = None

                    elif C[d] == 4:
                        if S[n, d] == 1:
                            r = int(X_orig[n, d]) - 1

                            prodC = np.ones(100)
                            u = rand_gen.normal(loc=0, scale=sYd, size=100)

                            # for r2 in range(R[d]):
                            #     if r2 != r:
                            #         Baux = np.copy(Bcat[d, r])
                            #         Baux -= Bcat[d, r2]
                            #         aux = np.dot(Zn, Baux)

                            #         for ii in range(100):
                            #             # prodC[ii] = prodC[ii] * \
                            #             #     stats.norm.cdf(x=u[ii] + aux, scale=1)
                            #             prodC[ii] = prodC[ii] * \
                            #                 phi(x=u[ii] + aux)

                            # sumC = prodC.sum()
                            # L[d, n] = np.log(sumC / 100)
                            L[d, n] = cat_pdf_numba(d, r, Bcat, Zn, u,  100, R[d])
                            if np.isnan(L[d, n]):
                                print(d, n)
                                0 / 0

                        elif S[n, d] == 2:
                            aux = np.dot(Zn, Bord[d])

                            if tnd == 1:
                                # L[d, n] = np.log(stats.norm.cdf(
                                #     x=theta[d, int(tnd) - 1] - aux, scale=1))
                                L[d, n] = np.log(phi(
                                    x=theta[d, int(tnd) - 1] - aux))
                            elif tnd == R[d]:
                                # L[d, n] = np.log(
                                #     1 - stats.norm.cdf(x=theta[d, int(tnd) - 2] - aux, scale=1))
                                L[d, n] = np.log(
                                    1 - phi(x=theta[d, int(tnd) - 2] - aux))
                            else:
                                # L[d, n] = np.log(stats.norm.cdf(
                                #     theta[d, int(tnd) - 1] - aux, scale=1) - stats.norm.cdf(x=theta[d, int(tnd) - 2] - aux, scale=1))
                                L[d, n] = np.log(phi(
                                    theta[d, int(tnd) - 1] - aux) - phi(x=theta[d, int(tnd) - 2] - aux))

                        elif S[n, d] == 3:
                            aux = np.dot(Zn, Bcount[d])
                            # L[d, n] = np.log(stats.norm.cdf(
                            #     x=f_1(tnd + 1, 2 / maxX[d]) - aux, scale=1) - stats.norm.cdf(x=f_1(tnd, 2 / maxX[d]) - aux, scale=1))
                            L[d, n] = np.log(phi(
                                x=f_1(tnd + 1, 2 / maxX[d]) - aux) - phi(x=f_1(tnd, 2 / maxX[d]) - aux))
                            if np.isnan(L[d, n]):
                                print(d, n)
                                0 / 0

                        aux = None
                        Baux = None

    # LX = compute_miss_log_likelihood(X, X_orig, Z, W, alpha_W, Kest, C, R, S,
    #                                  s2Y, sYd, s2U,
    #                                  Breal, Bint, Bpos, Bbin, Bcat, Bord, Bcount,
    #                                  maxX, meanX, minX,
    #                                  theta, theta_L, theta_H,
    #                                  Wint, rand_gen,)

    # print('AVG miss LL', L[np.nonzero(L)].mean(), L[np.nonzero(L)].shape)
    # print('AVG miss LX', LX[np.nonzero(LX)].mean(), LX[np.nonzero(LX)].shape)
    # assert_array_almost_equal(L, LX)

    print('errs', countErr, countErrC)
    totErrors = countErr + countErrC

    # Ys = (Yreal, Yint, Ypos,  Ycat, Yord, Ycount)
    thetas = (theta, theta_L, theta_H)
    X_stats = maxX, minX, meanX

    return totErrors, L, thetas, X_stats, samples


@numba.jit(Tuple((int64, int64, float64[:, :, :]))(
    float64[:, :, :], float64[:, :], float64[:, :], float64[:,
                                                            :], float64[:, :], float64[:, :], int64[:], int64,
    int64[:], float64[:, :], int64, float64[:, :], float64[:,
                                                           :], float64[:, :, :], int64, int64, int64,
    float64[:], float64[:], float64, float64, float64[:, :], float64[:], float64[:]), nopython=True)
def sample_S_W(Bcat, Bcount, Bint, Bord, Bpos, Breal, C, N, R, W, Wint, X, Z, U, countErr, countErrC, D,
               maxX, meanX, s2U, s2Y, theta, theta_H, theta_L):
    Pp = np.zeros((D, N, 4))
    p = np.zeros(4)
    logw = np.log(W)
    # time_diffs = np.zeros((D, 4))
    logsumC = np.zeros(100, dtype=np.float64)
    for d in range(D):

        if C[d] == 1:
            # c_1_start_t = perf_counter()
            for n in range(N):
                xnd = X[n, d]

                if xnd == -1 or np.isnan(xnd):
                    Pp[d, n, ] = W[d, :]
                else:
                    Zn = Z[:, n]
                    aux = np.dot(Zn, Breal[d])
                    p[0] = xpdf_re(xnd, 2 / (maxX[d] - meanX[d]),
                                   meanX[d], aux, s2Y, s2U) + logw[d, 0]
                    # p[0] += np.log(W[d, 0])

                    # p[1] = 0
                    aux = np.dot(Zn, Bint[d])
                    p[1] = xpdf_int(xnd, Wint, theta_L[d], theta_H[d], aux, s2Y, s2U) + logw[d, 1]
                    # p[1] += np.log(W[d, 1])

                    p[2] = -np.inf

                    #
                    # FIXME: on Isabel's code this is Bint[d], should be Bpos[d]
                    # p[3] = 0
                    # Bd_view = gsl_matrix_submatrix(Bint[d], 0, 0, Kest, 1)
                    aux = np.dot(Zn, Bpos[d])
                    p[3] = xpdf_pos(xnd, 2 / maxX[d], aux, s2Y, s2U) + logw[d, 3]
                    # p[3] += np.log(W[d, 3])

                    p_n = np.exp(p)
                    psum = p_n.sum()

                    Pp[d, n, ] = p_n / psum

                if np.isnan(p.sum()):
                    countErrC += 1
                    Pp[d, n, ] = W[d, :]
            # c_1_end_t = perf_counter()
            # time_diffs[d, 0] += c_1_end_t - c_1_start_t
        elif C[d] == 2:
            # c_2_start_t = perf_counter()
            for n in range(N):
                xnd = X[n, d]

                if xnd == -1 or np.isnan(xnd):
                    Pp[d, n, ] = W[d, :]
                else:

                    Zn = Z[:, n]

                    # p[0] = 0
                    aux = np.dot(Zn, Breal[d])
                    p[0] = xpdf_re(xnd, 2 / (maxX[d] - meanX[d]),
                                   meanX[d], aux, s2Y, s2U) + logw[d, 0]
                    # p[0] += np.log(W[d, 0])

                    # p[1] = 0
                    aux = np.dot(Zn, Bint[d])
                    p[1] = xpdf_int(xnd, Wint, theta_L[d], theta_H[d], aux, s2Y, s2U) + logw[d, 1]
                    # p[1] += np.log(W[d, 1])

                    #
                    # no dir (only in Isabel's code) and no pos real
                    p[2] = -np.inf
                    p[3] = -np.inf

                    p_n = np.exp(p)
                    psum = p_n.sum()

                    Pp[d, n, ] = p_n / psum

                if np.isnan(p.sum()):
                    countErrC += 1
                    Pp[d, n, ] = W[d, :]

            # c_2_end_t = perf_counter()
            # time_diffs[d, 1] += c_2_end_t - c_2_start_t
        # binary types are fixed!
        elif C[d] == 3:
            pass

        elif C[d] == 4:

            # c_4_start_t = perf_counter()
            # c_5_time = 0
            for n in range(N):
                xnd = X[n, d]

                if xnd == -1 or np.isnan(xnd):
                    Pp[d, n, ] = W[d, :]
                else:
                    Zn = Z[:, n]
                    xnd = int(X[n, d])

                    #
                    # categorical
                    r = int(xnd) - 1

                    # c_5_start_t = perf_counter()
                    # p[0] = cat_pdf_numba(d, r, Bcat, Zn, U[n, d, :100], 100, R[d]) + logw[d, 0]
                    p[0] = cat_pdf_numba_prealloc(d, r, Bcat, Zn, U[n, d, :100],
                                                  100, R[d], logsumC) + logw[d, 0]
                    # c_5_end_t = perf_counter()
                    # c_5_time += c_5_end_t - c_5_start_t
                    # p[0] += np.log(W[d, 0])

                    #
                    # ordinal
                    # p[1] = 0
                    aux = np.dot(Zn, Bord[d])
                    xr = int(xnd) - 1
                    xrr = int(xnd) - 2

                    if xnd == 1:
                        p[1] = np.log(phi(x=theta[d, xr] - aux)) + logw[d, 1]
                        # if np.isnan(p[1]) or np.isinf(p[1]):
                        #    print('x {} R[d] {} p[1] {} logW {} x=1 {}'.format(
                        #        xnd, R[d], p[1],
                        #        np.log(W[d, 1]),
                        #        np.log(phi(x=theta[d, xr] - aux))))
                    elif xnd == R[d]:
                        p[1] = np.log(1 - phi(x=theta[d, xrr] - aux)) + logw[d, 1]
                        # if 1 - phi(x=theta[d, xrr] - aux) <= 0:
                        #    print('llog', 1 - phi(x=theta[d, xrr] - aux), theta[d, xrr], aux)
                        if np.isnan(p[1]) or np.isinf(p[1]):
                            #    print('x {} R[d] {} p[1] {} logW {} x=Rd {}'.format(
                            #        xnd, R[d], p[1],
                            #        np.log(W[d, 1]),
                            #        np.log(1 - phi(x=theta[d, xrr] - aux))
                            #    ))
                            p[1] = LOG_ZERO + logw[d, 1]

                    else:
                        p[1] = np.log(phi(x=theta[d, xr] - aux) -
                                      phi(x=theta[d, xrr] - aux)) + logw[d, 1]

                    # p[1] += np.log(W[d, 1])
                    # p[2] = 0
                    aux = np.dot(Zn, Bcount[d])
                    p[2] = np.log(phi(x=f_1(xnd + 1, 2 / maxX[d]) - aux) - phi(x=f_1(xnd, 2 / maxX[d]) - aux)) + logw[
                        d, 2]
                    # p[2] += np.log(W[d, 2])
                    p[3] = -np.inf

                    pmax = np.max(p)

                    if np.isinf(pmax):
                        p_n = np.exp(p)
                    else:
                        p_n = np.exp(p - pmax)

                    psum = p_n.sum()

                    Pp[d, n, ] = p_n / psum

                if np.isnan(p.sum()):
                    countErr += 1
                    # print("Discrete: n={}, d={}, sn(d)= {}, ".format(n, d, S[n, d]))
                    # print("sum_p= {}: p_n: {} psum {} pmax {} p {}".format(p.sum(),
                    #                                                       p_n, psum, pmax, p))
                    Pp[d, n, ] = W[d, :]

    #         c_4_end_t = perf_counter()
    #         time_diffs[d, 3] += c_4_end_t - c_4_start_t
    #         time_diffs[d, 2] += c_5_time

    # print('\tTIME DIFFS', time_diffs)
    return countErr, countErrC, Pp


if __name__ == '__main__':

    #########################################
    # creating the opt parser
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str,
                        help='Specify the path to compressed matlab dataset')

    parser.add_argument('-o', '--output-path', type=str,
                        default='./exp/',
                        help='Output path to exp result')

    parser.add_argument('--miss', type=str,
                        default=None,
                        help='Output path to missing value mask')

    parser.add_argument('-i', '--iters', type=int,
                        default=100,
                        help='Number of iterations of Gibbs sampler')

    parser.add_argument('-b', '--burn-in', type=int,
                        default=4000,
                        help='Number of iterations to discard samples')

    parser.add_argument('--s2z', type=float,
                        default=1.0,
                        help='Init variance for Zs')

    parser.add_argument('--s2b', type=float,
                        default=1.0,
                        help='Init variance for Bs')

    parser.add_argument('--s2u', type=float,
                        default=0.001,
                        help='Init variance for Us')

    parser.add_argument('--s2y', type=float,
                        default=1.0,
                        help='Init variance for Ys')

    parser.add_argument('--int-exp', type=float,
                        default=INT_EXP,
                        help='Interval RVs range expansion proportion')

    parser.add_argument('--s2theta', type=float,
                        default=1.0,
                        help='Init variance for thetas')

    parser.add_argument('-k', type=int,
                        default=None,
                        help='Number of latent space components (for Zs and Bs)'
                        ' If None, it gets the sqrt(D)')

    parser.add_argument('--ll-history', type=int,
                        default=100,
                        help='Whether to save the history of average model LL after a number of iterations')

    parser.add_argument('--perf-history', type=int,
                        default=1,
                        help='Whether to save the history of model prections')

    parser.add_argument('--ravg-buffer', type=int, nargs='+',
                        default=[100],
                        help='Running average buffer size for computing lls')

    parser.add_argument('--fig-size', type=int, nargs='+',
                        default=(10, 7),
                        help='A tuple for the explanation fig size ')

    parser.add_argument('--show-figs', action='store_true',
                        help='Whether to show by screen the plotted figures')

    parser.add_argument('--seed', type=int, nargs='?',
                        default=1337,
                        help='Seed for the random generator')

    parser.add_argument('--save-all-params', action='store_true',
                        help='Whether to save as results not only lls, Ws and preds but all params (Zs, Ys, Bs)')

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

    # logging.info("Starting with arguments:\n%s", args)

    #
    # creating output dirs if they do not exist
    # dataset_name = os.path.basename(args.dataset).replace('.mat', '')
    # date_string = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # folder_path = '{}-{}'.format(dataset_name, date_string)
    # exp_path = '{}_i{}-k{}-b{}-sz{}-sb{}-su{}-sy{}'.format(dataset_name,
    #                                                        args.iters, args.k, args.burn_in,
    #                                                        args.s2z, args.s2b, args.s2u, args.s2y)
    # output_path = os.path.join(args.output_path, folder_path, exp_path)
    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True)

    args_out_path = os.path.join(output_path, 'args.json')
    json_args = json.dumps(vars(args))
    logging.info("Starting with arguments:\n%s\n\tdumped at %s", json_args, args_out_path)
    with open(args_out_path, 'w') as f:
        f.write(json_args)

    X, C, R = None, None, None
    X_orig = None
    train_data_path = os.path.join(args.dataset, 'train.data')
    if os.path.exists(train_data_path):
        load_start_t = perf_counter()
        with open(train_data_path, 'rb') as f:
            X = pickle.load(f)
        load_end_t = perf_counter()
        logging.info('Loaded train data {} from {} (in {} secs)'.format(X.shape,
                                                                        train_data_path,
                                                                        load_end_t - load_start_t))
        X_orig = np.copy(X)

        stats_map = {}
        data_stats_path = os.path.join(args.dataset, 'data.stats')
        with open(data_stats_path, 'rb') as f:
            stats_map = pickle.load(f)
        meta_types = stats_map['meta-types']
        domains = stats_map['domains']

        N, D = X.shape
        print('Loaded data with shape: {}'.format(X.shape))

        C = np.zeros(D, dtype=np.int64)
        for i, t in enumerate(meta_types):
            if t == MetaType.BINARY:
                C[i] = 3

            elif t == MetaType.REAL:
                if np.all(domains[i] >= 0):
                    C[i] = 1
                else:
                    C[i] = 2
            elif t == MetaType.DISCRETE:
                C[i] = 4
        print('\thMeta-types:{}'.format(C))

        max_X = np.array([np.nanmax(X[:, d]) for d in range(D)])
        logging.info('Max X: {}'.format(max_X))

        R = np.ones(D, dtype=np.int64)
        for d in range(D):
            if meta_types[d] == MetaType.DISCRETE:
                X[:, d] += 1
                R[d] = int(max_X[d] + 1)
        print('\tMaximal discrete cardinality: {}'.format(R))

    else:
        #
        # loading isabel data
        data_dict = scipy.io.loadmat(args.dataset)

        X = data_dict['X']
        X = X.astype(np.float64)
        print('Loaded data with shape: {}'.format(X.shape))
        X_orig = np.copy(X)

        C = data_dict['T'].flatten().astype(np.int64)
        print('\thMeta-types:{}'.format(C))

        R = data_dict['R'].flatten().astype(np.int64)
        print('\tMaximal discrete cardinality: {}'.format(R))

    rand_gen = np.random.RandomState(args.seed)

    #
    #
    if args.miss:
        # X_miss = miss_data_dict['miss']
        print('Looking for pickle at dir... {}'.format(args.miss))
        try:
            with open(args.miss, 'rb') as f:
                X_miss = pickle.load(f)

        except:
            print('FAILED to load pickle, trying matlab .mat file')
            miss_data_dict = scipy.io.loadmat(args.miss)
            X_miss_ids = miss_data_dict['miss']
            X_miss_ids = np.unravel_index(X_miss_ids.flatten().astype(np.int64) - 1,
                                          X.shape, order='F')
            X_miss = np.zeros(X.shape, dtype=bool)
            X_miss[X_miss_ids] = True

        print('Loaded missing data mask with shape: {}'.format(X_miss.shape))
        assert X_miss.shape == X.shape

        X[X_miss] = np.nan

    # import cProfile
    # cProfile.run("infer_data_types(X, C, R,                                   s2Z=1, s2B=1, s2Y=1, s2U=0.001, s2theta=1,                                   n_iters=20, max_K=5,                                   X_orig=X,                                   rand_gen=None)")

    infer_start_t = perf_counter()
    # Kest, W, totErrors, L, Ys, thetas, X_stats, lls_history, times, perf_history = \
    totErrors, L, thetas, X_stats, samples = infer_data_types(X, C, R,
                                                              s2Z=args.s2z, s2B=args.s2b, s2Y=args.s2y,
                                                              s2U=args.s2u, s2theta=args.s2theta,
                                                              n_iters=args.iters, max_K=args.k,
                                                              X_orig=X_orig,
                                                              burn_in=args.burn_in,
                                                              save_ll_history=args.ll_history,
                                                              save_perf_history=args.perf_history,
                                                              int_exp=args.int_exp,
                                                              save_all_params=args.save_all_params,
                                                              rand_gen=rand_gen)

    infer_end_t = perf_counter()
    print('Done in {}'.format(infer_end_t - infer_start_t))

    print(L)
    print('AVG miss LL', L[np.nonzero(L)].mean(), L[np.nonzero(L)].shape)

    print('Kest {} W_0 {} W_last {}totErrors {} L {} {}'.format(args.k, samples[0]['W'],
                                                                samples[-1]['W'],
                                                                totErrors, L, len(L.nonzero())))
    print(L[L.nonzero()], L.nonzero())

    # hll_hist = np.array([s['lls'].mean() for s in samples])
    # ll_hist, hll_hist, ll_counts = lls_history

    #
    # dumping results individually
    dump_samples_to_pickle(samples, output_path, out_file_name='lls.pklz',
                           key='lls', count_key='id')
    dump_samples_to_pickle(samples, output_path, out_file_name='mv-lls.pklz',
                           key='mv-lls', count_key='id')
    dump_samples_to_pickle(samples, output_path, out_file_name='mv-preds.pklz',
                           key='mv-preds', count_key='id')
    dump_samples_to_pickle(samples, output_path, out_file_name='mv-preds-lls.pklz',
                           key='mv-preds-lls', count_key='id')
    dump_samples_to_pickle(samples, output_path, out_file_name='mv-preds-scores.pklz',
                           key='mv-preds-scores', count_key='id')

    dump_samples_to_pickle(samples, output_path, out_file_name='W.pklz',
                           key='W', count_key='id')
    dump_samples_to_pickle(samples, output_path, out_file_name='S.pklz',
                           key='S', count_key='id')

    # #
    # # dropping everything to a pickle?
    # result_pickle_path = os.path.join(output_path, 'result-dump.pklz')
    # with gzip.open(result_pickle_path, 'wb') as f:
    #     res = {'K': args.k, 'samples': samples,
    #            # 'W': W,
    #            'totErrors': totErrors,
    #            'L': L,
    #            #'Ys': Ys,
    #            'thetas': thetas, 'X_stats': X_stats,
    #            # 'll_history': ll_hist,
    #            # 'hll_history': hll_hist,
    #            # 'll_count_history': ll_counts,
    #            # 'perf_history': perf_history,
    #            #'times': times
    #            }
    #     pickle.dump(res, f)
    # print('results dumped to {}'.format(result_pickle_path))

    ll_counts = np.array([s['id'] for s in samples if 'lls' in s])
    ll_hist = np.array([s['lls'] for s in samples if 'lls' in s])
    avg_ll_hist = ll_hist.mean(axis=1)
    # print(ll_counts, avg_ll_hist)

    mv_counts = np.array([s['id'] for s in samples if 'mv-lls' in s])
    mv_train_ll_hist = np.array([s['mv-lls'] for s in samples if 'mv-lls' in s])
    print('\nMV shape\n', mv_train_ll_hist.shape)
    mv_train_avg_ll_hist = mv_train_ll_hist.mean(axis=1) if len(mv_train_ll_hist) > 0 else None

    pp_counts = np.array([s['id'] for s in samples if 'mv-preds-scores' in s])
    perf_history = [s['mv-preds-scores'] for s in samples if 'mv-preds-scores' in s]

    #
    # unpacking results
    # Yreal, Yint, Ypos,  Ycat, Yord, Ycount = Ys
    # theta, theta_L, theta_H = thetas
    maxX, minX, meanX = X_stats

    #
    # plotting likelihoods
    if args.ll_history:

        import matplotlib
        if not args.show_figs:
            matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages

        # ll_hist = np.array(ll_hist).T
        # hll_hist = np.array(hll_hist).T
        # print('LL hist shape', ll_hist.shape)

        # avg_ll = ll_hist.mean(axis=0)
        # avg_hll = hll_hist.mean(axis=0)

        fig, ax = plt.subplots(figsize=args.fig_size)
        ax.plot(ll_counts, avg_ll_hist, label='sample avg LL')
        # ax.plot(ll_counts, avg_ll, label='sample avg LL')
        # ax.plot(ll_counts, avg_hll, label='sample avg HLL')
        for b in args.ravg_buffer:
            ravg_ll_hist = running_avg_numba(ll_hist.T, b)
            # ravg_hll_hist = running_avg_numba(hll_hist, b)
            ravg_ll = ravg_ll_hist.mean(axis=0)
            # ravg_hll = ravg_hll_hist.mean(axis=0)
            ax.plot(ll_counts, ravg_ll, label='last {} samples avg LL'.format(b))
            # ax.plot(ll_counts, ravg_hll, label='last {} samples avg HLL'.format(b))
        ax.legend()

        ll_hist_output = os.path.join(output_path, 'll-history.pdf')
        pp = PdfPages(ll_hist_output)
        pp.savefig(fig)
        pp.close()
        print('Saved LL history to {}'.format(ll_hist_output))

        if args.show_figs:
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

            mv_ll_hist_output = os.path.join(output_path, 'miss-ll-history.pdf')
            pp = PdfPages(mv_ll_hist_output)
            pp.savefig(fig)
            pp.close()
            print('Saved MV LL history to {}'.format(mv_ll_hist_output))

            if args.show_figs:
                plt.show()

    # #
    # # plotting predictions
    if args.perf_history and len(perf_history) > 0:

        perf_path = os.path.join(output_path, 'perf@{}iter'.format(args.iters))
        os.makedirs(perf_path, exist_ok=True)

        import matplotlib

        if not args.show_figs:
            matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages

        for d in range(X.shape[1]):

            metrics_d = perf_history[0][d].keys()
            print('Metrics for feature {}: {}'.format(d, metrics_d))

            for m in metrics_d:
                fig, ax = plt.subplots(figsize=args.fig_size)
                perf_d = [p[d][m] for p in perf_history]
                ax.plot(pp_counts, perf_d, label=m)
                ax.legend()

                perf_d_output = os.path.join(perf_path, '{}-f{}-perf.pdf'.format(m, d))
                pp = PdfPages(perf_d_output)
                pp.savefig(fig)
                pp.close()
                print('Saved perf {} for feature {} to {}'.format(m, d, perf_d_output))

                if args.show_figs:
                    plt.show()
