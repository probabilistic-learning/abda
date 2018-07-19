"""
Common utils for probability density processings

@author: antonio vergari
"""
import numpy as np
import scipy.stats
from scipy.special import logsumexp

import numba
from numba import jit, uint8, int64, float64, optional, boolean
from numba.types import Tuple


def is_missing_value(x):
    """
    TODO: use this function in the future to determine if a value has to be treated as missing
    """
    return np.isnan(x)


@numba.jit(nopython=True)
def numba_gaussian_pdf(x, scale):
    """
    Pdf of a gaussian with zero mean and std sigma to be wrapped in numba for efficiency
    """

    u = x / np.abs(scale)
    p = (1 / (np.sqrt(2 * np.pi) * np.abs(scale))) * np.exp(-u * u / 2)
    return p


@numba.jit(nopython=True)
def fre_1(x, w, mu):
    """
    Scaling and centering inverse function for real data pseudo transformations

    WRITEME: args

    TODO: refactor names
    """
    return w * (x - mu)


@numba.jit(nopython=True)
def fre(y, w, mu):
    """
    Scaling and centering  function for real data pseudo transformations

    WRITEME: args

    TODO: refactor names
    """
    return w * y + mu


@numba.jit(nopython=True)
def f_1(x, w):
    """
    Inverse transformation for positive real data

    WRITEME: args

    TODO: refactor names
    """
    return np.log(np.exp(w * x) - 1)
    # return w * np.log(np.exp(x) - 1)


@numba.jit(nopython=True)
def f_pos(x, w):
    """
    Transformation for positive real data

    WRITEME: args

    TODO: refactor names
    """
    # return np.log(np.exp(w * x) + 1)
    return w * np.log(np.exp(x) + 1)


@numba.jit(nopython=True)
def f_int(x,  w,  theta_L,  theta_H):
    """
    Transformation for interval real data

    WRITEME: args

    TODO: refactor names
    """
    return (theta_H - theta_L) / (1 + np.exp(-w * x)) + theta_L


@numba.jit(nopython=True)
def fint_1(x,  w,  theta_L,  theta_H):
    """
    Inverse transformation for interval real data

    WRITEME: args

    TODO: refactor names
    """
    return -1 / w * np.log((theta_H - x) / (x - theta_L))


@numba.jit()
def f_cat(x, Rd):
    """
    Transformation for categorical data

    x is assumed to have shape Rxn (also Rx1)

    Rd: uint
        cardinality of values for attribute
    """
    # print('FCAT', x.shape)
    # print(x[:Rd])
    return np.argmax(x[:Rd, :], axis=0) + 1


@numba.jit(nopython=True)
def f_ord(x, thetas, Rd):
    """
    Transformation for ordinal data

    x is assumed to have shape n

    thetas intervals (length maxRd)

    Rd: uint
        cardinality of values for attribute

    """

    # print(thetas[:Rd - 1])
    return np.digitize(x, thetas[:Rd - 1]) + 1


@numba.jit(nopython=True)
def f_count(x, w):
    """
    Transformation for count data

    x is assumed to have shape n

    thetas intervals (length maxRd)

    Rd: uint
        cardinality of values for attribute

    """

    return np.floor(f_pos(x, w)) + 1


@numba.jit(nopython=True)
def dfre_1(x,  w):
    """
    Derivative of the inverse transformations df^-1(x)/dx

    WRITEME: args

    TODO: refactor names
    """
    return w


@numba.jit(nopython=True)
def df_1(x,  w):

    return w / (1 - np.exp(-w * x))


@numba.jit(nopython=True)
def dfint_1(x, w, theta_L, theta_H):
    return 1 / w * (theta_H - theta_L) / ((theta_H - x) * (x - theta_L))


@numba.jit(nopython=True)
def inverse(A):
    """
    Invert data matrix
    Question:
    """
    return np.linalg.inv(A)


@numba.jit(nopython=True)
def xpdf_re(x, w, muX, mu, s2Y, s2u):

    # return np.log(stats.norm.pdf(x=fre_1(x, w, muX) - mu, scale=np.sqrt(s2Y + s2u))) + np.log(np.abs(dfre_1(x, w)))
    return np.log(numba_gaussian_pdf(x=fre_1(x, w, muX) - mu, scale=np.sqrt(s2Y + s2u))) + np.log(np.abs(dfre_1(x, w)))


EPS_ZERO = 1e-7


@numba.jit(nopython=True)
# @numba.jit(float64(float64, float64, float64, float64, float64), nopython=False)
def xpdf_pos(x, w, mu, s2Y, s2u):

    # # return np.log(stats.norm.pdf(x=f_1(x, w) - mu, scale=np.sqrt(s2Y + s2u))) + np.log(abs(df_1(x, w)))
    # gpd = numba_gaussian_pdf(x=f_1(x, w) - mu, scale=np.sqrt(s2Y + s2u))
    # # if np.isclose(gpd, 0):
    # gpd += EPS_ZERO

    return np.log(numba_gaussian_pdf(x=f_1(x, w) - mu, scale=np.sqrt(s2Y + s2u))) + np.log(abs(df_1(x, w)))


@numba.jit(nopython=True)
def xpdf_int(x, w, theta_L, theta_H, mu, s2Y, s2u):

    # return np.log(stats.norm.pdf(x=fint_1(x, w, theta_L, theta_H) - mu, scale=np.sqrt(s2Y + s2u))) + np.log(abs(dfint_1(x, w, theta_L, theta_H)))
    return np.log(numba_gaussian_pdf(x=fint_1(x, w, theta_L, theta_H) - mu, scale=np.sqrt(s2Y + s2u))) + np.log(abs(dfint_1(x, w, theta_L, theta_H)))


@numba.jit(nopython=True)
def WNpdf(x, theta, mu, sigma, n):

    kopt = int(mu / theta)
    px = 0.0
    for i in range(kopt - n, kopt + n + 1):
        # px += stats.norm.pdf(x=x - mu + i * theta, scale=sigma)
        px += numba_gaussian_pdf(x=x - mu + i * theta, scale=sigma)

    return px


@numba.jit(nopython=True)
def xpdf_dir(x, theta, mu, s2Y, s2u, n=100):

    return np.log(WNpdf(x, theta, mu, np.sqrt(s2Y + s2u), n))


@numba.jit
def mvnrnd(Mu, Sigma, rand_gen):
    """
    Sample from a multivariate gaussian with mean Mu (size d) and covariance matrix Sigma (size dxd)

    Using numpy's random utils
    """

    return rand_gen.multivariate_normal(Mu, Sigma)


@numba.jit
def mnrnd(p, rand_gen):
    """
    Sample from a categorical distribution
    """
    n_vals = len(p)

    return rand_gen.choice(n_vals, replace=False, p=p)


"""
Literal porting of gsl functions to operate on the the CDF of a gaussian for efficiently compiling them with numba

see https://github.com/ampl/gsl/blob/master/cdf/gauss.c
"""


@numba.jit
def truncnormrnd_scipy(mu, sigma, xlo, xhi, rand_gen, n_samples=1):
    """
    Sample from a truncated normal with mean mu, variance sigma and truncated at [xlo, xhi]
    """
    X = scipy.stats.truncnorm((xlo - mu) / sigma, (xhi - mu) / sigma, loc=mu, scale=sigma)

    return X.rvs(size=n_samples)


@numba.jit(nopython=True)
def rat_eval(a, na, b, nb,  x):
    """
    """

    u = a[na - 1]

    i = na - 1

    while i > 0:
        u = x * u + a[i - 1]
        i -= 1

    v = b[nb - 1]

    j = nb - 1
    while j > 0:
        v = x * v + b[j - 1]
        j -= 1

    r = u / v

    return r


A_SMALL = np.array([3.387132872796366608, 133.14166789178437745,
                    1971.5909503065514427, 13731.693765509461125,
                    45921.953931549871457, 67265.770927008700853,
                    33430.575583588128105, 2509.0809287301226727])

B_SMALL = np.array([1.0, 42.313330701600911252,
                    687.1870074920579083, 5394.1960214247511077,
                    21213.794301586595867, 39307.89580009271061,
                    28729.085735721942674, 5226.495278852854561])


@numba.jit(nopython=True)
def small(q):
    """
    """
    r = 0.180625 - q * q
    x = q * rat_eval(A_SMALL, 8, B_SMALL, 8, r)

    return x


A_INTERMEDIATE = np.array([1.42343711074968357734, 4.6303378461565452959,
                           5.7694972214606914055, 3.64784832476320460504,
                           1.27045825245236838258, 0.24178072517745061177,
                           0.0227238449892691845833, 7.7454501427834140764e-4])


B_INTERMEDIATE = np.array([1.0, 2.05319162663775882187,
                           1.6763848301838038494, 0.68976733498510000455,
                           0.14810397642748007459, 0.0151986665636164571966,
                           5.475938084995344946e-4, 1.05075007164441684324e-9])


@numba.jit(nopython=True)
def intermediate(r):
    """
    """
    x = rat_eval(A_INTERMEDIATE, 8, B_INTERMEDIATE, 8, (r - 1.6))
    return x


A_TAIL = np.array([6.6579046435011037772, 5.4637849111641143699,
                   1.7848265399172913358, 0.29656057182850489123,
                   0.026532189526576123093, 0.0012426609473880784386,
                   2.71155556874348757815e-5, 2.01033439929228813265e-7])

B_TAIL = np.array([1.0, 0.59983220655588793769,
                   0.13692988092273580531, 0.0148753612908506148525,
                   7.868691311456132591e-4, 1.8463183175100546818e-5,
                   1.4215117583164458887e-7, 2.04426310338993978564e-15])


@numba.jit(nopython=True)
def tail(r):
    x = rat_eval(A_TAIL, 8, B_TAIL, 8, (r - 5.0))
    return x


@numba.jit(nopython=True)
def gsl_cdf_ugaussian_Pinv(P):

    dP = P - 0.5

    if P == 1.0:
        return np.inf
    elif P == 0.0:
        return -np.inf

    if np.abs(dP) <= 0.425:
        x = small(dP)
        return x

    pp = P if P < 0.5 else 1.0 - P

    r = np.sqrt(-np.log(pp))

    if r <= 5.0:
        x = intermediate(r)
    else:
        x = tail(r)

    if (P < 0.5):
        return -x
    else:
        return x


M_2_SQRTPI = 1.12837916709551257389615890312  # 2/sqrt(pi)
# M_SQRT1_2 = 1.77245385090551602729816748334  # sqrt(pi)
M_SQRT1_2 = 0.70710678118654752440084436210  # sqrt(1/2)
M_1_SQRT2PI = (M_2_SQRTPI * M_SQRT1_2 / 2.0)
M_SQRT2 = 1.41421356237309504880168872421  # sqrt(2)
SQRT32 = 4.0 * M_SQRT2

"""
 * IEEE double precision dependent constants.

 *
 * GAUSS_EPSILON: Smallest positive value such that
 *                gsl_cdf_gaussian(x) > 0.5.
 * GAUSS_XUPPER: Largest value x such that gsl_cdf_gaussian(x) < 1.0.
 * GAUSS_XLOWER: Smallest value x such that gsl_cdf_gaussian(x) > 0.0.
 */
"""
GSL_DBL_EPSILON = 2.2204460492503131e-16
GAUSS_EPSILON = GSL_DBL_EPSILON / 2
GAUSS_XUPPER = 8.572
GAUSS_XLOWER = -37.519
GAUSS_SCALE = 16.0


@numba.jit(nopython=True)
def get_del(x, rational):

    xsq = np.floor(x * GAUSS_SCALE) / GAUSS_SCALE
    _del = (x - xsq) * (x + xsq)
    _del *= 0.5

    result = np.exp(-0.5 * xsq * xsq) * np.exp(-1.0 * _del) * rational

    # if np.isinf(result) or np.isnan(result):
    # print('geddel res {} x {} rat {} del {}', result, x, rational, _del)
    return result


A_G_SMALL = np.array([2.2352520354606839287,
                      161.02823106855587881,
                      1067.6894854603709582,
                      18154.981253343561249,
                      0.065682337918207449113
                      ])
B_G_SMALL = np.array([47.20258190468824187,
                      976.09855173777669322,
                      10260.932208618978205,
                      45507.789335026729956
                      ])


@numba.jit(nopython=True)
def gauss_small(x):
    """
    Normal cdf for fabs(x) < 0.66291
    """

    xsq = x * x
    xnum = A_G_SMALL[4] * xsq
    xden = xsq

    for i in range(3):
        xnum = (xnum + A_G_SMALL[i]) * xsq
        xden = (xden + B_G_SMALL[i]) * xsq

    result = x * (xnum + A_G_SMALL[3]) / (xden + B_G_SMALL[3])

    # if np.isinf(result) or np.isnan(result):
    # print('gauss-small res {} x {}', result, x)

    return result


C_G_MEDIUM = np.array([0.39894151208813466764,
                       8.8831497943883759412,
                       93.506656132177855979,
                       597.27027639480026226,
                       2494.5375852903726711,
                       6848.1904505362823326,
                       11602.651437647350124,
                       9842.7148383839780218,
                       1.0765576773720192317e-8
                       ])
D_G_MEDIUM = np.array([22.266688044328115691,
                       235.38790178262499861,
                       1519.377599407554805,
                       6485.558298266760755,
                       18615.571640885098091,
                       34900.952721145977266,
                       38912.003286093271411,
                       19685.429676859990727
                       ])


@numba.jit(nopython=True)
def gauss_medium(x):
    """
    Normal cdf for 0.66291 < fabs(x) < sqrt(32).
    """

    absx = np.abs(x)

    xnum = C_G_MEDIUM[8] * absx
    xden = absx

    for i in range(7):
        xnum = (xnum + C_G_MEDIUM[i]) * absx
        xden = (xden + D_G_MEDIUM[i]) * absx

    temp = (xnum + C_G_MEDIUM[7]) / (xden + D_G_MEDIUM[7])

    result = get_del(x, temp)

    # if np.isinf(result) or np.isnan(result):
    # print('gauss-medium res {} x {}', result, x)

    return result


P_G_LARGE = np.array([0.21589853405795699,
                      0.1274011611602473639,
                      0.022235277870649807,
                      0.001421619193227893466,
                      2.9112874951168792e-5,
                      0.02307344176494017303
                      ])
Q_G_LARGE = np.array([1.28426009614491121,
                      0.468238212480865118,
                      0.0659881378689285515,
                      0.00378239633202758244,
                      7.29751555083966205e-5
                      ])


@numba.jit(nopython=True)
def gauss_large(x):
    """
    Normal cdf for
    * {sqrt(32) < x < GAUSS_XUPPER} union {GAUSS_XLOWER < x < -sqrt(32)}.
    """

    absx = np.abs(x)
    xsq = 1.0 / (x * x)
    xnum = P_G_LARGE[5] * xsq
    xden = xsq

    for i in range(4):

        xnum = (xnum + P_G_LARGE[i]) * xsq
        xden = (xden + Q_G_LARGE[i]) * xsq

    temp = xsq * (xnum + P_G_LARGE[4]) / (xden + Q_G_LARGE[4])
    temp = (M_1_SQRT2PI - temp) / absx

    result = get_del(x, temp)

    # if np.isinf(result) or np.isnan(result):
    # print('gauss-large res  x ', result, x)

    return result


@numba.jit(float64(float64), nopython=True)
def phi(x):

    absx = np.abs(x)

    if (absx < GAUSS_EPSILON):

        result = 0.5
        return result

    elif (absx < 0.66291):

        result = 0.5 + gauss_small(x)
        return result

    elif (absx < SQRT32):

        result = gauss_medium(x)

        if (x > 0.0):

            result = 1.0 - result

        return result

    elif (x > GAUSS_XUPPER):

        result = 1.0
        return result

    elif (x < GAUSS_XLOWER):

        result = 0.0
        return result

    else:

        result = gauss_large(x)

        if (x > 0.0):

            result = 1.0 - result

    return result


@numba.jit
# def truncnormrnd(mu, sigma, xlo, xhi, n_samples=1):
def truncnormrnd(mu, sigma, xlo, xhi, rand_gen, n_samples=1):
    """
    Sample from a truncated normal
    """
    _plo = phi((xlo - mu) / sigma)
    _phi = phi((xhi - mu) / sigma)
    r = rand_gen.rand()
    # r = np.random.rand()
    r = _plo + (_phi - _plo) * r
    z = gsl_cdf_ugaussian_Pinv(r)
    x = mu + z * sigma
    return x


@numba.jit
def truncnormrnd_det(i, mu, sigma, xlo, xhi, rand_gen, n_samples=1):
    """
    Deterministic dummy for testing purposes
    """
    return mu + i


@numba.jit
def mvnrnd_det(i, Mu, Sigma, rand_gen):
    """
    Deterministic dummy for testing purposes
    """

    return Mu + i


@numba.jit
def mnrnd_det(i, p, rand_gen):
    """
    Deterministic dummy for testing purposes
    """

    return i % len(p)


@numba.jit
def random_dirichlet_det(i, p, size):
    """
    Deterministic dummy for testing purposes
    """
    _p = np.array([i + 1 for j in range(len(p))])
    return _p / _p.sum()


@numba.jit
def random_normal_det(i, loc, scale, size=1):
    """
    Deterministic dummy for testing purposes
    """
    return np.array([loc + i for j in range(size)])


@numba.jit
def random_rand_det(i, _max=500):
    """
    Deterministic dummy for testing purposes
    """
    return i / _max


from numba import jit, uint8, int64, float64, optional, boolean


@numba.jit(float64[:](float64[:, :]), nopython=True)
def amax_2D_numba(a):
    """
    Optimized maximum version of numba operating along the last axis of a 2D vector
    """
    N, D = a.shape
    # amax = np.zeros((N, D))

    # for n in range(N):
    #     amax[n, :] = np.max(a[n, :])
    amax = np.zeros(N)

    for n in range(N):
        amax[n] = np.max(a[n, :])
        if not np.isfinite(amax[n]):
            amax[n] = 0

    return amax


@numba.jit(float64[:, :](float64[:, :, :]), nopython=True)
def amax_3D_numba(a):
    """
    Optimized maximum version of numba operating along the last axis of a 2D vector
    """
    N, D, L = a.shape
    # amax = np.zeros((N, D))

    # for n in range(N):
    #     amax[n, :] = np.max(a[n, :])
    amax = np.zeros((N, D))

    for n in range(N):
        for d in range(D):
            amax[n, d] = np.max(a[n, d, :])
            if not np.isfinite(amax[n, d]):
                amax[n, d] = 0

    return amax


@numba.jit(float64[:](float64[:, :], float64[:, :]),
           # locals={'a_max': float64[:, :]},
           nopython=True)
def logsumexp_2D_numba(a, b):
    """
    Optimized logsumexp op in numba, operating only on last axis of a 2D vector
    """

    # a_max = np.max(a, axis=-1)

    N, D = a.shape
    a_max = amax_2D_numba(a)

    tmp = np.zeros((N, D))
    for d in range(D):
        tmp[:, d] = b[:, d] * np.exp(a[:, d] - a_max)

    s = np.sum(tmp, axis=-1)

    out = np.log(s)

    out += a_max

    return out


@numba.jit(float64[:, :](float64[:, :, :], float64[:, :, :]),
           # locals={'a_max': float64[:, :]},
           nopython=True)
def logsumexp_3D_numba(a, b):
    """
    Optimized logsumexp op in numba, operating only on last axis of a 3D vector
    """

    # a_max = np.max(a, axis=-1)

    N, D, L = a.shape
    a_max = amax_3D_numba(a)

    tmp = np.zeros((N, D, L))
    for l in range(L):
        tmp[:, :, l] = b[:, :,  l] * np.exp(a[:, :, l] - a_max)

    s = np.sum(tmp, axis=-1)

    out = np.log(s)

    out += a_max

    return out


@numba.jit(float64(float64[:], float64[:]), nopython=True)
def logsumexp_1Dw_numba(a, b):
    """
    Optimized logsumexp op in numba, operating only on last axis of a 1D vector
    """

    # a_max = np.max(a, axis=-1)
    a_max = np.max(a)
    if not np.isfinite(a_max):
        a_max = 0

    tmp = b * np.exp(a - a_max)

    s = np.sum(tmp, axis=-1)

    out = np.log(s)

    out += a_max

    return out


@numba.jit(float64(float64[:]), nopython=True)
def logsumexp_1D_numba(a):
    """
    Optimized logsumexp op in numba, operating only on last axis of a 1D vector
    """

    # a_max = np.max(a, axis=-1)

    a_max = np.max(a)
    if not np.isfinite(a_max):
        a_max = 0

    tmp = np.exp(a - a_max)

    s = np.sum(tmp, axis=-1)

    out = np.log(s)

    out += a_max

    return out


# @numba.jit(nopython=True)


# @numba.jit(float64(int64, int64,
#                    float64[:, :, :],
#                    float64[:], float64[:], int64, int64), nopython=True)
# def cat_pdf_numba(d, r, Bcat, Zn, u,  n_samples_cat, R_d):
#     prodC = np.ones(n_samples_cat)
#     # u = rand_gen.normal(loc=0, scale=sYd, size=n_samples_cat)
#     # u = np.array([np.random.normal(loc=0, scale=sYd)
#     #               for j in range(n_samples_cat)])

#     for r2 in range(R_d):
#         if r2 != r:
#             Baux = np.copy(Bcat[d, r, :])
#             Baux -= Bcat[d, r2, :]
#             aux = np.dot(Zn, Baux)

#             for ii in range(n_samples_cat):
#                 # prodC[ii] = prodC[ii] * \
#                 #     stats.norm.cdf(x=u[ii] + aux, scale=1)
#                 # print(ii, u[ii], aux, phi(x=u[ii] + aux))
#                 prodC[ii] = prodC[ii] * \
#                     phi(x=u[ii] + aux)

#     sumC = prodC.sum()
#     # print(sumC, Zn, R_d, Bcat[d, r, :], d, r)
#     return np.log(sumC / n_samples_cat)


@numba.jit(float64(int64, int64,
                   optional(float64[:, :, :]),
                   float64[:], float64[:],
                   int64, uint8), nopython=True)
def cat_pdf_numba(d, r, Bcat, Zn, u,  n_samples_cat, R_d):
    # logsumC = float64(0)
    logsumC = np.zeros(n_samples_cat, dtype=np.float64)

    for r2 in range(R_d):
        if r2 != r:
            Baux = np.copy(Bcat[d, r, :])
            Baux -= Bcat[d, r2, :]
            aux = np.dot(Zn, Baux)

            for ii in range(n_samples_cat):
                # logsumC += np.log(phi(x=u[ii] + aux))
                logsumC[ii] += np.log(phi(x=u[ii] + aux))

    #
    # this shall be a logsumexp
    # return logsumC - np.log(n_samples_cat)
    # return logsumexp(logsumC) - np.log(n_samples_cat)
    return logsumexp_1D_numba(logsumC) - np.log(n_samples_cat)


from math import erf, sqrt


@numba.njit
def phi_erf_numba(x):
    return (1.0 + erf(x / sqrt(2.0))) / 2.0


@numba.jit(float64(int64, int64,
                   optional(float64[:, :, :]),
                   float64[:], float64[:],
                   int64, uint8, float64[:]), nopython=True)
def cat_pdf_numba_prealloc(d, r, Bcat, Zn, u,  n_samples_cat, R_d, logsumC):
    # logsumC = float64(0)
    # logsumC = np.zeros(n_samples_cat, dtype=np.float64)
    logsumC.fill(0)
    for r2 in range(R_d):
        if r2 != r:
            Baux = np.copy(Bcat[d, r, :])
            Baux -= Bcat[d, r2, :]
            aux = np.dot(Zn, Baux)

            for ii in range(n_samples_cat):
                # logsumC += np.log(phi(x=u[ii] + aux))
                logsumC[ii] += np.log(phi_erf_numba(x=u[ii] + aux))

    #
    # this shall be a logsumexp
    # return logsumC - np.log(n_samples_cat)
    # return logsumexp(logsumC) - np.log(n_samples_cat)
    return logsumexp_1D_numba(logsumC) - np.log(n_samples_cat)


@numba.jit(float64[:, :, :](float64[:, :], float64[:, :, :],
                            uint8[:], uint8[:], float64[:],
                            float64[:, :], float64[:, :], int64,
                            optional(float64[:, :]), optional(
                                float64[:, :]), optional(float64[:, :]),
                            optional(float64[:, :]), optional(float64[:, :, :]),
                            optional(float64[:, :]), optional(float64[:, :]),
                            float64, float64, float64, int64,
                            float64[:], float64[:], float64[:],
                            float64[:, :], float64[:], float64[:],
                            int64[:],
                            int64[:],
                            boolean[:],
                            boolean[:],
                            int64[:],
                            int64), nopython=True)
def compute_lvstd_leaf_ll_samp_Z(X, LL,
                                 C, R, u,
                                 Z, Z_sampled, Kest,
                                 Breal, Bpos, Bint, Bbin, Bcat, Bord, Bcount,
                                 s2Y, sYd, s2U, Wint,
                                 maxX, meanX, minX,
                                 theta, theta_L, theta_H,
                                 instance_ids,
                                 feature_ids,
                                 # scope,
                                 instance_scope,
                                 feature_scope,
                                 feature_scope_map,
                                 n_samples_cat
                                 ):
    N = len(instance_ids)
    #
    # LL is of shape D x N x L
    L = LL.shape[2]

    for n, d in zip(instance_ids, feature_ids):

        if not feature_scope[d]:
            continue

        xnd = X[n, d]

        dn = feature_scope_map[d]

        Zn = Z[:, n]

        if C[dn] == 1:
            #
            # real
            aux = np.dot(Zn, Breal[dn])
            ll = xpdf_re(xnd, 2 / (maxX[d] - meanX[d]),
                         meanX[d], aux, s2Y, s2U)
            LL[d, n, 0] = ll

            #
            # int
            aux = np.dot(Zn, Bint[dn])
            ll = xpdf_int(xnd, Wint, theta_L[d], theta_H[d], aux, s2Y, s2U)
            LL[d, n, 1] = ll

            #
            # pos
            aux = np.dot(Zn, Bpos[dn])
            ll = xpdf_pos(xnd,  2 / maxX[d], aux, s2Y, s2U)
            LL[d, n, 2] = ll

        elif C[dn] == 2:

            #
            # real
            aux = np.dot(Zn, Breal[dn])
            ll = xpdf_re(xnd, 2 / (maxX[d] - meanX[d]),
                         meanX[d], aux, s2Y, s2U)
            LL[d, n, 0] = ll

            #
            # int
            aux = np.dot(Zn, Bint[dn])
            ll = xpdf_int(xnd, Wint, theta_L[d], theta_H[d], aux, s2Y, s2U)
            LL[d, n, 1] = ll

        elif C[dn] == 3:
            continue

        elif C[dn] == 4:

            xnd = int(xnd)
            r = int64(xnd - 1)
            rr = int64(xnd - 2)

            #
            # cat

            ll = cat_pdf_numba(dn, r, Bcat, Zn, u, n_samples_cat,  R[dn])
            LL[d, n, 3] = ll

            #
            # ord
            aux = np.dot(Zn, Bord[dn])

            if xnd == 1:
                ll = np.log(phi(x=theta[d, r] - aux))
            elif xnd == R[dn]:
                ll = np.log(1 - phi(x=theta[d, rr] - aux))
            else:
                ll = np.log(phi(theta[d, r] - aux) -
                            phi(x=theta[d, rr] - aux))

            LL[d, n, 4] = ll

            #
            # count
            aux = np.dot(Zn, Bcount[dn])
            ll = np.log(phi(x=f_1(xnd + 1, 2 / maxX[d]) - aux) -
                        phi(x=f_1(xnd, 2 / maxX[d]) - aux))
            LL[d, n, 5] = ll

        if np.any(np.isinf(LL[d, n, :])):
            print('NNINF', LL[d, n, :])

    return LL


def compute_lvstd_leaf_ll_sss(X, LL,
                              C, R, u,
                              Z, Z_sampled, Kest,
                              Breal, Bpos, Bint, Bbin, Bcat, Bord, Bcount,
                              s2Y, sYd, s2U, Wint,
                              maxX, meanX, minX,
                              theta, theta_L, theta_H,
                              instance_ids,
                              feature_ids,
                              # scope,
                              instance_scope,
                              feature_scope,
                              feature_scope_map,
                              n_samples_cat
                              ):

    N = len(instance_ids)
    #
    # LL is of shape D x N x L
    L = LL.shape[2]
    S = len(Z_sampled)

    for n, d in zip(instance_ids, feature_ids):
        # for n in instance_ids:
        #     for dn, d in enumerate(feature_ids):

            # for dn, n, d in enumerate(zip(instance_ids, feature_ids)):

        # #
        # # if not in scope, we have to marginalize over it?
        if feature_scope[d]:

            h = None

            xnd = X[n, d]

            #
            # D x N x L x S
            # LL_ZS = np.zeros((L, S))
            # LL_ZS.fill(-np.inf)

            LL_real = []
            LL_int = []
            LL_pos = []
            LL_cat = []
            LL_ord = []
            LL_count = []

            Baux = np.zeros(Kest)

            # dn = get_feature_id(d)
            dn = feature_scope_map[d]

            LL[d, n, :] = 0
            # LL[dn, n, :] = 0

            Zs = None
            # if n in instance_scope:
            if instance_scope[n]:
                Zs = np.zeros((1, Kest))
                Zs[0, :] = Z[:, n]

                # Zs = Zs.reshape(1, Zs.shape[0])
            else:
                # Zs = Z_sampled
                Zs = np.zeros((1, Kest))
                Zs[0, :] = Z[:, n]

            len_Z = Zs.shape[0]

            for j in range(len_Z):
                Zn = Zs[j]

                if C[dn] == 1:
                    #
                    # real
                    aux = np.dot(Zn, Breal[dn])
                    ll = xpdf_re(xnd, 2 / (maxX[d] - meanX[d]),
                                 meanX[d], aux, s2Y, s2U)
                    # LL[dn, n, 0] += ll
                    # LL_ZS[0, j] = ll
                    LL_real.append(ll)
                    # if (np.isinf(ll)):
                    #     print('C1 real', dn, j, n, ll)

                    #
                    # int
                    aux = np.dot(Zn, Bint[dn])
                    ll = xpdf_int(xnd, Wint, theta_L[d], theta_H[d], aux, s2Y, s2U)
                    # LL[dn, n, 1] += ll
                    # LL_ZS[1, j] = ll
                    LL_int.append(ll)
                    # if (np.isinf(ll)):
                    #     print('C1 int', dn, j, n, ll)

                    #
                    # pos
                    aux = np.dot(Zn, Bpos[dn])
                    ll = xpdf_pos(xnd,  2 / maxX[d], aux, s2Y, s2U)
                    # LL[dn, n, 2] += ll
                    # LL_ZS[2, j] = ll
                    LL_pos.append(ll)
                    # if (np.isinf(ll)):
                    #     print('C1 pos', dn, j, n, ll)

                    # if S[n, d] == 1:
                    #     h = 0
                    # elif S[n, d] == 2:
                    #     h = 1
                    # elif S[n, d] == 3:
                    #     raise NotImplementedError('Circuluar data are not supposed to be used')

                    # elif S[n, d] == 4:
                    #     h = 2

                elif C[dn] == 2:

                    #
                    # real
                    aux = np.dot(Zn, Breal[dn])
                    ll = xpdf_re(xnd, 2 / (maxX[d] - meanX[d]),
                                 meanX[d], aux, s2Y, s2U)
                    # LL[dn, n, 0] += ll
                    # LL_ZS[0, j] = ll
                    LL_real.append(ll)
                    # if (np.isinf(ll)):
                    #     print('C2 real', dn, j, n, ll)

                    #
                    # int
                    aux = np.dot(Zn, Bint[dn])
                    ll = xpdf_int(xnd, Wint, theta_L[d], theta_H[d], aux, s2Y, s2U)
                    # LL[dn, n, 1] += ll
                    # LL_ZS[1, j] = ll
                    LL_int.append(ll)
                    # if (np.isinf(ll)):
                    #     print('C2 int', dn, j, n, ll)

                    # if S[n, d] == 1:
                    #     h = 0
                    # elif S[n, d] == 2:
                    #     h = 1
                    # elif S[n, d] == 3:
                    #     raise NotImplementedError('Circuluar data are not supposed to be used')

                    # elif S[n, d] == 4:
                    #     raise NotImplementedError('This feature is expected not to be positive')

                elif C[dn] == 3:
                    continue

                elif C[dn] == 4:

                    xnd = int(xnd)
                    r = int(xnd - 1)
                    rr = int(xnd - 2)

                    #
                    # cat

                    # prodC = np.ones(n_samples_cat)
                    # u = rand_gen.normal(loc=0, scale=sYd, size=n_samples_cat)

                    # for r2 in range(R[dn]):
                    #     if r2 != r:
                    #         Baux = np.copy(Bcat[dn, r])
                    #         Baux -= Bcat[dn, r2]
                    #         aux = np.dot(Zn, Baux)

                    #         for ii in range(n_samples_cat):
                    #             # prodC[ii] = prodC[ii] * \
                    #             #     stats.norm.cdf(x=u[ii] + aux, scale=1)
                    #             prodC[ii] = prodC[ii] * \
                    #                 phi(x=u[ii] + aux)

                    # sumC = prodC.sum()
                    # LL[d, n, 3] += np.log(sumC / n_samples_cat)
                    ll = cat_pdf_numba(dn, r, Bcat,
                                       Zn, u,  n_samples_cat, R[dn])
                    # LL_ZS[3, j] = ll
                    LL_cat.append(ll)
                    # LL[dn, n, 3] += ll

                    # if (np.isinf(ll)):
                    #     print('C4 cat', dn, j, n, ll)

                    #
                    # ord
                    aux = np.dot(Zn, Bord[dn])

                    if xnd == 1:
                        ll = np.log(phi(x=theta[d, r] - aux))
                        # if (np.isinf(ll)):
                        #     print('C4 ord 1', dn, j, n, ll)

                    elif xnd == R[dn]:
                        ll = np.log(1 - phi(x=theta[d, rr] - aux))
                        # if (np.isinf(ll)):
                        #     print('C4 ord 2', dn, j, n, ll)

                    else:
                        ll = np.log(phi(theta[d, r] - aux) -
                                    phi(x=theta[d, rr] - aux))
                        # if (np.isinf(ll)):
                        #     print('C4 ord 3', dn, j, n, ll)

                    # LL[dn, n, 4] += ll
                    # LL_ZS[4, j] = ll
                    LL_ord.append(ll)

                    #
                    # count
                    aux = np.dot(Zn, Bcount[dn])
                    ll = np.log(phi(x=f_1(xnd + 1, 2 / maxX[d]) - aux) -
                                phi(x=f_1(xnd, 2 / maxX[d]) - aux))
                    # LL[dn, n, 5] += ll
                    # LL_ZS[5, j] = ll
                    LL_count.append(ll)

                    # if (np.isinf(ll)):
                    #     print('C4 count', dn, j, n, ll)

                    # if S[n, d] == 1:
                    #     h = 3

                    # elif S[n, d] == 2:
                    #     h = 4

                    # elif S[n, d] == 3:
                    #     h = 5

            #
            # averaging
            # LL[dn, n, :] = LL[dn, n, :] / len_Z
            # LL[d, n, :] = LL[d, n, :] / len_Z

            #
            # making a mixture
            # print(d, n, LL[d, n, :], LL_ZS, np.log(len_Z), len_Z)
            # LL[d, n, :] = logsumexp(LL_ZS, axis=-1) - np.log(len_Z)

            # if LL_real:
            #     LL[d, n, 0] = logsumexp(LL_real) - np.log(len_Z)
            # if LL_int:
            #     LL[d, n, 1] = logsumexp(LL_int) - np.log(len_Z)
            # if LL_pos:
            #     LL[d, n, 2] = logsumexp(LL_pos) - np.log(len_Z)
            # if LL_cat:
            #     LL[d, n, 3] = logsumexp(LL_cat) - np.log(len_Z)
            # if LL_ord:
            #     LL[d, n, 4] = logsumexp(LL_ord) - np.log(len_Z)
            # if LL_count:
            #     LL[d, n, 5] = logsumexp(LL_count) - np.log(len_Z)

            if LL_real:
                # lse = logsumexp_1D_numba(np.array(LL_real))
                # lses = logsumexp(np.array(LL_real))
                # LL[d, n, 0] = lse - np.log(len_Z)
                # assert np.isclose(lse, lses), 'lse {}, lses {}'.format(lse, lses, )
                LL[d, n, 0] = logsumexp_1D_numba(np.array(LL_real)) - np.log(len_Z)
            if LL_int:
                LL[d, n, 1] = logsumexp_1D_numba(np.array(LL_int)) - np.log(len_Z)
            if LL_pos:
                LL[d, n, 2] = logsumexp_1D_numba(np.array(LL_pos)) - np.log(len_Z)
            if LL_cat:
                LL[d, n, 3] = logsumexp_1D_numba(np.array(LL_cat)) - np.log(len_Z)
            if LL_ord:
                LL[d, n, 4] = logsumexp_1D_numba(np.array(LL_ord)) - np.log(len_Z)
            if LL_count:
                LL[d, n, 5] = logsumexp_1D_numba(np.array(LL_count)) - np.log(len_Z)

            if np.any(np.isinf(LL[d, n, :])):
                print('NNINF', LL[d, n, :], len_Z)

            # if LL_real:
            #     LL[dn, n, 0] = logsumexp(LL_real) - np.log(len_Z)
            # if LL_int:
            #     LL[dn, n, 1] = logsumexp(LL_int) - np.log(len_Z)
            # if LL_pos:
            #     LL[dn, n, 2] = logsumexp(LL_pos) - np.log(len_Z)
            # if LL_cat:
            #     LL[dn, n, 3] = logsumexp(LL_cat) - np.log(len_Z)
            # if LL_ord:
            #     LL[dn, n, 4] = logsumexp(LL_ord) - np.log(len_Z)
            # if LL_count:
            #     LL[dn, n, 5] = logsumexp(LL_count) - np.log(len_Z)

            # if np.any(np.isinf(LL[dn, n, :])):
            #     print('NNINF', LL[dn, n, :])

    return LL


@numba.jit(float64[:, :, :](float64[:, :], float64[:, :, :],
                            uint8[:], int64[:], float64[:],
                            float64[:, :],
                            # float64[:, :],
                            int64,
                            optional(float64[:, :]),
                            optional(float64[:, :]),
                            optional(float64[:, :]),
                            optional(float64[:, :]), optional(float64[:, :, :]),
                            optional(float64[:, :]), optional(float64[:, :]),
                            float64, float64, float64, int64,
                            float64[:], float64[:], float64[:],
                            float64[:, :], float64[:], float64[:],
                            int64[:],
                            int64[:],
                            boolean[:],
                            boolean[:],
                            int64[:],
                            int64), nopython=True)
def compute_lvstd_leaf_ll(X, LL,
                          C, R, u,
                          Z,
                          # Z_sampled,
                          Kest,
                          Breal, Bpos, Bint, Bbin, Bcat, Bord, Bcount,
                          s2Y, sYd, s2U, Wint,
                          maxX, meanX, minX,
                          theta, theta_L, theta_H,
                          instance_ids,
                          feature_ids,
                          # scope,
                          instance_scope,
                          feature_scope,
                          feature_scope_map,
                          n_samples_cat
                          ):

    # N = len(instance_ids)
    #
    # LL is of shape D x N x L
    # L = LL.shape[2]

    for n, d in zip(instance_ids, feature_ids):
        # for n in instance_ids:
        #     for dn, d in enumerate(feature_ids):

            # for dn, n, d in enumerate(zip(instance_ids, feature_ids)):

        # #
        # # if not in scope, we have to marginalize over it?
        if feature_scope[d]:

            xnd = X[n, d]

            # dn = get_feature_id(d)
            dn = feature_scope_map[d]

            LL[d, n, :] = 0
            # LL[dn, n, :] = 0

            Zn = Z[:, n]

            if C[dn] == 1:
                #
                # real
                aux = np.dot(Zn, Breal[dn])
                ll = xpdf_re(xnd, 2 / (maxX[d] - meanX[d]),
                             meanX[d], aux, s2Y, s2U)
                LL[d, n, 0] = ll

                #
                # int
                aux = np.dot(Zn, Bint[dn])
                ll = xpdf_int(xnd, Wint, theta_L[d], theta_H[d], aux, s2Y, s2U)
                LL[d, n, 1] = ll

                #
                # pos
                aux = np.dot(Zn, Bpos[dn])
                ll = xpdf_pos(xnd,  2 / maxX[d], aux, s2Y, s2U)
                LL[d, n, 2] = ll

            elif C[dn] == 2:

                #
                # real
                aux = np.dot(Zn, Breal[dn])
                ll = xpdf_re(xnd, 2 / (maxX[d] - meanX[d]),
                             meanX[d], aux, s2Y, s2U)
                LL[d, n, 0] = ll

                #
                # int
                aux = np.dot(Zn, Bint[dn])
                ll = xpdf_int(xnd, Wint, theta_L[d], theta_H[d], aux, s2Y, s2U)
                LL[d, n, 1] = ll

            elif C[dn] == 3:
                continue

            elif C[dn] == 4:

                xnd = int(xnd)
                r = int(xnd - 1)
                rr = int(xnd - 2)

                #
                # cat

                ll = cat_pdf_numba(dn, r, Bcat,
                                   Zn, u,  n_samples_cat, R[dn])
                LL[d, n, 3] = ll

                # if (np.isinf(ll)):
                #     print('C4 cat', dn, j, n, ll)

                #
                # ord
                aux = np.dot(Zn, Bord[dn])

                if xnd == 1:
                    ll = np.log(phi(x=theta[d, r] - aux))
                    # if (np.isinf(ll)):
                    #     print('C4 ord 1', dn, j, n, ll)

                elif xnd == R[dn]:
                    ll = np.log(1 - phi(x=theta[d, rr] - aux))
                    # if (np.isinf(ll)):
                    #     print('C4 ord 2', dn, j, n, ll)

                else:
                    ll = np.log(phi(theta[d, r] - aux) -
                                phi(x=theta[d, rr] - aux))
                    # if (np.isinf(ll)):
                    #     print('C4 ord 3', dn, j, n, ll)

                LL[d, n, 4] = ll

                #
                # count
                aux = np.dot(Zn, Bcount[dn])
                ll = np.log(phi(x=f_1(xnd + 1, 2 / maxX[d]) - aux) -
                            phi(x=f_1(xnd, 2 / maxX[d]) - aux))
                LL[d, n, 5] = ll

                # if (np.isinf(ll)):
                #     print('C4 count', dn, j, n, ll)

                # if S[n, d] == 1:
                #     h = 3

                # elif S[n, d] == 2:
                #     h = 4

                # elif S[n, d] == 3:
                #     h = 5

            if np.any(np.isinf(LL[d, n, :])):
                print('NNINF', LL[d, n, :])

    return LL


@numba.jit(nopython=False)
def running_avg_numba(a, b):
    D = a.shape[-1]
    avg_a = np.zeros(a.shape)
    for d in range(D):
        a_s = a[..., np.maximum(d - b, 0):d + 1]
        #  print(a_s.shape)
        # if d < 10:
        #     print(a_s[:10], np.mean(a_s, axis=-1)[:10])
        avg_a[..., d] = np.mean(a_s, axis=-1)
    return avg_a


@numba.jit(Tuple((float64[:, :, :], int64, int64))(
    #   D   N      W               X
    int64, int64, float64[:, :], float64[:, :],
    #   C           R    kest    Z
    uint8[:], int64[:], int64, float64[:, :, :],
    #   Breal           Bint                Bpos                Bcat                Bord
    float64[:, :, :], float64[:, :, :], float64[:, :, :], float64[:, :, :, :], float64[:, :, :],
    #   Bcount          u       wint    theta           thetal      thetah
    float64[:, :, :], float64[:, :,
                              :], int64, float64[:, :], float64[:], float64[:],
    #   maxX        meanX    s2y      s2u      syd      n_samples
    float64[:], float64[:], float64, float64, float64, int64), nopython=True)
def sample_S_W(D, N, W, X, C, R, Kest, Z, Breal, Bint, Bpos, Bcat, Bord, Bcount, u, Wint, theta, theta_L, theta_H, maxX, meanX, s2Y, s2U, sYd, n_samples_cat):
    pparam = np.zeros((N, D, 4), dtype=np.float64)
    countErrC = 0
    countErr = 0

    for d in range(D):
        SB = None
        muB = None

        if C[d] == 1:
            for n in range(N):

                xnd = X[n, d]

                Baux = np.zeros(Kest, dtype=np.float64)
                p = np.zeros(4, dtype=np.float64)

                if xnd == -1:
                    p[:] = W[d, :]
                    assert False, "we shouldn't be here"
                else:

                    Zn = Z[:, n, d]

                    p[0] = 0
                    aux = np.dot(Zn, Breal[n, d, :])

                    p[0] += xpdf_re(xnd, 2 / (maxX[d] - meanX[d]), meanX[d], aux, s2Y, s2U)
                    p[0] += np.log(W[d, 0])

                    p[1] = 0
                    aux = np.dot(Zn, Bint[n, d, :])
                    p[1] += xpdf_int(xnd, Wint, theta_L[d], theta_H[d], aux, s2Y, s2U)
                    p[1] += np.log(W[d, 1])

                    p[2] = -np.inf

                    p[3] = 0
                    aux = np.dot(Zn, Bpos[n, d, :])
                    p[3] += xpdf_pos(xnd, 2 / maxX[d], aux, s2Y, s2U)
                    p[3] += np.log(W[d, 3])

                    p_n = np.exp(p)
                    psum = p_n.sum()

                    p[:] = p_n / psum

                if not np.isnan(p.sum()):
                    pparam[n, d, :] = p
                else:
                    countErrC += 1
                    p = W[d, :]
                    pparam[n, d, :] = p

        elif C[d] == 2:
            for n in range(N):

                xnd = X[n, d]

                Baux = np.zeros(Kest, dtype=np.float64)
                p = np.zeros(4, dtype=np.float64)

                if xnd == -1:
                    p[:] = W[d, :]
                    assert False, "we shouldn't be here"
                else:

                    Zn = Z[:, n, d]

                    p[0] = 0
                    aux = np.dot(Zn, Breal[n, d, :])
                    p[0] += xpdf_re(xnd, 2 / (maxX[d] - meanX[d]), meanX[d], aux, s2Y, s2U)
                    p[0] += np.log(W[d, 0])

                    p[1] = 0
                    aux = np.dot(Zn, Bint[n, d, :])
                    p[1] += xpdf_int(xnd, Wint, theta_L[d], theta_H[d], aux, s2Y, s2U)
                    p[1] += np.log(W[d, 1])

                    #
                    # no dir (only in Isabel's code) and no pos real
                    p[2] = -np.inf

                    p[3] = -np.inf

                    p_n = np.exp(p)
                    psum = p_n.sum()

                    p = p_n / psum

                if not np.isnan(p.sum()):
                    pparam[n, d, :] = p

                else:
                    countErrC += 1
                    p = W[d, :]
                    pparam[n, d, :] = p

        #
        # binary types are fixed!
        elif C[d] == 3:
            pass

        elif C[d] == 4:
            # paramW = np.copy(alpha_W[d, :])

            for n in range(N):

                xnd = X[n, d]

                Baux = np.zeros(Kest, dtype=np.float64)
                p = np.zeros(4, dtype=np.float64)

                if xnd == -1:
                    p[:] = W[d, :]
                    assert False, "we shouldn't be here"
                else:
                    Zn = Z[:, n, d]

                    #
                    # categorical
                    p[0] = 0

                    r = int(xnd) - 1

                    p[0] = cat_pdf_numba(d, r, Bcat[n, :, :, :], Zn,
                                         u[n, d, :], n_samples_cat, R[d])

                    p[0] += np.log(W[d, 0])

                    # print('p[0]', p, sumC, W[d, 0])

                    #
                    # ordinal
                    p[1] = 0
                    aux = np.dot(Zn, Bord[n, d, :])
                    xr = int(xnd) - 1
                    xrr = int(xnd) - 2

                    # p[1] = ord_log_likelihood(xnd, R, theta, d, aux)
                    if xnd == 1:
                        # p[1] = np.log(stats.norm.cdf(x=theta[d, xr] - aux, scale=1))
                        p[1] = np.log(phi(x=theta[d, xr] - aux))
                        # print('log', stats.norm.cdf(x=theta[d, xr] - aux, scale=1))
                        if np.isnan(p[1]) or np.isinf(p[1]):
                            print('x {} R[d] {} p[1] {} logW {} x=1 {}',
                                  xnd, R[d], p[1],
                                  np.log(W[d, 1]),
                                  np.log(phi(x=theta[d, xr] - aux)))
                    elif xnd == R[d]:
                        # p[1] = np.log(1 - stats.norm.cdf(x=theta[d, xrr] - aux, scale=1))
                        p[1] = np.log(1 - phi(x=theta[d, xrr] - aux))
                        if 1 - phi(x=theta[d, xrr] - aux) <= 0:
                            print('llog', 1 - phi(x=theta[d, xrr] - aux), theta[d, xrr], aux)
                        if np.isnan(p[1]) or np.isinf(p[1]):
                            print('x {} R[d] {} p[1] {} logW {} x=Rd {}',
                                  xnd, R[d], p[1],
                                  np.log(W[d, 1]),
                                  np.log(1 - phi(x=theta[d, xrr] - aux))
                                  )

                    else:
                        # p[1] = np.log(stats.norm.cdf(x=theta[d, xr] - aux, scale=1) -
                        #               stats.norm.cdf(x=theta[d, xrr] - aux, scale=1))
                        p[1] = np.log(phi(x=theta[d, xr] - aux) -
                                      phi(x=theta[d, xrr] - aux))
                        if (phi(x=theta[d, xr] - aux) -
                                phi(x=theta[d, xrr] - aux)) <= 0:
                            print('lllog', phi(x=theta[d, xr] - aux),
                                  phi(x=theta[d, xrr] - aux))  # -
                        if np.isnan(p[1]) or np.isinf(p[1]):
                            print(
                                'x {} R[d] {} p[1] {} logW {} x= {}, phi1 {} phi2 {} taux1 {} taux2 {} t1 {} t2{}',
                                xnd, R[d], p[1],
                                np.log(W[d, 1]),
                                np.log(phi(x=theta[d, xr] - aux) -
                                       phi(x=theta[d, xrr] - aux)),
                                phi(x=theta[d, xr] - aux),
                                phi(x=theta[d, xrr] - aux),
                                theta[d, xr] - aux,
                                theta[d, xrr] - aux,
                                theta[d, xr],
                                theta[d, xrr]
                            )

                            # stats.norm.cdf(x=theta[d, xrr] - aux, scale=1), 'thetas', theta[d, xr], theta[d, xrr], stats.norm.cdf(x=theta[d, xr] - aux, scale=1),  stats.norm.cdf(x=theta[d, xrr] - aux, scale=1))

                    p[1] += np.log(W[d, 1])

                    # count
                    p[2] = 0
                    aux = np.dot(Zn, Bcount[n, d, :])
                    p[2] = np.log(phi(x=f_1(xnd + 1, 2 / maxX[d]) - aux) -
                                  phi(x=f_1(xnd, 2 / maxX[d]) - aux))

                    p[2] += np.log(W[d, 2])

                    p[3] = -np.inf

                    pmax = np.max(p)

                    if np.isinf(pmax):
                        p_n = np.exp(p)
                    else:
                        p_n = np.exp(p - pmax)

                    psum = p_n.sum()

                    _p = p_n / psum

                if not np.isnan(_p.sum()):
                    pparam[n, d, :] = _p
                else:
                    countErr += 1
                    print("Discrete: n={}, d={}, ", n, d)
                    print("sum_p= {}: p_n: {} psum {} pmax {} p {}", p.sum(), p_n, psum, pmax, p)
                    pparam[n, d, :] = W[d, :]

    return (pparam, countErrC, countErr)
