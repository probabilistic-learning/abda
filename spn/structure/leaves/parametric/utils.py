'''
Created on April 29, 2018

@author: Alejandro Molina
'''
import numpy as np
from scipy.stats import *

from spn.structure.leaves.parametric.Parametric import *
from spn.structure.leaves.parametric.pp import *


def get_scipy_obj_params(node):
    if isinstance(node, Gaussian):
        return norm, {"loc": node.mean, "scale": node.stdev}

    elif isinstance(node, Gamma):
        return gamma, {"a": node.alpha, "scale": 1.0 / node.beta}

    elif isinstance(node, LogNormal):
        return lognorm, {"scale": np.exp(node.mean), "s": node.stdev}

    elif isinstance(node, Poisson):
        return poisson, {"mu": node.mean}

    elif isinstance(node, Geometric):
        return geom, {"p": node.p}

    elif isinstance(node, Beta):
        return beta, {"a": node.alpha, "b": node.beta}

    elif isinstance(node, Gumbel):
        return gumbel_r, {"loc": node.mu, "scale": node.beta}

    elif isinstance(node, Laplace):
        return laplace, {"loc": node.mu, "scale": node.b}

    elif isinstance(node, Wald):
        return invgauss, {"mu": node.mu / node.lam, "scale": node.lam}

    elif isinstance(node, Weibull):
        return weibull_min, {"c": node.alpha, "scale": node.beta}

    elif isinstance(node, Exponential):
        return expon, {"scale": 1 / node.l}

    elif isinstance(node, Bernoulli):
        return bernoulli, {"p": node.p}

    else:
        raise Exception("unknown node type %s " % type(node))


def get_scipy_obj_params_from_str(dist_name, dist_params):
    if dist_name == 'Gaussian':
        return norm, {"loc": dist_params['mean'], "scale": dist_params['stdev']}

    elif dist_name == 'Gamma':
        return gamma, {"a": dist_params['alpha'], "scale": 1.0 / dist_params['beta']}

    elif dist_name == 'LogNormal':
        return lognorm, {"scale": np.exp(dist_params['mean']),
                         "s": dist_params['stdev']}

    elif dist_name == 'Poisson':
        return poisson, {"mu": dist_params['mean']}

    elif dist_name == 'Geometric':
        return geom, {"p": dist_params['p']}

    elif dist_name == 'Beta':
        return beta, {"a": dist_params['alpha'], "b": dist_params['beta']}

    elif dist_name == 'Gumbel':
        return gumbel_r, {"loc": dist_params['mu'], "scale": dist_params['beta']}

    elif dist_name == 'Laplace':
        return laplace, {"loc": dist_params['mu'], "scale": dist_params['b']}

    elif dist_name == 'Wald':
        return invgauss, {"mu": dist_params['mu'] / dist_params['lam'], "scale": dist_params['lam']}

    elif dist_name == 'Weibull':
        return weibull_min, {"c": dist_params['alpha'], "scale": dist_params['beta']}

    elif dist_name == 'Exponential':
        return expon, {"scale": 1 / dist_params['l']}

    elif dist_name == 'Bernoulli':
        return bernoulli, {"p": dist_params['p']}

    else:
        raise Exception("unknown distribution %s " % type(dist_name))
