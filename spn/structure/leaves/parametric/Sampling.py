'''
Created on April 15, 2018

@author: Alejandro Molina
@author: Antonio Vergari
'''

from spn.structure.leaves.parametric.Parametric import Parametric, Gaussian, Gamma, Poisson, Categorical, LogNormal, \
    Geometric, Exponential, Bernoulli

from spn.structure.leaves.parametric.pp import Beta, Laplace, Wald, Weibull, Gumbel
import numpy as np

from spn.structure.leaves.parametric.utils import get_scipy_obj_params, get_scipy_obj_params_from_str


def sample_parametric_node(node, n_samples, rand_gen):
    assert isinstance(node, Parametric)
    assert n_samples > 0

    X = None
    if isinstance(node, Gaussian) or isinstance(node, Gamma) or isinstance(node, LogNormal) or \
            isinstance(node, Poisson) or isinstance(node, Geometric) or isinstance(node, Exponential) or\
            isinstance(node, Bernoulli) or isinstance(node, Beta) or isinstance(node, Weibull) or\
            isinstance(node, Laplace) or isinstance(node, Wald) or isinstance(node, Gumbel):

        scipy_obj, params = get_scipy_obj_params(node)

        X = scipy_obj.rvs(size=n_samples, random_state=rand_gen, **params)

    elif isinstance(node, Categorical):
        X = rand_gen.choice(np.arange(node.k), p=node.p, size=n_samples)

    else:
        raise Exception('Node type unknown: ' + str(type(node)))

    return X


SUPPORTED_DIST_SAMPLING = set(['Gaussian', 'Gamma', 'LogNormal', 'Poisson',
                               'Geometric', 'Exponential', 'Bernoulli', 'Beta', 'Weibull',
                               'Laplace', 'Wald', 'Gumbel'])


def sample_parametric_dist(dist_name, dist_params,  n_samples, rand_gen):

    assert n_samples > 0

    X = None
    if dist_name in SUPPORTED_DIST_SAMPLING:
       # isinstance(node, Gaussian) or isinstance(node, Gamma) or isinstance(node, LogNormal) or \
       #      isinstance(node, Poisson) or isinstance(node, Geometric) or isinstance(node, Exponential) or\
       #      isinstance(node, Bernoulli) or isinstance(node, Beta) or isinstance(node, Weibull) or\
       #      isinstance(node, Laplace) or isinstance(node, Wald) or isinstance(node, Gumbel):

        scipy_obj, scipy_params = get_scipy_obj_params_from_str(dist_name, dist_params)

        X = scipy_obj.rvs(size=n_samples, random_state=rand_gen, **scipy_params)

    elif dist_name == 'Categorical':
        X = rand_gen.choice(np.arange(dist_params['k']), p=dist_params['p'], size=n_samples)

    else:
        raise Exception('Unknown distribution: ' + dist_name)

    return X
