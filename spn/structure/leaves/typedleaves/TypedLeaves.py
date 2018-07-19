'''
Created on May 04, 2018

@author: Alejandro Molina
'''
from copy import deepcopy


from spn.structure.Base import Sum, get_nodes_by_type
from spn.structure.leaves.parametric.Parametric import *
from spn.structure.leaves.parametric.pp import *


class TypeLeaf(Sum):
    """
    Super class for TypeMixture and TypeMixtureUnconstrained
    """

    def __init__(self, meta_type, scope):
        Sum.__init__(self)
        self.meta_type = meta_type
        self.scope = scope


class TypeMixture(TypeLeaf):
    """
    Implements a mixture over certain data types for a univariate feature d
    in which mixture proportions (weights) are globally shared for all features
    of the same type.

    It has associated:
        - a list of Parametric Leaves (same number as the mixture components)
        - a shared copy of the global weights w^{d}
        - the meta-type for the considered feature (redundant)

    It is employed in the Model-HA1 and Model-HA2
    """

    def __init__(self, meta_type, scope):
        TypeLeaf.__init__(self, meta_type, scope)
        # self.meta_type = meta_type
        # self.scope = scope

    def add_parametric_leaf(self, l):
        assert self.meta_type == l.type.meta_type, 'meta-type:{} leaf meta-type{}'.format(self.meta_type,
                                                                                          l.type.meta_type)
        #
        # not adding a weight!
        self.children.append(l)


class TypeMixtureUnconstrained(TypeLeaf):
    """
    Implements a mixture over certain data types for a univariate feature d
    in which mixture proportions (weights) are NOT shared  for all features
    of the same type.

    NOTE: Essentially, it is just a Sum node, the class is introduced to distinguish between the node types
    at runtime

    It has associated:
        - a list of Parametric Leaves (same number as the mixture components)
        - a LOCAL copy of the global weights w^{d}
        - the meta-type for the considered feature (redundant)

    """

    def __init__(self, meta_type, scope):
        TypeLeaf.__init__(self, meta_type, scope)
        # self.meta_type = meta_type
        # self.scope = scope

    def add_parametric_leaf(self, l, w):
        assert self.meta_type == l.type.meta_type

        self.children.append(l)
        self.weights.append(w)


class ParametricMixture(Sum):
    """
    Implements a mixture over certain parametric forms a particular type
    It is employed in the Model-HB1
    """

    def __init__(self, type, meta_type, scope):
        Sum.__init__(self)
        self.meta_type = meta_type
        self.type = type
        self.scope = scope

    def add_parametric_leaf(self, l, w):
        assert self.meta_type == l.type.meta_type

        self.children.append(l)
        self.weights.append(w)


PARAM_FORM_TYPE_MAP = {
    Gaussian: Type.REAL,
    Gamma: Type.POSITIVE,
    Exponential: Type.POSITIVE,
    LogNormal: Type.POSITIVE,
    Categorical: Type.CATEGORICAL,
    Geometric: Type.COUNT,
    Poisson: Type.COUNT,
    Bernoulli: Type.BINARY,
    NegativeBinomial: Type.COUNT,
    Hypergeometric: Type.COUNT
}


LEAF_TYPES = {'tmw', 'tm', 'pmw', 'pm', 'tm-pmw', 'tm-pm'}


def type_mixture_leaf_factory(leaf_type,
                              leaf_meta_type,
                              type_to_param_map,
                              scope,
                              init_weights=None):
    """
    Factory method to create a type mixture leaf (aka fat leaf)
    according to one specified model (leaf_type):

    - 'tmw': one mixture over types with SHARED weights (TypeMixture node) and one parametric form per type
    - 'tm': one mixture over types with FREE local weights (TypeMixtureUnconstrained node) and one parametric form per type
    - 'pmw': one mixture over parametric forms with SHARED weights (TypeMixture node)
    - 'pm': one mixture over parametric forms with FREE local weights (TypeMixtureUnconstrained node)
    - 'tm-pmw': one mixture over types with SHARED weights (TypeMixture node) and more than one parametric form per type (one ParamtricMixture node per type)
    - 'tm-pm': one mixture over types with FREE local weights (TypeMixtureUnconstrained node) and more than one parametric form per type (one ParamtricMixture node per type)

    Parametric forms associated to each type are specified in the type_to_param_map:

    {type: {parametric_node: {'params':parameters, 'prior':prior}}}

    NOTE: the order in which leaves are created depends on the order in which the appear in the type_to_param_map it must be an OrderedDict
    """

    def _check_one_param_form_per_type():
        for type, param_types in type_to_param_map.items():
            assert len(param_types) == 1, 'More than one parametric form {} per type {}'.format(
                param_types, type)

    leaf = None
    priors = {}

    if leaf_type == 'tmw' or leaf_type == 'pmw':

        if leaf_type == 'tmw':
            _check_one_param_form_per_type()

        leaf = TypeMixture(meta_type=leaf_meta_type, scope=scope)
        for _type, param_types in type_to_param_map.items():
            for param_class, param_map in param_types.items():
                param_leaf = param_class(scope=scope, **param_map['params'])
                leaf.add_parametric_leaf(param_leaf)
                priors[param_leaf] = param_map['prior']

    elif leaf_type == 'tm' or leaf_type == 'pm':

        if leaf_type == 'tm':
            _check_one_param_form_per_type()

        leaf = TypeMixtureUnconstrained(meta_type=leaf_meta_type, scope=scope)
        for _type, param_types in type_to_param_map.items():
            for param_class, param_map in param_types.items():
                print(param_map)
                param_leaf = param_class(scope=scope,  **param_map['params'])
                leaf.add_parametric_leaf(param_leaf, init_weights[param_class])
                # if leaf_type == 'tm':
                #     leaf.add_parametric_leaf(param_leaf, init_weights[j])
                # elif leaf_type == 'pm':
                #     leaf.add_parametric_leaf(param_leaf, init_weights[param_class])
                priors[param_leaf] = param_map['prior']

    #
    # FIXME: tm-pm/w fat leaves are likely broken for the init weights to pass
    # Need more careful testing
    elif leaf_type == 'tm-pmw':

        leaf = TypeMixture(meta_type=leaf_meta_type, scope=scope)
        for _type, param_types in type_to_param_map.items():
            param_leaf = ParametricMixture(type=_type, meta_type=leaf_meta_type, scope=scope)
            leaf.add_parametric_leaf(param_leaf)
            for param_class, param_map in param_types.items():
                p_leaf = param_class(scope=scope,  **param_map['params'])
                param_leaf.add_parametric_leaf(p_leaf, init_weights[_type][param_class])
                priors[p_leaf] = param_map['prior']

    elif leaf_type == 'tm-pm':
        leaf = TypeMixtureUnconstrained(meta_type=leaf_meta_type, scope=scope)
        for _type, param_types in type_to_param_map.items():
            param_leaf = ParametricMixture(type=_type, meta_type=leaf_meta_type, scope=scope)
            leaf.add_parametric_leaf(param_leaf, init_weights[_type]['w'])
            for param_class, param_map in param_types.items():
                p_leaf = param_class(scope=scope,  **param_map['params'])
                param_leaf.add_parametric_leaf(p_leaf, init_weights[_type][param_class])
                priors[p_leaf] = param_map['prior']

    else:
        raise ValueError('Unrecognized (fat) leaf type {}'.format(leaf_type))

    return leaf, priors


def create_random_parametric_leaf(data, ds_context, scope):
    """
    Method to be employed by LearnSPN-like pipeline to create a randomized type (mixture) leaf, based on convext parameters
    """

    assert len(scope) == 1, "scope of univariate histogram for more than one variable?"
    assert data.shape[1] == 1, "data has more than one feature?"

    idx = scope[0]
    meta_type = ds_context.meta_types[idx]
    type = ds_context.types[idx]
    type_to_param_map = ds_context.param_form_map[type]
    rand_gen = ds_context.rand_gen

    leaf = TypeMixtureUnconstrained(meta_type=meta_type, scope=scope)

    param_map = type_to_param_map[idx]

    #
    # radomly select a parametric concrete distribution
    rand_param_id = rand_gen.choice(len(param_map))
    rand_param_form = param_map[rand_param_id]

    #
    # then remove it from the dictionary for not reusing it in the same scope
    print('PARAM LIST before removing', param_map)
    param_map.pop(rand_param_id)
    print('PARAM LIST after removing', param_map, ds_context.param_form_map[type])

    param_class, param_map = rand_param_form
    param_leaf = param_class(scope=scope,  **param_map)
    leaf.add_parametric_leaf(param_leaf, 1.0)

    return leaf


from scipy.stats import gamma, lognorm, expon, geom


def mle_param_fit_gamma(data):
    x = np.copy(data)
    x = x[~np.isnan(x)]
    #
    # negative data? impossible gamma
    if np.any(x <= 0):
        return {'alpha': 1.1, 'beta': 1}

    # print('X gamme', x)
    x[np.isclose(x, 0)] = 1e-6
    #
    # zero variance? adding noise
    if np.isclose(np.std(x), 0):
        return {'alpha': np.mean(x), 'beta': 1}

    p = gamma.fit(x, floc=0)
    print('gamma fit params', p)
    alpha, loc, theta = p
    beta = 1.0 / theta
    if np.isfinite(alpha):
        return {'alpha': alpha, 'beta': beta}
    else:
        return {'alpha': 1.1, 'beta': 1}


def mle_param_fit_gaussian(data):
    data = data[~np.isnan(data)]
    mean = np.mean(data)
    stddev = np.std(data)
    return {'mean': mean, 'stdev': stddev}


def mle_param_fit_poisson(data):
    data = data[~np.isnan(data)]
    l = np.mean(data)
    return {'mean': l}


def mle_param_fit_exponential(data):
    data = data[~np.isnan(data)]
    # scale = expon.fit(data, floc=0)
    # l = 1.0 / scale
    l = np.mean(data)
    return {'l': l}


def mle_param_fit_geometric(data):
    data = data[~np.isnan(data)]
    p = len(data) / data.sum()
    return {'p': p}


def mle_param_fit_categorical(data, domain, EPS=1e-7):
    data = data[~np.isnan(data)]
    k = int(np.max(domain) + 1)
    p = np.zeros(k) + EPS
    for i in range(k):
        p[i] += np.sum(data == i)
    p = p / p.sum()
    return {'p': p}


def mle_fit_parametric_form_data(param_class, data, domain):

    fit_params = None

    if param_class == Gaussian:
        fit_params = mle_param_fit_gaussian(data)
    elif param_class == Gamma:
        fit_params = mle_param_fit_gamma(data)
    elif param_class == Exponential:
        fit_params = mle_param_fit_exponential(data)
    elif param_class == Categorical:
        fit_params = mle_param_fit_categorical(data, domain)
    elif param_class == Geometric:
        fit_params = mle_param_fit_geometric(data)
    elif param_class == Poisson:
        fit_params = mle_param_fit_poisson(data)
    else:
        raise ValueError('Unrecognized distribution to fit {}'.format(param_class))

    return fit_params


def fit_params_from_data(type_param_map, param_init, data, domain):
    """
    Fix or init parametric form params based on data
    if fit_map[c] is empty or null then no alteration to class c parameters
    would be provided
    """
    fit_type_param_map = deepcopy(type_param_map)
    for _type, param_types in type_param_map.items():
        for param_class, param_map in param_types.items():
            fit_param_map = None
            if param_init == 'mle':
                fit_param_map = mle_fit_parametric_form_data(param_class, data, domain)
            elif param_init == 'default':
                if param_class == Gamma:
                    est_param_map = mle_fit_parametric_form_data(param_class,
                                                                 data, domain)
                    #
                    # update only alpha, since it is remaining fixed
                    fit_param_map = {'alpha': est_param_map['alpha'],
                                     'beta': param_map['params']['beta']}
                elif param_class == Categorical:
                    fit_param_map = mle_fit_parametric_form_data(param_class,
                                                                 data, domain)
                else:
                    fit_param_map = dict(param_map['params'])

            elif param_init == 'prior':
                #
                # TODO: sample from the specified prior,
                # for the Categorical, it would be needed to estimate the dirichelt prior hyperparameters
                # for the Gamma, it would be advisable to do an MLE estimation for alpha, still
                raise NotImplementedError('Init parameter from prior still to be implemented')
            else:
                raise ValueError('Unrecognized param init mode {}'.format(param_init))

            fit_type_param_map[_type][param_class]['params'] = fit_param_map

            #
            # update Dirichlet prior for the Categorical,
            if param_class == Categorical:
                print('\n\n\n\nCATEGORICAL\n\n\n\n')
                dir_cat_alphas = np.zeros_like(
                    fit_type_param_map[_type][Categorical]['params']['p'])
                print(fit_type_param_map[_type][Categorical]['prior'].alphas_0, dir_cat_alphas)
                dir_cat_alphas[:] = fit_type_param_map[_type][Categorical]['prior'].alphas_0
                fit_type_param_map[_type][Categorical]['prior'].alphas_0 = dir_cat_alphas

    return fit_type_param_map


def create_type_leaf(data, ds_context, scope):
    """
    Method to be employed by LearnSPN-like pipeline to create a type leaf, based on context parameters
    """
    assert len(scope) == 1, "scope of univariate histogram for more than one variable?"
    assert data.shape[1] == 1, "data has more than one feature?"

    idx = scope[0]
    meta_type = ds_context.meta_types[idx]
    leaf_type = ds_context.leaf_type
    param_map = ds_context.param_form_map[meta_type]
    init_weights = ds_context.init_weights_map[meta_type]
    param_init = ds_context.param_init
    domains = ds_context.domains

    #
    # eventually filling parameters that are 'fixed' during learning
    param_map = fit_params_from_data(param_map, param_init, data, domains[idx])

    leaf, leaf_prior = type_mixture_leaf_factory(leaf_type=leaf_type,
                                                 leaf_meta_type=meta_type,
                                                 type_to_param_map=param_map,
                                                 scope=scope,
                                                 init_weights=init_weights)

    #
    # assign row_ids (through context)
    # leaf.row_ids = ds_context.row_ids

    #
    # store the prior back into the contex
    # this is ugly...FIXME
    ds_context.priors.update(leaf_prior)

    return leaf


def create_leaf_univariate(data, ds_context, scope):
    assert len(scope) == 1, "scope of univariate for more than one variable?"
    assert data.shape[1] == 1, "data has more than one feature?"

    idx = scope[0]

    family = ds_context.family[idx]

    if family == "poisson":
        assert np.all(data >= 0), "poisson negative?"
        mean = np.mean(data)
        return Poisson(mean)

    raise Exception('Unknown family: ' + family)


def set_leaf_params(node, param_map, leaf_node_type=Parametric):
    """
    Resets the parameters for each leaf node
    """

    leaves = get_nodes_by_type(node, leaf_node_type)
    for l in leaves:
        if l.id in param_map:
            for param, param_value in param_map[l.id].items():
                print('SETTING', l.params)
                setattr(l, param, param_value)
                print('SETTING', l.params)

    return node


def set_omegas_params(node, param_map, node_type=Sum):
    """
    Resets the weights omegas for each sum node
    """

    sums = get_nodes_by_type(node, node_type)
    for s in sums:
        if s.id in param_map:
            setattr(s, 'weights', param_map[s.id])

    return node


TYPE_PARAM_MAP = {
    Gaussian: 0,
    Gamma: 1,
    Exponential: 2,
    Poisson: 3,
    Categorical: 4,
    Geometric: 5,
    LogNormal: 6,
    Beta: 7,
    Gumbel: 8,
    Laplace: 9,
    Wald: 10,
    Weibull: 11,
}

INV_TYPE_PARAM_MAP = {p_id: p_class for p_class, p_id in TYPE_PARAM_MAP.items()}


def get_type_partitioning_leaves(spn, leaf_partitioning, leaf_type=Parametric,
                                 type_map=TYPE_PARAM_MAP):
    """
    Return a type partitioning as a NxD matrix representing the parametric form/type
    associated to each entry, computed from a NxD matrix representing a leaf_partitioning
    (labelling each entry with the id of the leaf it is associated to)
    """

    N, D = leaf_partitioning.shape

    leaf_type_map = {l.id: l.__class__ for l in get_nodes_by_type(spn, leaf_type)}
    type_P = np.zeros((N, D), dtype=np.int64)

    for n in range(N):
        for d in range(D):
            type_P[n, d] = type_map[leaf_type_map[leaf_partitioning[n, d]]]

    return type_P
