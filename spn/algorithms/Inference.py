'''
Created on March 21, 2018

@author: Alejandro Molina
'''
from collections import Counter

import numpy as np
from scipy.special import logsumexp


from spn.structure.Base import Product, Sum, Leaf, reset_node_counters, max_node_id
from spn.structure.leaves.typedleaves.TypedLeaves import TypeMixtureUnconstrained

EPSILON = 0.000000000000001

_node_log_likelihood = {}


def add_node_likelihood(node_type, lambda_func):
    _node_log_likelihood[node_type] = lambda_func


_node_mpe_likelihood = {}


def add_node_mpe_likelihood(node_type, lambda_func):
    _node_mpe_likelihood[node_type] = lambda_func


def likelihood(node, data, dtype=np.float64, context=None, node_log_likelihood=_node_log_likelihood, lls_matrix=None):
    l = log_likelihood(node, data, dtype=dtype, context=context,
                       node_log_likelihood=node_log_likelihood, llls_matrix=lls_matrix)
    if lls_matrix is not None:
        lls_matrix[:, :] = np.exp(lls_matrix)
    return np.exp(l)


def log_likelihood(node, data, dtype=np.float64, context=None, node_log_likelihood=_node_log_likelihood, llls_matrix=None):

    assert len(data.shape) == 2, "data must be 2D, found: {}".format(data.shape)

    if node_log_likelihood is not None:
        t_node = type(node)
        if t_node in node_log_likelihood:
            ll = node_log_likelihood[t_node](node, data, dtype=dtype, context=context, node_log_likelihood=node_log_likelihood)
            if llls_matrix is not None:
                assert ll.shape[1] == 1, ll.shape[1]
                llls_matrix[:, node.id] = ll[:, 0]
                # if np.any(np.isposinf(ll)):
                #     print(ll, node, node.params, data[np.isposinf(ll)])
                #     0 / 0
            return ll

    is_product = isinstance(node, Product)

    is_sum = isinstance(node, Sum)

    if not (is_product or is_sum):
        raise Exception('Node type unknown: ' + str(type(node)))

    llchildren = np.zeros((data.shape[0], len(node.children)), dtype=dtype)

    # TODO: parallelize here
    for i, c in enumerate(node.children):
        llchild = log_likelihood(c, data, dtype=dtype, context=context,
                                 node_log_likelihood=node_log_likelihood, llls_matrix=llls_matrix)
        assert llchild.shape[0] == data.shape[0]
        assert llchild.shape[1] == 1
        llchildren[:, i] = llchild[:, 0]

    if is_product:
        ll = np.sum(llchildren, axis=1).reshape(-1, 1)

        # if np.any(np.isposinf(ll)):
        #     print(ll, node, data[np.isposinf(ll)])
        #     0 / 0

    elif is_sum:
        assert np.isclose(np.sum(node.weights), 1.0), "unnormalized weights {} for node {}".format(
            node.weights, node)
        b = np.array(node.weights, dtype=dtype)

        ll = logsumexp(llchildren, b=b, axis=1).reshape(-1, 1)

        # if np.any(np.isinf(ll)):
        #     inf_ids = np.isinf(ll.flatten())
        #     print(ll, ll.mean(), node, node.weights,
        #           data[inf_ids, node.scope[0]], llchildren[inf_ids, :],
        #           [(c, c.params) for c in node.children])
        #     0 / 0

    else:
        raise Exception('Node type unknown: ' + str(type(node)))

    assert ll.shape[1] == 1

    if llls_matrix is not None:
        llls_matrix[:, node.id] = ll[:, 0]

    return ll


# TODO: test this function super thorougly


def mpe_likelihood(node, data, log_space=True, dtype=np.float64,  context=None, node_mpe_likelihood=_node_mpe_likelihood,
                   lls_matrix=None):
    #
    # for leaves it should be the same, marginalization is being taken into account
    if node_mpe_likelihood is not None:
        t_node = type(node)
        if t_node in node_mpe_likelihood:
            ll = node_mpe_likelihood[t_node](node, data, log_space=log_space, dtype=dtype,  context=context, node_mpe_likelihood=node_mpe_likelihood)
            if lls_matrix is not None:
                assert ll.shape[1] == 1, ll.shape[1]
                lls_matrix[:, node.id] = ll[:, 0]
            return ll

    is_product = isinstance(node, Product)

    is_sum = isinstance(node, Sum)

    # print('nnode id', node.id, is_product, is_sum)

    if not (is_product or is_sum):
        raise Exception('Node type unknown: ' + str(type(node)))

    llchildren = np.zeros((data.shape[0], len(node.children)), dtype=dtype)

    # TODO: parallelize here
    for i, c in enumerate(node.children):
        llchild = mpe_likelihood(c, data, log_space=True, dtype=dtype,  context=context, node_mpe_likelihood=node_mpe_likelihood,
                                 lls_matrix=lls_matrix)
        assert llchild.shape[0] == data.shape[0]
        assert llchild.shape[1] == 1
        llchildren[:, i] = llchild[:, 0]

    if is_product:
        ll = np.sum(llchildren, axis=1).reshape(-1, 1)

        if not log_space:
            ll = np.exp(ll)

    elif is_sum:
        #
        # this actually computes the weighted max
        # b = np.array(node.weights, dtype=dtype)

        # ll = logsumexp(llchildren, b=b, axis=1).reshape(-1, 1)
        w_lls = llchildren + np.log(node.weights)
        # print(node.id, 'WLLs', w_lls, llchildren, np.log(node.weights))
        ll = np.max(w_lls, axis=1, keepdims=True)

        if not log_space:
            ll = np.exp(ll)
    else:
        raise Exception('Node type unknown: ' + str(type(node)))

    assert ll.shape[1] == 1

    if lls_matrix is not None:
        lls_matrix[:, node.id] = ll[:, 0]

    return ll


def conditional_log_likelihood(node_joint, node_marginal, data, log_space=True, dtype=np.float64):
    result = log_likelihood(node_joint, data, dtype) - \
        log_likelihood(node_marginal, data, dtype)
    if log_space:
        return result

    return np.exp(result)


def mpe(node, data, context=None):
    """
    Computing the MPE solution of a query, assuming that the unknown RVs---for which the MPE assignment shall be computed---are represented in data with NaNs (np.nan)

    Very similar to sampling_induced_trees
    """

    mpe_ass = np.zeros_like(data)
    mpe_ass[:] = np.nan
    print(mpe_ass.shape, 'ASS shape', data.shape)

    # max_id = reset_node_counters(node)
    max_id = max_node_id(node)

    lls = np.zeros((data.shape[0], max_id + 1))
    #
    # the likelihood of an MPN now would take care of
    # marginalization at the leaves
    mpe_likelihood(node, data, context=context, lls_matrix=lls)

    # We do not collect all the Zs from all sum nodes as before, but only to those
    # traversed during the top-down descent
    def _mpe_induced_tree(node, row_ids):
        if len(row_ids) == 0:
            return

        if isinstance(node, Product):
            for c in node.children:
                _mpe_induced_tree(c, row_ids)
            return

        if isinstance(node, Sum):
            w_children_log_probs = np.zeros((len(row_ids), len(node.weights)))
            for i, c in enumerate(node.children):
                w_children_log_probs[:, i] = lls[row_ids, c.id] + np.log(node.weights[i])

            mpe_child_branches = np.argmax(w_children_log_probs, axis=1)

            for i, c in enumerate(node.children):
                new_row_ids = row_ids[mpe_child_branches == i]
                _mpe_induced_tree(c, new_row_ids)

        if isinstance(node, Leaf):
            query_ids = np.zeros(mpe_ass.shape[0], dtype=bool)
            query_ids[row_ids] = np.isnan(data[row_ids, node.scope])
            mpe_ass[query_ids, node.scope] = node.mode
            return

    _mpe_induced_tree(node, np.arange(data.shape[0]))

    return mpe_ass, lls


def compute_global_type_weights(node, leaf_type=TypeMixtureUnconstrained, aggr_type=False):
    """
    Computes the global type weights as a map

      W = {d: w^{d}}

    where d are the features represented in the SPN rooted at node
    and w^{d} is a numpy array of the types/parametric forms associated to type of feature d
    """

    W = {}

    def _global_type_weights(node, W):

        if isinstance(node, leaf_type):

            if aggr_type:
                type_W_acc = Counter()
                for i, c in enumerate(node.children):
                    type_W_acc[c.type] += node.weights[i]
                W[node.scope[0]] = dict(type_W_acc)
            else:
                W[node.scope[0]] = {c.__class__: node.weights[i]
                                    for i, c in enumerate(node.children)}
            return W

        if isinstance(node, Product):
            for c in node.children:
                W = _global_type_weights(c, W)
            return W

        if isinstance(node, Sum):

            W_children = {}
            # for d, w in W.items():
            #     W_children[d] = np.zeros_like(w)

            for i, c in enumerate(node.children):
                W_c = _global_type_weights(c, dict(W))

                for d, w in W_c.items():
                    if d not in W_children:
                        # W_children[d] = np.zeros_like(w)
                        W_children[d] = {c_p: 0.0 for c_p, _w_c in w.items()}

                    # W_children[d] += w * node.weights[i]
                    for c_p, _w_c in w.items():
                        # print(c, d, W_children[d])
                        W_children[d][c_p] += w[c_p] * node.weights[i]

            return W_children

    return _global_type_weights(node, W)
