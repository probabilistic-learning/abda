'''
Created on April 5, 2018

@author: Alejandro Molina
'''
import logging

import numpy as np

from spn.algorithms.Inference import likelihood, log_likelihood
from spn.io.Text import str_to_spn, to_JSON
from spn.structure.Base import Product, Sum, Leaf, get_nodes_by_type, reset_node_counters
from spn.structure.StatisticalTypes import MetaType
from spn.structure.leaves.parametric.Parametric import Parametric
from spn.structure.leaves.parametric.Sampling import sample_parametric_node
from spn.structure.leaves.typedleaves.TypedLeaves import TypeMixtureUnconstrained, TypeMixture


def init_spn_sampling(node):
    all_nodes = get_nodes_by_type(node)

    map_id_nodes = {}
    for n in all_nodes:
        map_id_nodes[n.id] = n

    reset_node_counters(node)

    return map_id_nodes


def validate_ids(node):
    all_nodes = get_nodes_by_type(node)

    ids = set()
    for n in all_nodes:
        ids.add(n.id)

    assert len(ids) == len(all_nodes), "not all nodes have ID's"

    assert min(ids) == 0 and max(ids) == len(ids) - 1, "ID's are not in order"


def sample_induced_trees(node, data, rand_gen):
    # this requires ids to be set, and to be ordered from 0 to N
    validate_ids(node)

    max_id = reset_node_counters(node)

    lls = np.zeros((data.shape[0], max_id + 1))
    log_likelihood(node, data, llls_matrix=lls)

    map_rows_cols_to_node_id = np.zeros(data.shape, dtype=np.int64) - 1

    # We do not collect all the Zs from all sum nodes as before, but only to those
    # traversed during the top-down descent
    def _sample_induced_trees(node, row_ids):
        if len(row_ids) == 0:
            return
        node.row_ids = row_ids

        if isinstance(node, Product):
            for c in node.children:
                _sample_induced_trees(c, row_ids)
            return

        if isinstance(node, Sum):
            w_children_log_probs = np.zeros((len(row_ids), len(node.weights)))
            for i, c in enumerate(node.children):
                w_children_log_probs[:, i] = lls[row_ids, c.id] + np.log(node.weights[i])

            z_gumbels = rand_gen.gumbel(loc=0, scale=1,
                                        # size=(w_children_log_probs.shape[1], w_children_log_probs.shape[0]))
                                        size=(w_children_log_probs.shape[0], w_children_log_probs.shape[1]))
            g_children_log_probs = w_children_log_probs + z_gumbels
            rand_child_branches = np.argmax(g_children_log_probs, axis=1)

            for i, c in enumerate(node.children):
                new_row_ids = row_ids[rand_child_branches == i]
                node.edge_counts[i] = len(new_row_ids)
                _sample_induced_trees(c, new_row_ids)

        if isinstance(node, Leaf):
            map_rows_cols_to_node_id[row_ids, node.scope] = node.id
            return

    _sample_induced_trees(node, np.arange(data.shape[0]))

    return map_rows_cols_to_node_id, lls


def sample_instances(node, D, n_samples, rand_gen, return_Zs=True, return_partition=True, dtype=np.float64):
    """
    Implementing hierarchical sampling

    D could be extracted by traversing node
    """

    sum_nodes = get_nodes_by_type(node, Sum)
    n_sum_nodes = len(sum_nodes)

    if return_Zs:
        Z = np.zeros((n_samples, n_sum_nodes), dtype=np.int64)
        Z_id_map = {}
        for j, s in enumerate(sum_nodes):
            Z_id_map[s.id] = j

    if return_partition:
        P = np.zeros((n_samples, D), dtype=np.int64)

    instance_ids = np.arange(n_samples)
    X = np.zeros((n_samples, D), dtype=dtype)

    _max_id = reset_node_counters(node)

    def _sample_instances(node, row_ids):
        if len(row_ids) == 0:
            return
        node.row_ids = row_ids

        if isinstance(node, Product):
            for c in node.children:
                _sample_instances(c, row_ids)
            return

        if isinstance(node, Sum):
            w_children_log_probs = np.zeros((len(row_ids), len(node.weights)))
            for i, c in enumerate(node.children):
                w_children_log_probs[:, i] = np.log(node.weights[i])

            z_gumbels = rand_gen.gumbel(loc=0, scale=1,
                                        size=(w_children_log_probs.shape[0], w_children_log_probs.shape[1]))
            g_children_log_probs = w_children_log_probs + z_gumbels
            rand_child_branches = np.argmax(g_children_log_probs, axis=1)

            for i, c in enumerate(node.children):
                new_row_ids = row_ids[rand_child_branches == i]
                node.edge_counts[i] = len(new_row_ids)
                _sample_instances(c, new_row_ids)

                if return_Zs:
                    Z[new_row_ids, Z_id_map[node.id]] = i

        if isinstance(node, Leaf):
            #
            # sample from leaf
            X[row_ids, node.scope] = sample_parametric_node(
                node, n_samples=len(row_ids), rand_gen=rand_gen)
            if return_partition:
                P[row_ids, node.scope] = node.id

            return

    _sample_instances(node, instance_ids)

    if return_Zs:
        if return_partition:
            return X, Z, P

        return X, Z

    if return_partition:
        return X, P

    return X


def sample_Ws(W, meta_types, S_counts, W_priors, rand_gen):
    for d in W_priors.keys():
        if meta_types[d] == MetaType.BINARY:
            continue

        W[d][:] = rand_gen.dirichlet(S_counts[d] + W_priors[d], size=1)[0, :]


def sample_spn_weights(node, rand_gen, omega_prior="uniform", omega_uninf_prior=10, leaf_omega_uninf_prior=0.1):
    assert omega_uninf_prior >= 0, "omega uniform prior can't be negative"
    #
    # We are sampling/updating all the Omegas here---that is, for all the sum nodes in the SPN---
    # when we have not visited it, if the endge_counts are 0, then we should be sampling from the prior
    #

    n = len(node.row_ids)

    def get_edge_counts(sum_node):
        edge_counts = np.array(sum_node.edge_counts)

        X = rand_gen.choice(np.arange(len(edge_counts)), p=sum_node.weights,
                            size=n - np.sum(edge_counts))
        for i in range(len(edge_counts)):
            edge_counts[i] += np.sum(X == i)

        assert n == np.sum(edge_counts), "edges of sum node don't have all the instances"
        return edge_counts

    for sum_node in get_nodes_by_type(node, Sum):

        if isinstance(sum_node, TypeMixture):
            continue

        init_w_priors = None

        #
        #
        if isinstance(sum_node, TypeMixtureUnconstrained):
            init_w_priors = np.array([leaf_omega_uninf_prior for c in sum_node.children])
        else:
            #
            # FIXME: in the new implementation this learnspn prior is broken, right Ale?
            if omega_prior == 'learnspn':
                init_w_priors = np.array([c.instance_counts_prior() for c in sum_node.children])
            elif omega_prior == 'uniform':
                init_w_priors = np.array([omega_uninf_prior for c in sum_node.children])
            elif omega_prior == 'forced-uniform':
                init_w_priors = np.array(
                    [omega_uninf_prior - sum_node.edge_counts for c in sum_node.children])
            elif omega_prior is None or omega_prior == 'None':
                init_w_priors = np.zeros(len(sum_node.children))
            else:
                raise ValueError('Unrecognized omega prior {}'.format(omega_prior))

        edge_counts = get_edge_counts(sum_node)

        sum_node.weights[:] = rand_gen.dirichlet(edge_counts + init_w_priors, 1)[0, :]
        logging.debug("%s node %s, edge_counts %s, weights %s, prior %s" %
                      (sum_node.__class__.__name__, sum_node.id, sum_node.edge_counts, sum_node.weights, init_w_priors))


# def random_partition_data_matrix(N, D, meta_types, param_form_map, m, rand_gen, return_spn=False,
#                                  dtype=np.float32):
#     """
#     Generates a random guillotine partitioning for a NxD data matrix X
#     by implicitly building one spn representing such a partitioning
#     given

#       - the list of meta types (size D) associated to features in X
#       - a map associating potetial parametric forms to each meta type
#       - the minimum number of instances (m) before stopping the process
#     """

#     X = np.zeros((N, D), dtype=dtype)
#     spn = None

#     def _random_partition_slice(row_ids, col_ids):

#         #
#         # create leaf
#         if len(col_ids) == 1:

#             return

#         #
#         # naive factorization
#         if len(row_ids) < m:
#             node =

#     if return_spn:
#         return X, spn

#     return X


def validate_row_partitioning(node, instance_set):
    """
    Checks whether samples are currently correctly partitioned in an SPN.
    The input node is assumed to be the root of the sub-network considered
    """

    def _validate_row_partitioning(node):

        if isinstance(node, Parametric):
            return set(node.row_ids)

        if isinstance(node, Product):

            instances_sets = []
            for c in node.children:
                c_inst_set = _validate_row_partitioning(c)
                if instances_sets:
                    sibl_set = instances_sets[-1]
                    assert sibl_set == c_inst_set, "Not identical sets of instances {} {}".format(
                        sibl_set, c_inst_set)

                instances_sets.append(c_inst_set)

            row_set = instances_sets[0]
            assert set(node.row_ids) == row_set
            return row_set

        if isinstance(node, Sum):
            instances_sets = []
            for c in node.children:
                c_inst_set = _validate_row_partitioning(c)
                for sibl_set in instances_sets:
                    assert len(sibl_set & c_inst_set) == 0, "Overlapping sets of instances {} {}".format(
                        sibl_set, c_inst_set)
                instances_sets.append(c_inst_set)

            row_set = set.union(*instances_sets)
            assert set(node.row_ids) == row_set
            return row_set

    whole_row_ids = _validate_row_partitioning(node)
    instance_set = set(instance_set)
    assert instance_set == whole_row_ids, ' Expected row ids {}, got {}'.format(instance_set,
                                                                                whole_row_ids)


if __name__ == '__main__':
    n = str_to_spn("""
            (
            Histogram(W1|[ 0., 1., 2.];[0.3, 0.7])
            *
            Histogram(W2|[ 0., 1., 2.];[0.3, 0.7])
            )    
            """, ["W1", "W2"])

    n = str_to_spn("""
            (0.3 * Histogram(W1|[ 0., 1., 2.];[0.2, 0.8])
            +
            0.7 * Histogram(W1|[ 0., 1., 2.];[0.1, 0.9])
            )    
            """, ["W1", "W2"])

    print(to_JSON(n))

    map_id_node = init_spn_sampling(n)

    data = np.vstack((np.asarray([1.5, 0.5]), np.asarray([0.5, 0.5]),
                      np.asarray([0.7, 0.5]), np.asarray([0.5, 0.7])))

    print(data)
    rand_gen = np.random.RandomState(17)
    map_rows_cols_to_node_id, lls = sample_induced_trees(n, data, rand_gen)
    print(map_rows_cols_to_node_id)
    print(map_id_node)
    print(lls)

    sample_spn_weights(n, rand_gen)
    print(to_JSON(n))
