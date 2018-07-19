'''
Created on March 25, 2018

@author: Alejandro Molina
'''
from collections import Counter

from spn.structure.Base import get_nodes_by_type, Sum, Product, Leaf, get_number_of_edges, get_number_of_layers, Node
from spn.structure.leaves.parametric.Parametric import Parametric
from spn.structure.leaves.typedleaves.TypedLeaves import TypeLeaf


def get_structure_stats_dict(node):
    num_nodes = len(get_nodes_by_type(node, Node))
    sum_nodes = len([s for s in get_nodes_by_type(node, Sum) if type(s) == Sum])
    prod_nodes = len(get_nodes_by_type(node, Product))
    leaf_nodes = len(get_nodes_by_type(node, Leaf))
    param_leaf_nodes = len(get_nodes_by_type(node, Parametric))
    type_leaf_nodes = len(get_nodes_by_type(node, TypeLeaf))
    edges = get_number_of_edges(node)
    layers = get_number_of_layers(node)

    return {'nodes': num_nodes, 'edges': edges, 'layers': layers,
            'sum': sum_nodes, 'prod': prod_nodes, 'leaf': leaf_nodes,
            'param-leaf': param_leaf_nodes, 'type-leaf': type_leaf_nodes}


def get_structure_stats_dict2(node):
    node_types = dict(Counter([type(n) for n in get_nodes_by_type(node)]))
    num_nodes = len(get_nodes_by_type(node, Node))
    edges = get_number_of_edges(node)
    layers = get_number_of_layers(node)

    return {'nodes': num_nodes, 'edges': edges, 'layers': layers}.update(node_types)


def get_structure_stats(node):
    num_nodes = len(get_nodes_by_type(node, Node))
    sum_nodes = len(get_nodes_by_type(node, Sum))
    prod_nodes = len(get_nodes_by_type(node, Product))
    leaf_nodes = len(get_nodes_by_type(node, Leaf))
    edges = get_number_of_edges(node)
    layers = get_number_of_layers(node)

    return """---Structure Statistics---
# nodes             %s
    # sum nodes     %s
    # prod nodes    %s
    # leaf nodes    %s
# edges             %s
# layers            %s""" % (num_nodes, sum_nodes, prod_nodes, leaf_nodes, edges, layers)
