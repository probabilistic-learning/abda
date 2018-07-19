'''
Created on March 20, 2018

@author: Alejandro Molina
'''
import numpy as np


class Node:
    def __init__(self):
        self.id = 0
        self.scope = []

    @property
    def name(self):
        return "%sNode_%s" % (self.__class__.__name__, self.id)

    def __repr__(self):
        return self.name

    def __rmul__(self, weight):
        assert type(weight) == int or type(weight) == float
        self._tmp_weight = weight
        return self

    def __mul__(self, node):
        assert isinstance(node, Node)
        assert len(node.scope) > 0, "right node has no scope"
        assert len(self.scope) > 0, "left node has no scope"
        assert len(set(node.scope).intersection(set(self.scope))
                   ) == 0, "children's scope is not disjoint"
        result = Product()
        result.children.append(self)
        result.children.append(node)
        result.scope.extend(self.scope)
        result.scope.extend(node.scope)
        assign_ids(result)
        return result

    def __add__(self, node):
        assert isinstance(node, Node)
        assert hasattr(node, "_tmp_weight"), "right node has no weight"
        assert hasattr(self, "_tmp_weight"), "left node has no weight"
        assert len(node.scope) > 0, "right node has no scope"
        assert len(self.scope) > 0, "left node has no scope"
        assert set(node.scope) == (set(self.scope)), "children's scope are not the same"

        from numpy import isclose
        assert isclose(1.0, self._tmp_weight + node._tmp_weight), \
            "unnormalized weights, maybe trying to add many nodes at the same time?"

        result = Sum()
        result.children.append(self)
        result.weights.append(self._tmp_weight)
        result.children.append(node)
        result.weights.append(node._tmp_weight)
        result.scope.extend(self.scope)
        result._tmp_weight = self._tmp_weight + node._tmp_weight
        assign_ids(result)
        return result


class Sum(Node):
    def __init__(self):
        Node.__init__(self)
        self.weights = []
        self.children = []


class Product(Node):
    def __init__(self):
        Node.__init__(self)
        self.children = []


class Leaf(Node):
    def __init__(self, scope=None):
        Node.__init__(self)
        if scope is not None:
            if type(scope) == int:
                self.scope.append(scope)
            elif type(scope) == list:
                self.scope.extend(scope)
            else:
                raise Exception("invalid scope type %s " % (type(scope)))


class Context:
    def __init__(self, meta_types=None, domains=None):
        self.meta_types = meta_types
        self.domains = domains

    def get_meta_types_by_scope(self, scopes):
        return [self.meta_types[s] for s in scopes]

    def get_domains_by_scope(self, scopes):
        return [self.domains[s] for s in scopes]


def get_number_of_edges(node):
    return sum([len(c.children) for c in get_nodes_by_type(node, (Sum, Product))])


def get_number_of_layers(node):
    if isinstance(node, Leaf):
        return 1

    return max(map(get_number_of_layers, node.children)) + 1


def rebuild_scopes_bottom_up(node):
    # this function is not safe (updates in place)
    if isinstance(node, Leaf):
        return node.scope

    new_scope = set()
    for c in node.children:
        new_scope.update(rebuild_scopes_bottom_up(c))
    node.scope = list(new_scope)
    return node.scope


def bfs(root, func):
    import collections

    seen, queue = set([root]), collections.deque([root])
    while queue:
        node = queue.popleft()
        func(node)
        if not isinstance(node, Leaf):
            for node in node.children:
                if node not in seen:
                    seen.add(node)
                    queue.append(node)


def get_nodes_by_type(node, ntype=Node):
    result = []

    def add_node(node):
        if isinstance(node, ntype):
            result.append(node)

    bfs(node, add_node)

    return result


def assign_ids(node, ids=None):
    if ids is None:
        ids = {}

    def assign_id(node):
        if node not in ids:
            ids[node] = len(ids)

        node.id = ids[node]

    bfs(node, assign_id)


def get_parent_map(root):
    node_parent_map = {}
    for n in get_nodes_by_type(root, ntype=Node):
        if hasattr(n, 'children'):
            for c in n.children:
                node_parent_map[c] = n

    node_parent_map[root] = None
    return node_parent_map


def reset_node_counters(node):
    all_nodes = get_nodes_by_type(node)
    max_id = 0
    for n in all_nodes:
        # reset sum node counts
        if isinstance(n, Sum):
            n.edge_counts = np.zeros(len(n.children), dtype=np.int64)
        # sets nr_nodes to the max id
        max_id = max(max_id, n.id)
        n.row_ids = []
    return max_id


def max_node_id(node):
    all_nodes = get_nodes_by_type(node)
    max_id = 0
    for n in all_nodes:
        max_id = max(max_id, n.id)
    return max_id


def compute_leaf_global_mix_weights(spn):

    global_mix_weights = {}

    def _compute_leaf_global_mix_weights(node, w):

        global_mix_weights[node] = w
        # if isinstance(node, Leaf):
        #     global_mix_weights[node] = w
        # elif isinstance(node, Product):
        if isinstance(node, Product):
            for c in node.children:
                _compute_leaf_global_mix_weights(c, w)

        elif isinstance(node, Sum):
            for c, w_c in zip(node.children, node.weights):
                _compute_leaf_global_mix_weights(c, w * w_c)

    _compute_leaf_global_mix_weights(spn, w=1.0)

    return global_mix_weights


def compute_partition_id_map(spn):
    """
    A partition id is the id of the (closest) product node parent
    """

    parent_map = get_parent_map(spn)
    part_map = {}

    def _compute_partition_id_map(node):

        if isinstance(node, Leaf):
            p = parent_map[node]
            g_p = parent_map[p]
            assert isinstance(g_p, Product), '{} {}'.format(node, g_p)

            if g_p:
                part_map[node] = g_p.id
            else:
                part_map[node] = node.id

        elif isinstance(node, Product):

            part_map[node] = node.id
            for c in node.children:
                _compute_partition_id_map(c)

        elif isinstance(node, Sum):
            p = parent_map[node]
            if p:
                part_map[node] = p.id
            else:
                part_map[node] = node.id
            for c in node.children:
                _compute_partition_id_map(c)

    _compute_partition_id_map(spn)

    return part_map
