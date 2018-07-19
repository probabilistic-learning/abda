'''
Created on March 21, 2018

@author: Alejandro Molina
'''
from spn.io.Text import spn_to_str_equation
from spn.io.Text import add_str_to_spn, add_node_to_str

from spn.structure.StatisticalTypes import MetaType
from spn.structure.leaves.parametric.Text import add_parametric_text_support
from spn.structure.leaves.typedleaves.TypedLeaves import TypeMixture, TypeMixtureUnconstrained


def type_mixture_to_str(node, feature_names=None, node_to_str=None):
    def fmt_chld(w, c): return str(w) + \
                               "*(" + spn_to_str_equation(c, feature_names, node_to_str) + ")"

    children_strs = map(lambda i: fmt_chld(
        node.weights[i], node.children[i]), range(len(node.children)))

    return "TypeMixture(" + str(node.meta_type.name) + ", " + " + ".join(children_strs) + ")"


def type_mixture_uncon_to_str(node, feature_names=None, node_to_str=None):
    def fmt_chld(w, c): return str(w) + \
                               "*(" + spn_to_str_equation(c, feature_names, node_to_str) + ")"

    children_strs = map(lambda i: fmt_chld(
        node.weights[i], node.children[i]), range(len(node.children)))

    return "TypeMixtureUnconstrained(" + str(node.meta_type.name) + ", " + " + ".join(children_strs) + ")"


#
# FIXME: TypeMixtures are assuming constrained, SHARED global weights
def type_mixture_tree_to_spn(tree, features, obj_type, tree_to_spn):
    node = TypeMixture(MetaType[tree.children[0]])
    children = tree.children[1:]
    for i in range(int(len(children) / 2)):
        j = 2 * i
        w, c = children[j], children[j + 1]
        node.weights.append(float(w))
        node.children.append(tree_to_spn(c, features))
    return node


def type_mixture_uncon_tree_to_spn(tree, features, obj_type, tree_to_spn):
    node = TypeMixtureUnconstrained(MetaType[tree.children[0]])
    children = tree.children[1:]
    for i in range(int(len(children) / 2)):
        j = 2 * i
        w, c = children[j], children[j + 1]
        node.weights.append(float(w))
        node.children.append(tree_to_spn(c, features))
    return node


#
# This is likely broken
def add_typed_leaves_text_support():
    add_parametric_text_support()
    add_node_to_str(TypeMixture, type_mixture_to_str)
    add_node_to_str(TypeMixtureUnconstrained, type_mixture_uncon_to_str)

    add_str_to_spn("typemixture", type_mixture_tree_to_spn,
                   """typemixture: "TypeMixture(" [PARAMNAME]"," [DECIMAL "*" node ("+" DECIMAL "*" node)*] ")" """,
                   TypeMixture)

    add_str_to_spn("typemixtureunconstrained", type_mixture_uncon_tree_to_spn,
                   """typemixtureunconstrained: "TypeMixtureUnconstrained(" [PARAMNAME]"," [DECIMAL "*" node ("+" DECIMAL "*" node)*] ")" """,
                   TypeMixtureUnconstrained)
