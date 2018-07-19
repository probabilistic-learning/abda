'''
Created on March 20, 2018

@author: Alejandro Molina
'''
from spn.structure.Base import Sum, Leaf, Product
from spn.structure.leaves.typedleaves.TypedLeaves import TypeMixture


def is_consistent(node):
    '''
    all children of a product node have different scope
    '''

    assert node is not None

    if len(node.scope) == 0:
        #print(node.scope, '0 scope const')
        return False

    if isinstance(node, Leaf):
        return True

    if isinstance(node, Product):
        nscope = set(node.scope)

        allchildscope = set()
        sum_features = 0
        for child in node.children:

            sum_features += len(child.scope)
            #print('cs ', sum_features, child.scope, child.__class__.__name__, node.scope)
            allchildscope = allchildscope | set(child.scope)

        if allchildscope != set(nscope) or sum_features != len(allchildscope):
            #print(allchildscope, set(nscope), sum_features, len(allchildscope), 'cons')
            return False

    return all(map(is_consistent, node.children))


def is_complete(node):
    '''
    all children of a sum node have same scope as the parent
    '''

    assert node is not None

    if len(node.scope) == 0:
        #print(node.scope, '0 scope')
        return False

    if isinstance(node, Leaf):
        return True

    if isinstance(node, Sum):
        nscope = set(node.scope)

        for child in node.children:
            if nscope != set(child.scope):
                #print(node.scope, child.scope, 'mismatch scope')
                return False

    return all(map(is_complete, node.children))


def is_aligned(node):
    if isinstance(node, Leaf):
        return True

    #
    # we have now can have sum nodes with no local weights (TypeMixture)
    if isinstance(node, Sum) and not isinstance(node, TypeMixture) and len(node.children) != len(node.weights):
        return False

    return all(map(is_aligned, node.children))


def is_valid(node):

    a = is_consistent(node)
    b = is_complete(node)
    c = is_aligned(node)

    print(a, b, c)

    return a and b and c
