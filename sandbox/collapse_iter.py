from collections import defaultdict
from functools import reduce
import operator


import networkx as nx
from block import *


def get_children(expr):
    '''In cbc.block tree'''
    if isinstance(expr, (block_add, block_sub)):
        return (expr.A, expr.B)

    if isinstance(expr, block_mul):
        return tuple(expr.chain)

    if isinstance(expr, block_transpose):
        return (expr.A, )

    return tuple()


def is_terminal(expr):
    '''Root node in cbc.block tree'''
    return not get_children(expr)


def expr2graph(expr):
    '''Represent cbc.block expression as DAG'''
    g = nx.DiGraph()
    terminal_labels = defaultdict(list)
    
    if is_terminal(expr):
        g.add_node('A_0')
        terminal_labels[expr].append(0)

        return g, terminal_labels

    terminal_counter = 0
    subexpr = [expr]
    while subexpr:
        parent = subexpr.pop()
            
        children = get_children(parent)
        # Just for fun let's make the tree at most binary
        if len(children) > 2:
            child, *rest = children

            if isinstance(parent, block_mul):
                op = operator.mul
            else:
                raise ValueError
            
            children = (child, reduce(op, rest))
        
        for child in children:
            
            if is_terminal(child):
                terminal_labels[child].append(terminal_counter)
                terminal_counter += 1
                child_node = f'A_{terminal_labels[child][-1]}'
            else:
                subexpr.append(child)
                child_node = child

            g.add_edge(parent, child_node)
            
    return g, terminal_labels


def terminal_nodes(graph):
    '''Nodes representing matrices/number in cbc.block AST'''
    for node in graph.nodes:
        if graph.out_degree(node) == 0:
            yield node

            
def is_ubinary_tree(graph):
    '''Check'''
    for node in graph.nodes:
        if graph.out_degree(node) > 2:
            return False
    return True
            

def collapsables(graph):
    '''Subtrees that can be collapsed into a terminal'''
    terminals = list(terminal_nodes(graph))
    parents = defaultdict(list)
    for node in terminals:
        for edge in graph.in_edges(node):
            parent, = set(edge) - set((node, ))
            parents[parent].append(node)
    
    for parent, children in parents.items():
        g = nx.DiGraph()
        g.add_edges_from((parent, child) for child in children)
        yield g


# def collapse_tree(tree, terminals):
#     root, = (node for node in tree.nodes if tree.out_degree(node) > 0)
#     args = nx.ancestors(tree, node)

#     if isinstance(root, block_add):
#         return collapse_add(*args)

# def collapse(expr):
#     while not is_terminal(expr):
#         graph, terminals = expr2graph(expr)

#         sub_exprs = list(collapsables(graph))
#         for sub_expr in sub_exprs:
#             terminal = collapse_(sub_expr)
            
#             node_name
#             replace_sub_expr(graph, sub_expr, 

#     return expr
    
# --------------------------------------------------------------------

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import dolfin as df
    
    mesh = df.UnitSquareMesh(2, 2)
    V = df.FunctionSpace(mesh, 'CG', 1)
    u, v = df.TrialFunction(V), df.TestFunction(V)

    M = df.assemble(df.inner(u, v)*df.dx)

    g, terminals = expr2graph(M)
    
    expr = 2*M + 3*M - M + M.T + M*M*M
    g, terminals = expr2graph(expr)

    pos = nx.spectral_layout(g)

    fig, ax = plt.subplots()
    nx.draw(g, pos=pos, ax=ax)
    nx.draw_networkx_labels(g, pos=pos,
                            verticalalignment='bottom',
                            horizontalalignment='right',
                            font_size=16,
                            ax=ax)
    plt.show()
