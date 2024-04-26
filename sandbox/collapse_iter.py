from collections import defaultdict
from functools import reduce
import operator

import networkx as nx
import dolfin as df
from block import block_add, block_sub, block_mul, block_transpose
from xii.linalg.matrix_utils import (is_petsc_vec, is_petsc_mat, is_number,
                                     as_petsc)
from petsc4py import PETSc


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
        
        for (index, child) in enumerate(children):
            # We want to keep track of order of arguments so we attach ...
            if is_terminal(child):
                child_node = f'A_{terminal_counter}'                                
                terminal_labels[child].append(child_node)
                terminal_counter += 1
            else:
                subexpr.append(child)
                child_node = child
            # ... index to edge
            g.add_edge(parent, child_node, index=index)
            
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
            

def collapsable_trees(graph):
    '''Subtrees that can be collapsed into a terminal'''
    # Our definnition of tree is a node whose followers are terminals
    terminals = set(terminal_nodes(graph))
    parents = defaultdict(list)
    for node in terminals:
        for edge in graph.in_edges(node):
            index = graph.get_edge_data(*edge)['index']
            # This is a suspect
            parent, = set(edge) - set((node, ))
            # It is okay only if the parent is connected to other terminals
            set(graph.successors(parent)) <= terminals and parents[parent].append((node, index))

    for parent in parents:
        # When building the subtree we want to preserve the order
        children, _ = zip(*sorted(parents[parent], key=lambda p: p[1]))
        g = nx.DiGraph()
        for (index, child) in enumerate(children):
            g.add_edge(parent, child, index=index)
        yield g


def get_tree_root(tree):
    '''Assuming (+, A, B) this is +'''
    root, = (node for node in tree.nodes if tree.out_degree(node) > 0)
    return root


def replace_tree(graph, tree, node):
    '''Replace tree in the graph by node'''
    root = get_tree_root(tree)
    # Grab
    predecessors = list(graph.predecessors(root))
    # We have graph (+, A, B)
    if not predecessors: return graph, True
    # There's more
    root_parent, = predecessors
    # When replace subtree we need to preserve index as well
    index = graph.get_edge_data(root_parent, root)['index']
    # Remove all the nodes
    graph.remove_edges_from(tree.edges)
    graph.remove_nodes_from(tree.nodes)
    # Put in a new one
    graph.add_edge(root_parent, node, index=index)

    return graph, False


def collapse_tree(tree, terminals):
    '''Unary/binary operations over matrices'''
    root = get_tree_root(tree)
    args = [arg for arg in tree.successors(root)]
    # Keep order
    args = sorted(args, key=lambda node, tree=tree: tree.get_edge_data(root, node)['index'])
    # Lookup the values
    args = [terminals[arg] for arg in args]

    if isinstance(root, block_add):
        return collapse_add(*args)

    if isinstance(root, block_sub):
        return collapse_sub(*args)

    if isinstance(root, block_mul):
        return collapse_mul(*args)

    if isinstance(root, block_transpose):
        return collapse_transpose(*args)
    # Otherwise
    raise ValueError


def collapse_transpose(A):
    '''to Transpose'''
    if is_petsc_mat(A):
        A_ = as_petsc(A)
        C_ = PETSc.Mat()
        A_.transpose(C_)
        return df.PETScMatrix(C_)
    # Recurse
    assert is_number(A)
    return A


def collapse_add(A, B):
    '''A + B to single matrix'''
    if is_petsc_mat(A) and is_number(B) and abs(B) < 1E-14:
        return A
    if is_petsc_mat(B) and is_number(A) and abs(A) < 1E-14:
        return B    
    # Base case
    if is_petsc_mat(A) and is_petsc_mat(B):
        A_ = as_petsc(A)
        B_ = as_petsc(B)
        assert A_.size == B_.size
        C_ = A_.copy()
        # C = A + B
        C_.axpy(1., B_, PETSc.Mat.Structure.DIFFERENT)
        return df.PETScMatrix(C_)
    # Otherwise
    raise ValueError


def collapse_sub(A, B):
    '''A - B to single matrix'''
    # Base case
    if is_petsc_mat(A) and is_petsc_mat(B):
        A_ = as_petsc(A)
        B_ = as_petsc(B)
        assert A_.size == B_.size
        C_ = A_.copy()
        # C = A - B
        C_.axpy(-1., B_, PETSc.Mat.Structure.DIFFERENT)
        return df.PETScMatrix(C_)
    # Recurse
    raise ValueError


def collapse_mul(A, B):
    '''A*B*C to single matrix'''
    # A0 * A1 * ...
    # Two matrices
    if is_petsc_mat(A) and is_petsc_mat(B):
        A_ = as_petsc(A)
        B_ = as_petsc(B)
        assert A_.size[1] == B_.size[0]
        C_ = PETSc.Mat()
        A_.matMult(B_, C_)
        return df.PETScMatrix(C_)
    
    # One of them is a number
    if is_petsc_mat(A) and is_number(B):
        A_ = as_petsc(A)
        C_ = A_.copy()
        C_.scale(B)
        return df.PETScMatrix(C_)

    if is_petsc_mat(B) and is_number(A):
        B_ = as_petsc(B)
        C_ = B_.copy()
        C_.scale(A)
        return df.PETScMatrix(C_)
    
    # Otherwise
    raise ValueError


def collapse(expr):
    '''To single matrix'''
    if is_terminal(expr): return expr

    graph, terminals = expr2graph(expr)
    # Here terminals is terminal -> it's name; we need to flip it
    terminals = {val: key for (key, values) in terminals.items() for val in values}
    # Starting from leave we want to change the expression to simple tree
    done = False
    while not done:
        for tree in list(collapsable_trees(graph)):
            # FIXME: this is a mock up
            # terminal = collapse_tree(sub_expr)
            mat = collapse_tree(tree, terminals)
            node = str(mat)
            # In graph we put name ...
            graph, done = replace_tree(graph, tree, node)
            # ... and we have a lookup
            terminals[node] = mat
    # Final one
    return collapse_tree(graph, terminals)
            
    return expr

# --------------------------------------------------------------------

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from xii import ii_convert
    import dolfin as df
    import numpy as np    
    
    mesh = df.UnitSquareMesh(16, 16)
    V = df.FunctionSpace(mesh, 'CG', 1)
    u, v = df.TrialFunction(V), df.TestFunction(V)

    M = df.assemble(df.inner(u, v)*df.dx)
    B = df.assemble(df.Constant(3)*df.inner(u, v)*df.dx)    

    # 
    #
    #
    #
    expr = 2*M + 3*M + M.T + M*M*M - M + B*M*B.T 
    
    g, terminals = expr2graph(expr)

    dt = df.Timer()
    A = collapse(expr)
    print(dt.stop())

    dt = df.Timer()    
    B = ii_convert(expr)
    print(dt.stop())    
    
    print(np.linalg.norm(A.array() - B.array()))
