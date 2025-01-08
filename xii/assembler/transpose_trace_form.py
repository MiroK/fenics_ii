from ufl_legacy.corealg.traversal import traverse_unique_terminals
from xii.assembler.ufl_utils import *
import dolfin as df
import numpy as np
import ufl_legacy as ufl


def transpose_trace_cell(o):
    '''TODO'''
    # Space
    if hasattr(o, 'ufl_cell'):
        return transpose_trace_cell(o.ufl_cell())
    # Foo like
    if hasattr(o, 'ufl_element'):
        return transpose_trace_cell(o.ufl_element().cell())
    # Elm
    if hasattr(o, 'cell'):
        return transpose_trace_cell(o.cell())

    # Another cell
    cell_name = {'triangle': 'tetrahedron',
                 'interval': 'triangle'}[o.cellname()]

    return ufl.Cell(cell_name, o.geometric_dimension())


def transpose_trace_element(elm):
    '''TODO'''
    # Want exact match here; otherwise VectorElement is MixedElement and while
    # it works I don't find it pretty
    if type(elm) == df.MixedElement:
        return df.MixedElement(list(map(transpose_trace_element, elm.sub_elements())))
    
    # FIXME: Check out Witze Bonn's work on da Rham for trace spaces
    # in the meantime KISS
    family = elm.family()
    
    family_map = {'Lagrange': 'Lagrange'}
    # This seems like a reasonable fall back option
    family = family_map[family]#, 'Discontinuous Lagrange')

    degree = elm.degree()  # Preserve degree
    cell = transpose_trace_cell(elm)

    # How to construct:
    # There is an issue here where e.g. Hdiv are not scalars, their
    # element is FiniteElement but we want trace space from VectorElement
    elmtype_map = {0: df.FiniteElement,
                   1: df.VectorElement,
                   2: df.TensorElement}
    # So let's check first for elements where scalar = FiniteElm, vector == VectorElm 
    rank = len(elm.value_shape())
    if elmtype_map[rank] == type(elm):
        elm = type(elm)  # i.e. vector element stays vector element
    else:
        elm = elmtype_map[rank]


    return elm(family, cell, degree)


def transpose_trace_space(V, mesh):
    '''TODO'''    
    return df.FunctionSpace(mesh, transpose_trace_element(V.ufl_element()))


def TransposeTrace(v, trace_mesh, full_mesh=None):
    '''TODO'''
    # Prevent Trace(grad(u)). But it could be interesting to have this
    assert is_terminal(v)
    assert hasattr(trace_mesh, 'parent_entity_map')

    if full_mesh is None:
        full_mesh, = trace_mesh.parent_meshes
    assert full_mesh.id() in trace_mesh.parent_entity_map

    if isinstance(v, ufl.Coefficient):
        v =  df.Function(v.function_space(), v.vector())
    else:
        # Object copy?
        v = [df.TestFunction, df.TrialFunction][v.number()](v.function_space())
        
    v.transpose_trace_ = {'trace_mesh': trace_mesh, 'full_mesh': full_mesh}

    return v


def is_transpose_trace_integrand(expr, tdim=None):
    '''Some of the arguments need restriction'''
    return any(hasattr(arg, 'transpose_trace_')
               for arg in traverse_unique_terminals(expr))


def is_transpose_trace_integral(integral):
    '''TODO'''
    return is_transpose_trace_integrand(integral.integrand())


def transpose_trace_integrals(form):
    '''TODO'''
    return list(filter(is_transpose_trace_integral, form.integrals()))
