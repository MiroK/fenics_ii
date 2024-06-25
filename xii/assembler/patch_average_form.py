from xii.assembler.ufl_utils import *

from ufl_legacy.corealg.traversal import traverse_unique_terminals
import dolfin as df
import ufl_legacy


def patch_average_cell(o):
    '''
    UFL cell corresponding to restriction of o[cell] to its edges, performing
    this restriction on o[function-like], or objects in o[function space]
    '''
    # Space
    if hasattr(o, 'ufl_cell'):
        return patch_average_cell(o.ufl_cell())
    # Foo like
    if hasattr(o, 'ufl_element'):
        return patch_average_cell(o.ufl_element().cell())
    # Another cell
    cell_name = {'tetrahedron': 'interval',
                 'triangle': 'interval'}[o.cellname()]
    
    return ufl_legacy.Cell(cell_name, o.geometric_dimension())


def patch_average_space(V, mesh, data):
    '''Construct a space over mesh where surface averages of V should live'''
    assert mesh.ufl_cell() == patch_average_cell(V)
    # We extract it
    TV = data['vertex_f'].function_space()
    # Sanity
    assert V.ufl_element().value_shape() == TV.ufl_element().value_shape()
    
    return TV


def CellPatchAverage(v, vertex_f, patch_f, patch_coloring=None):
    '''
    CellPathcAverage(v, ...) as a P1 function on the network (mesh of 
    vertex_f) whose values are obtained by patch averaging
    '''
    # FIXME: Don't want to deal with vectors at this point
    assert v.ufl_shape == ()
    
    omega = patch_f.mesh()
    assert omega.topology().dim() == patch_f.dim()
    # We want a scalar function that encodes dof color -> patch
    Vgamma = vertex_f.function_space()
    gamma = Vgamma.mesh()
    assert vertex_f.ufl_shape == ()

    assert is_terminal(v)
    assert patch_average_cell(v) == gamma.ufl_cell()

    if isinstance(v, ufl_legacy.Coefficient):
        v =  df.Function(v.function_space(), v.vector())
    else:
        # Object copy?
        v = [df.TestFunction, df.TrialFunction][v.number()](v.function_space())

    v.patch_average_ = {'vertex_f': vertex_f, 'patch_f': patch_f, 'patch_coloring': patch_coloring}

    return v


def is_patch_average_integral(integral):
    '''A point trace integral is one where some argument is a point trace.'''
    return (topological_dim(integral) == 1 and
            any(hasattr(t, 'patch_average_') for t in traverse_unique_terminals(integral.integrand())))


def patch_average_integrals(form):
    '''Extract point trace integrals from the form'''
    return list(filter(is_patch_average_integral, form.integrals()))
