# Average normal flux over shape

from xii.assembler.ufl_utils import *
from xii.linalg.matrix_utils import is_number

from ufl_legacy.corealg.traversal import traverse_unique_terminals
import dolfin as df
import ufl_legacy as ufl


def flux_average_cell(o):
    '''1d'''
    # Space
    if hasattr(o, 'ufl_cell'):
        return flux_average_cell(o.ufl_cell())
    # Foo like
    if hasattr(o, 'ufl_element'):
        return flux_average_cell(o.ufl_element().cell())

    # Another cell
    cell_name = {'tetrahedron': 'interval'}[o.cellname()]
    
    return ufl.Cell(cell_name, o.geometric_dimension())


def flux_average_space(V, mesh):
    '''Construct a space over mesh where surface averages of V should live'''
    # Sanity
    assert mesh.ufl_cell() == flux_average_cell(V)

    elm = V.ufl_element()
    family = elm.family()
    degree = elm.degree()
    assert len(elm.value_shape()) == 1, (elm.value_shape(), )
    
    # Since tangent of a mesh cell/segment is uniquely defined only inside
    # the cell it is natural to represent everything in DG
    family = 'Discontinuous Lagrange'
    # v.n will always give scalar
    return df.FunctionSpace(mesh, 'DG', degree)


def FluxAverage(v, line_mesh, shape, normalize=True):
    '''Average v.n over shape'''
    assert is_terminal(v)
    assert flux_average_cell(v) == line_mesh.ufl_cell()
    assert shape is not None
        
    if isinstance(v, ufl.Coefficient):
        # FIXME: this is here because tying to coefficients of v makes
        # no sense since we will change the space
        raise ValueError
    else:
        V_ = v.function_space()
        V = df.FunctionSpace(V_.mesh(), 'DG', V_.ufl_element().degree())
        # v.n is a scalar so
        v = [df.TestFunction, df.TrialFunction][v.number()](V)

    v.flux_average_ = {'mesh': line_mesh, 'shape': shape, 'domain': V_, 'normalize': normalize}

    return v


def is_flux_average_integrand(expr, tdim):
    '''Some of the arguments need restriction'''
    return any((topological_dim(arg) == tdim + 2) and hasattr(arg, 'flux_average_')
               for arg in traverse_unique_terminals(expr))


def is_flux_average_integral(integral):
    '''Volume integral over an embedded line cell'''
    return is_flux_average_integrand(integral.integrand(), topological_dim(integral))


def flux_average_integrals(form):
    '''Extract trace integrals from the form'''
    return list(filter(is_flux_average_integral, form.integrals()))
