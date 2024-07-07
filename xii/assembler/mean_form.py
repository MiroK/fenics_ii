from xii.assembler.ufl_utils import *

from ufl.corealg.traversal import traverse_unique_terminals
import dolfin as df
import ufl


def mean_cell(o):
    '''
    UFL cell corresponding to mean of o over domain where it is defefined
    '''
    # Space
    if hasattr(o, 'ufl_cell'):
        return mean_cell(o.ufl_cell())
    # Foo like
    if hasattr(o, 'ufl_element'):
        return mean_cell(o.ufl_element().cell())
    # Another cell
    return o


def mean_space(V, mesh, data):
    '''Construct a space over mesh where surface averages of V should live'''
    assert mesh.ufl_cell() == mean_cell(V)
    # FIXME: lift this later
    assert V.ufl_element().value_shape() == ()

    return df.FunctionSpace(mesh, 'Real', 0)


def Mean(v):
    '''v -> |domain|^{-1} * \int_{domain} v *dx '''
    # FIXME: Don't want to deal with vectors at this point
    assert v.ufl_shape == ()
    assert is_terminal(v)

    if isinstance(v, ufl.Coefficient):
        v =  df.Function(v.function_space(), v.vector())
    else:
        # Object copy?
        v = [df.TestFunction, df.TrialFunction][v.number()](v.function_space())

    v.mean_ = {}

    return v


def is_mean_integral(integral):
    '''A point trace integral is one where some argument is a point trace.'''
    return any(hasattr(t, 'mean_') for t in traverse_unique_terminals(integral.integrand()))


def mean_integrals(form):
    '''Extract point trace integrals from the form'''
    return list(filter(is_mean_integral, form.integrals()))
