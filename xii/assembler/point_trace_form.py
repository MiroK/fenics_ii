from ufl.corealg.traversal import traverse_unique_terminals
from xii.assembler.ufl_utils import *
import dolfin as df
import numpy as np
import ufl


def point_trace_cell(o):
    '''The same cell'''
    return o


def point_trace_space(V, mesh):
    '''Space from point trace values live - these are just R^n'''
    # Fow now only allow scalars
    assert V.ufl_element().value_shape() == ()
    return df.FunctionSpace(mesh, 'R', 0)


def PointTrace(v, mmesh):
    '''Annotatef v copy for being a point trace at point'''
    # Prevent Restriction(grad(u)). But it could be interesting to have this
    assert is_terminal(v)
    # Don't allow point because then it's difficult to check len
    assert isinstance(mmesh, (list, tuple, np.ndarray))
    # A copy!
    v = (df.TrialFunction if v.number() == 1 else df.TestFunction)(v.function_space())
    v.dirac_ = {'point': mmesh}

    return v


def is_point_trace_integral(integral):
    '''A point trace integral is one where some argument is a point trace.'''
    return any(hasattr(t, 'dirac_') for t in traverse_unique_terminals(integral.integrand()))


def point_trace_integrals(form):
    '''Extract point trace integrals from the form'''
    return filter(is_point_trace_integral, form.integrals())
