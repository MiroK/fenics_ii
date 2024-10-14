from ufl_legacy.corealg.traversal import traverse_unique_terminals
from xii.assembler.ufl_utils import *
import dolfin as df
import numpy as np
import ufl_legacy as ufl


def point_trace_cell(o):
    '''The same cell'''
    return o


def point_trace_space(V, mesh):
    '''Space from point trace values live - these are just R^n'''
    shape = V.ufl_element().value_shape()
    # Scalars
    if shape == ():
        return df.FunctionSpace(mesh, 'R', 0)
    # Elsewhere only allow vectors
    if len(shape) == 1:
        assert isinstance(V.ufl_element(), ufl.VectorElement)
        return df.VectorFunctionSpace(mesh, 'R', 0, shape[0])
    # Or tensors
    if len(shape) == 2:
        assert isinstance(V.ufl_element(), ufl.TensorElement)
        return df.TensorFunctionSpace(mesh, 'R', 0, shape)
    

def PointTrace(v, point, cell):
    '''Annotatef v copy for being a point trace at point'''
    # Prevent Restriction(grad(u)). But it could be interesting to have this
    assert is_terminal(v)
    # FIXME: the point trace mat logic works only for spacec with point eval
    # dofs
    assert v.ufl_element().family() in ('Lagrange', 'Discontinuous Lagrange')
    # Don't allow point because then it's difficult to check len
    assert isinstance(point, (int, np.int32, np.int64, np.uint32, np.uint64))
    assert isinstance(cell, (int, np.int32, np.int64, np.uint32, np.uint64)), type(cell)

    # A copy!
    v = reconstruct(v)
    v.dirac_ = {'point': point, 'cell': cell}

    return v


def is_point_trace_integral(integral):
    '''A point trace integral is one where some argument is a point trace.'''
    return any(hasattr(t, 'dirac_') for t in traverse_unique_terminals(integral.integrand()))


def point_trace_integrals(form):
    '''Extract point trace integrals from the form'''
    return list(filter(is_point_trace_integral, form.integrals()))
