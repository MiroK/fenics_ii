from xii.linalg.matrix_utils import petsc_serial_matrix
from xii.linalg.convert import convert
import xii.assembler.trace_matrix as trace_m
import xii.assembler.trace_form as trace_f
from dolfin import (PETScMatrix, Point, Cell, TrialFunction, TestFunction,
                    inner, assemble, cells, Constant, dx)
from petsc4py import PETSc
import numpy as np


# Restriction operators are potentially costly so we memoize the results.
# Let every operator deal with cache keys as it sees fit
def memoize_trace(trace_mat):
    '''Cached trace'''
    cache = {}
    def cached_trace_mat(V, TV, trace_mesh, data):
        key = ((V.ufl_element(), V.mesh().id()),
               (TV.ufl_element(), TV.mesh().id()))
               
        if key not in cache:
            cache[key] = trace_mat(V, TV, trace_mesh, data)
        return cache[key]

    return cached_trace_mat


@memoize_trace
def surface_mean_mat(V, TV, trace_mesh, data):
    '''Representation of the operator'''
    assert TV.ufl_element().family() == 'Real'
    assert V.ufl_element().value_shape() == TV.ufl_element().value_shape()

    assert TV.mesh().id() == trace_mesh.id()

    Tmat = surface_mean_matrix(V, TV, trace_mesh)
    return Tmat
                

def surface_mean_matrix(V, Q, surface_mesh):
    '''We want to represent the mapping V -> TV -> R'''
    TV = trace_f.trace_space(V, surface_mesh)
    
    T_2_TV = PETScMatrix(trace_m.trace_mat_no_restrict(V, TV, surface_mesh))

    u, q = TrialFunction(TV), TestFunction(Q)
    surface_area = sum(c.volume() for c in cells(surface_mesh))
    M = assemble(Constant(1/surface_area)*inner(u, q)*dx)


    return convert(M*T_2_TV)
