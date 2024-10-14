from ufl_legacy.corealg.traversal import traverse_unique_terminals
import xii.assembler.trace_form as trace
from xii.assembler.ufl_utils import *
import dolfin as df
import numpy as np
import ufl_legacy as ufl


def surface_mean_cell(o):
    '''Will live on the surface and in this respect is a trace'''
    return trace.trace_cell(o)


def surface_mean_element(elm):
    '''
    Produce an intermerdiate element for computing with surface mean of 
    functions in FEM space over elm
    '''
    assert elm.value_shape() == ()

    cell = surface_mean_cell(elm)

    return df.FiniteElement('Real', cell, 0)


def surface_mean_space(V, mesh):
    '''Single DOF for to represent it'''    
    return df.FunctionSpace(mesh, surface_mean_element(V.ufl_element()))


def SurfaceMean(v, mmesh):
    '''Normalized by the surface area in order to preserve unit of v'''
    assert is_terminal(v)

    assert surface_mean_cell(v) == mmesh.ufl_cell()
    
    if isinstance(v, ufl.Coefficient):
        v =  df.Function(v.function_space(), v.vector())
    else:
        # Object copy?
        v = [df.TestFunction, df.TrialFunction][v.number()](v.function_space())

    v.surface_mean_ = {'mesh': mmesh}

    return v


def is_surface_mean_integrand(expr, tdim):
    '''Some of the arguments need restriction'''
    return any((topological_dim(arg)-1)  == tdim
               for arg in traverse_unique_terminals(expr))


def is_surface_mean_integral(integral):
    '''Volume integral over an embedded cell'''
    return all((integral.integral_type() == 'cell',
                is_surface_mean_integrand(integral.integrand(), topological_dim(integral))))


def surface_mean_integrals(form):
    '''Extract surface_mean integrals from the form'''
    return list(filter(is_surface_mean_integral, form.integrals()))
