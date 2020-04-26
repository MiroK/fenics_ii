from __future__ import absolute_import
from ufl.corealg.traversal import traverse_unique_terminals
from xii.assembler.ufl_utils import *
import dolfin as df
import ufl
from six.moves import filter


def injection_cell(o):
    '''The same cell type but in a different mesh'''
    # Space
    if hasattr(o, 'ufl_cell'):
        return injection_cell(o.ufl_cell())
    # Foo like
    if hasattr(o, 'ufl_element'):
        return injection_cell(o.ufl_element().cell())

    # Another cell
    return o


def injection_space(V, mesh):
    '''Construct a space where restrictions of V to mesh should live'''
    # Sanity
    assert mesh.ufl_cell() == injection_cell(V)
    # All is the same
    return df.FunctionSpace(mesh, V.ufl_element())


def Injection(v, fmesh):
    '''
    Annotated function for being a injection from space on coarse mesh 
    to space of fine mesh.
    '''
    assert is_terminal(v)
    assert injection_cell(v) == fmesh.ufl_cell()

    # Type check
    if hasattr(v, 'function_space'):
        cmesh = v.function_space().mesh()
        assert cmesh.has_child() and fmesh.has_parent()
        assert cmesh.child().id() == fmesh.id()
        assert fmesh.parent().id() == cmesh.id()

    if isinstance(v, df.Coefficient):
        v =  df.Function(v.function_space(), v.vector())
    else:
        v = [df.TestFunction, df.TrialFunction][v.number()](v.function_space())

    v.injection_ = {'mesh': fmesh}
    
    return v


def is_injection(arg):
    '''Very crude check'''
    return hasattr(arg, 'injection_')


def is_injection_integrand(expr):
    '''Is it?'''
    return any((is_injection(arg) for arg in traverse_unique_terminals(expr)))


def is_injection_integral(integral):
    '''
    Cell integral over domain that is that over domain that has child
    '''
    if not integral.integral_type() == 'cell':
        return False

    fmesh = integral.ufl_domain().ufl_cargo()
    
    if not is_injection_integrand(integral.integrand()):
        return False

    for arg in filter(is_injection, traverse_unique_terminals(integral.integrand())):
        if arg.injection_['mesh'].id() == fmesh.id():
            return True

    return False


def injection_integrals(form):
    '''Extract injection integrals from the form'''
    return list(filter(is_injection_integral, form.integrals()))
