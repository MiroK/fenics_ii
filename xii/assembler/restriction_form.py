from ufl_legacy.corealg.traversal import traverse_unique_terminals
from xii.assembler.ufl_utils import *
from xii.meshing.subdomain_mesh import SubDomainMesh
import dolfin as df
import ufl_legacy as ufl


def restriction_cell(o):
    '''The same cell type but in a different mesh'''
    # Space
    if hasattr(o, 'ufl_cell'):
        return restriction_cell(o.ufl_cell())
    # Foo like
    if hasattr(o, 'ufl_element'):
        return restriction_cell(o.ufl_element().cell())

    # Another cell
    return o


def restriction_space(V, mesh):
    '''Construct a space where restrictions of V to mesh should live'''
    # Sanity
    assert mesh.ufl_cell() == restriction_cell(V)
    # All is the same
    return df.FunctionSpace(mesh, V.ufl_element())


def Restriction(v, mmesh):
    '''
    Annotated function for being a restriction onto mmesh which is must 
    be a SubDomainMesh
    '''
    # Prevent Restriction(grad(u)). But it could be interesting to have this
    assert is_terminal(v)
    assert restriction_cell(v) == mmesh.ufl_cell()
    # assert isinstance(mmesh, SubDomainMesh)

    if isinstance(v, ufl.Coefficient):
        v =  df.Function(v.function_space(), v.vector())
    else:
        # Object copy?
        v = [df.TestFunction, df.TrialFunction][v.number()](v.function_space())

    v.restriction_ = {'mesh': mmesh}
    
    return v


def is_restriction(arg):
    '''Very crude check'''
    return hasattr(arg, 'restriction_')


def is_restriction_integrand(expr):
    '''Is it?'''
    return any((is_restriction(arg) for arg in traverse_unique_terminals(expr)))


def is_restriction_integral(integral):
    '''
    There is a restriction one some of the integrands
    '''
    return all((integral.integral_type() == 'cell',
                is_restriction_integrand(integral.integrand())))


def restriction_integrals(form):
    '''Extract restriction integrals from the form'''
    return list(filter(is_restriction_integral, form.integrals()))
