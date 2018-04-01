from ufl.corealg.traversal import traverse_unique_terminals
from xii.assembler.ufl_utils import *
from xii.meshing.subdomain_mesh import SubDomainMesh
import dolfin as df
import ufl


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
    assert isinstance(mmesh, SubDomainMesh)
    # A copy!
    v = reconstruct(v)
    v.restriction_ = {'mesh': mmesh}

    return v


def is_restriction_integral(integral):
    '''
    A restriction integral here is:
    1) the measure is defined over a subdomain mesh
    2) in the parent map of the mesh we find a mesh of at least on argument
       of the integrand
    '''
    # Domain of the measure
    restriction_domain = integral.ufl_domain().ufl_cargo()
    if not isinstance(restriction_domain, SubDomainMesh):
        return False
    
    mapping = restriction_domain.parent_entity_map

    for t in traverse_unique_terminals(integral.integrand()):
        # Of argument
        domain = t.ufl_domain()
        if domain is not None:
            mesh = domain.ufl_cargo()
            if mesh.id() in mapping:
                return True
    return False


def restriction_integrals(form):
    '''Extract restriction integrals from the form'''
    return filter(is_restriction_integral, form.integrals())
