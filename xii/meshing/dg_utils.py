import dolfin as df
import numpy as np


def pcws_constant(marker):
    '''Turn cell function to P0'''
    assert isinstance(marker, df.cpp.mesh.MeshFunctionSizet)
    mesh = marker.mesh()
    # We want a cell function
    assert mesh.topology().dim() == marker.dim()

    V = df.FunctionSpace(mesh, 'DG', 0)
    c2d = V.dofmap().entity_dofs(mesh, marker.dim())

    f = df.Function(V)
    values = f.vector().get_local()
    values[c2d] = np.asarray(marker.array(), dtype=float)
    f.vector().set_local(values)
    
    return f


def OrientedFacetNormal(K):
    '''Grab the normal from K'''
    if isinstance(K, df.cpp.mesh.MeshFunctionSizet):
        K = pcws_constant(K)
        return OrientedFacetNormal(K)

    mesh = K.function_space().mesh()
    n = df.FacetNormal(mesh)
    return orient_facet_normal(n, K)


def orient_facet_normal(n, K):
    '''
    Given a subdomain marking orient facet normals to point from higher
    to lower tag on the interfaces
    '''
    if isinstance(K, df.cpp.mesh.MeshFunctionSizet):
        K = pcws_constant(K)
        return orient_facet_normal(n, K)

    mesh = K.function_space().mesh()
    assert n.ufl_domain().ufl_cargo().id() == mesh.id()

    V = df.VectorFunctionSpace(mesh, 'Discontinuous Lagrange Trace', 0)
    v = df.TestFunction(V)
    hA = df.FacetArea(mesh)

    cond_n = df.conditional(df.gt(K('+'), K('-')), n('+'), n('-'))
    form = (1/hA('+'))*df.inner(v('+'), cond_n)*df.dS + (1/hA)*df.inner(v, n)*df.ds

    n = df.Function(V)
    df.assemble(form, tensor=n.vector())

    return n
