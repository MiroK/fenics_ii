from __future__ import absolute_import
from xii.assembler.trace_form import Trace
import xii.assembler.xii_assembly 
from xii.linalg.bc_apply import apply_bc
from xii.linalg.fe_space_op import FESpaceOperator

from xii.linalg.convert import numpy_to_petsc
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
import dolfin as df
import numpy as np
from six.moves import map


# Extension operators are potentially costly so we memoize the results.
# Let every operator deal with cache keys as it sees fit
def memoize_ext(ext_mat):
    '''Cached extension mapping'''
    cache = {}
    def cached_ext_mat(V, TV, extended_mesh, data):
        key = ((V.ufl_element(), V.mesh().id()),
               (TV.ufl_element(), TV.mesh().id()),
               data['type'])

        if data['data'] is not None:
            key = sum(list(data['data'].items()), key)
            
        if key not in cache:
            cache[key] = ext_mat(V, TV, extended_mesh, data)
        return cache[key]

    return cached_ext_mat


@memoize_ext
def extension_mat(V, EV, extended_mesh, data):
    '''
    Dispatch to individual methods for extending functions from V to 
    EV(on extended_mesh)
    '''
    assert EV.mesh().id() == extended_mesh.id()
    
    assert V.mesh().topology().dim() == 1
    assert EV.mesh().topology().dim() in (1, 2)  # Line to line or line to triangle

    assert V.ufl_element().degree() == EV.ufl_element().degree()
    assert V.ufl_element().family() == EV.ufl_element().family()
    assert V.mesh().geometry().dim() == EV.mesh().geometry().dim()
    assert V.mesh().geometry().dim() in (2, 3)

    # NOTE: add more here
    # - something based on radial basis function
    # - using the method of Green's functions
    if data['type'] == 'uniform':
        return uniform_extension_matrix(V, EV)
    elif data['type'] == 'harmonic':
        return harmonic_extension_operator(V, EV, data['data']['aux_facet_f'])


def uniform_extension_matrix(V, EV):
    '''
    Map vector of coeficients of V(over 1d domain) to vector of coefficients of 
    EV(over 2d domain). The spaces need to use the same element type.
    '''
    gdim = V.mesh().geometry().dim()
    assert gdim == EV.mesh().geometry().dim()
    
    # For Vector and Tensor elements more than 1 degree of freedom is 
    # associated with the same geometric point. It is therefore cheaper
    # to compute the mapping based only on the scalar/one subspace considerations.
    is_tensor_elm = isinstance(V.ufl_element(), (df.VectorElement, df.TensorElement))
    # Base on scalar
    if is_tensor_elm:
        V_dofs_x = V.sub(0).collapse().tabulate_dof_coordinates().reshape((-1, gdim))
        EV_dofs_x = EV.sub(0).collapse().tabulate_dof_coordinates().reshape((-1, gdim))
    # Otherwise 'scalar', (Hdiv element belong here as well)
    else:
        V_dofs_x = V.tabulate_dof_coordinates().reshape((V.dim(), gdim))
        EV_dofs_x = EV.tabulate_dof_coordinates().reshape((EV.dim(), gdim))
        
    # Compute distance from every EV dof(row) to every V dof(column)
    lookup = cdist(EV_dofs_x, V_dofs_x)
    # Make sure the two domains do not intersect
    assert np.linalg.norm(lookup, np.inf) > 0

    # Now get the closest dof to E
    columns = np.argmin(lookup, axis=1)
    # Every scalar can be used used to set all the components
    if is_tensor_elm:
        shift = V.dim()/len(V_dofs_x)
        component_idx = np.arange(shift)
        # shift*dof + components
        columns = (shift*np.array([columns]).T + component_idx).flatten()

    # As csr (1 col per row)
    values = np.ones_like(columns)
    rows = np.arange(EV.dim()+1)

    E = csr_matrix((values, columns, rows), shape=(EV.dim(), V.dim()))

    return numpy_to_petsc(E)


def harmonic_extension_operator(V, EV, auxiliary_facet_f):
    '''
    Given function f in V the extension E(f) in EV is obtained as a trace 
    to EV's mesh of uh where uh solves

      -Delta u_h = delta_gamma*f on Omega,
    
    Here gamma is the mesh of V and Omega is a submesh of the background 
    mesh. Its facets are marked with `auxliary_facet_f` in such a way 
    that 1 is a domain of homog. Dirichlet boundary conditions.
    '''
    # For now this shall only work in 2d
    assert V.mesh().geometry().dim() == 2 and EV.mesh().geometry().dim() == 2
    
    aux_mesh = auxiliary_facet_f.mesh()
    assert aux_mesh.geometry().dim() == 2
    assert 1 in set(auxiliary_facet_f.array())

    class FOO(FESpaceOperator):
        '''Extend from V to VE'''
        def __init__(self, V1=V, V0=EV, facet_f=auxiliary_facet_f):
            FESpaceOperator.__init__(self, V1, V0)
            self.facet_f = auxiliary_facet_f
            
        def matvec(self, b):
            '''Action on vector from V1'''
            V, Q = self.V1, self.V0
            
            gamma_mesh = V.mesh()
            auxiliary_facet_f = self.facet_f
            aux_mesh = auxiliary_facet_f.mesh()
            
            # The problem for uh
            V2 = df.FunctionSpace(aux_mesh, Q.ufl_element().family(), Q.ufl_element().degree())
            u, v = df.TrialFunction(V2), df.TestFunction(V2)

            # Wrap the vector as function ...
            f = df.Function(V, b)
            # ... to be used in the weak form for the Laplacian
            a = df.inner(df.grad(u), df.grad(v))*df.dx + df.inner(u, v)*df.dx
            L = df.inner(f, Trace(v, gamma_mesh))*df.dx(domain=gamma_mesh)

            A, b = list(map(xii.assembler.xii_assembly.assemble, (a, L)))
            # We have boundary conditions to apply
            # bc = df.DirichletBC(V2, df.Constant(0), auxiliary_facet_f, 1)
            # A, b = apply_bc(A, b, bc)

            uh = df.Function(V2)
            df.solve(A, uh.vector(), b)

            # Now take trace of that
            ext_mesh = Q.mesh()
            # Get the trace at extended domain by projection
            p, q = df.TrialFunction(Q), df.TestFunction(Q)
            
            f = Trace(uh, ext_mesh)
            a = df.inner(p, q)*df.dx
            L = df.inner(f, q)*df.dx(domain=ext_mesh)

            A, b  = list(map(xii.assembler.xii_assembly.assemble, (a, L)))
            # We have boundary conditions to apply
            # FIXME: inherit from uh?
            # bc = df.DirichletBC(Q, uh, 'on_boundary')
            # A, b = apply_bc(A, b, bc)

            qh = df.Function(Q)
            df.solve(A, qh.vector(), b)

            return qh.vector()
    # And the intance of that
    return FOO().collapse()  # So that we get the matrix and don't have
                             # to deal with transpose

# -------------------------------------------------------------------

if __name__ == '__main__':
    from dolfin import *
    from xii import EmbeddedMesh, StraightLineMesh
    import numpy as np

    set_log_level(WARNING)

    nx = 63
    ny = (nx+1)/2
    A, B = (nx-1)/2./nx, (nx+1)/2./nx

    mesh = UnitSquareMesh(nx, ny)

    cell_f = MeshFunction('size_t', mesh, 2, 0)
    CompiledSubDomain('x[0] > A - tol && x[0] < B + tol', A=A, B=B, tol=DOLFIN_EPS).mark(cell_f, 1)

    left = CompiledSubDomain('A-tol < x[0] && x[0] < A+tol', A=A, tol=DOLFIN_EPS)
    right = CompiledSubDomain('A-tol < x[0] && x[0] < A+tol', A=B, tol=DOLFIN_EPS)

    # We would be extending to
    facet_f = MeshFunction('size_t', mesh, 1, 0)
    left.mark(facet_f, 1)
    right.mark(facet_f, 1)

    ext_mesh = EmbeddedMesh(facet_f, 1)
    EV = FunctionSpace(ext_mesh, 'CG', 1)

    # The auxiliary problem would be speced at
    aux_mesh = SubMesh(mesh, cell_f, 1)
    facet_f = MeshFunction('size_t', aux_mesh, 1, 0)
    DomainBoundary().mark(facet_f, 1)
    left.mark(facet_f, 2)
    right.mark(facet_f, 2)

    # Extending from
    gamma_mesh = StraightLineMesh(np.array([0.5, 0]), np.array([0.5, 1]), 3*ny)
    V = FunctionSpace(gamma_mesh, 'CG', 1)

    E = harmonic_extension_operator(V, EV, auxiliary_facet_f=facet_f)

    v = interpolate(Constant(1), V).vector()
    q = Function(EV, E*v)
    File('foo.pvd') << q
    
    from xii.linalg.convert import collapse
    E_ = collapse(E)

    q = Function(EV, E_*v)
    File('foo_.pvd') << q


    

