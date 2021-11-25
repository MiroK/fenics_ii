from block.algebraic.petsc import LU
from block.block_base import block_base
import functools
from dolfin import *
import numpy as np


def orthonormalize(basis, M, tol=1E-10):
    '''M-orthonormal basis'''
    # Assuming symmetry
    n = len(basis)
    G = np.zeros((n, n))
    for i, zi in enumerate(basis):
        G[i, i] = zi.inner(M*zi)
        for j in range(i, n):
            zj = basis[j]
            G[i, j] = zi.inner(M*zj)
            G[j, i] = G[i, j]

    vals, vecs = np.linalg.eig(G)
    # Symmatric matrices have real spectra so something would be wrong if
    assert np.all(np.abs(vals.imag) < tol)
    vals = vals.real
    # The matrix should have been an inner product so 
    assert np.all(vals > 0)
    # Want invariant subspaces of dim 1 for each eigenvalue, so unique eigenvalues
    assert len(np.unique(np.round(vals, int(-np.log10(tol))))) == len(vals)

    vecs = vecs.T 

    # Recombine according to eignvectors to get orthonormal basis
    return [(1/sqrt(val))*sum(ak*zk for ak, zk in zip(alphas, basis))
            for val, alphas in zip(vals, vecs)]
    

class Projector(block_base):
    '''P*u = u - (z, u)*z where (,) is the L^2 inner product'''
    def __init__(self, V, nullspace):
        basis = [interpolate(z, V).vector() for z in nullspace]

        u, v = TrialFunction(V), TestFunction(V)
        M = assemble(inner(u, v)*dx)

        basis = orthonormalize(basis, M)

        block_base.__init__(self)

        self.primal_basis = basis  # Primal vectors
        self.dual_basis = [M*z for z in basis]  # NOTE: <pi, dj> = id
        self.M = M

    def matvec(self, x):
        '''Project primal vector returning primal vector'''
        basis = zip(self.primal_basis, self.dual_basis)
        M = self.M
        # Compute part in the nullspace
        z, Mz = next(basis)
        y = (Mz.inner(x))*z
        for z, Mz in basis:
            y.axpy(Mz.inner(x), z)
        # And subtract
        return x - y

    @functools.lru_cache    
    def collapse(self):
        '''Explicit matrix representation of the operator'''
        from xii.linalg.matrix_utils import diagonal_matrix, row_matrix
        from xii import ii_convert

        D = row_matrix(self.dual_basis)
        B = row_matrix(self.primal_basis)
        I = diagonal_matrix(B.size(1), 1.)
        return ii_convert(I - B.T*D)

    
class L20_inner_product(block_base):
    '''
    Operator such that  <v, L20*u> = (u - (z,u)*z, v - (z, v)*z) 
    where (,) is L^2 inner product and z the nullspace vectors
    '''
    def __init__(self, V, nullspace):
        block_base.__init__(self)
        self.P = Projector(V, nullspace)

    def matvec(self, x):
        '''the matrix representation is M - (M*Z)*Z.T*M'''
        return (self.P.M)*self.P*x

    @functools.lru_cache    
    def collapse(self):
        '''Explicit matrix representation of the operator'''
        from xii import ii_convert
        
        return ii_convert((self.P.M)*self.P.collapse())

    
class L20_riesz_map(block_base):
    '''Invert L20 inner product'''
    def __init__(self, V, nullspace, inv=LU):
        block_base.__init__(self)
        self.P = Projector(V, nullspace)
        self.iM = inv(self.P.M)
        # M - (M*Z)*Z.T*M = M*P and so we want inv(P)*inv(M)
        # inv(P) uses woodbory formula applied to id - outer(pi, di)
        #
        #   inv(id - outer(pi, di)) = id + outer(pi, di)/(1-sum(inner(pi, di)))
        #
        self.primal_basis = self.P.primal_basis
        self.dual_basis = self.P.dual_basis
        # The denominator 1-sum(inner(pi, di)) can be precomputed and since
        # inner(pi, di) = 1 we have
        self.scale = 1 - len(self.primal_basis)

    def matvec(self, x):
        '''Sherman-Woodburry'''
        y = self.iM*x  # From dual to primal
        basis = zip(self.primal_basis, self.dual_basis)

        pi, di = next(basis)
        w = di.inner(y)*pi
        for pi, di in basis:
            w.axpy(di.inner(y), pi)
        w *= (1/self.scale)
        
        return x + w
    
# -------------------------------------------------------------------

if __name__ == '__main__':
    mesh = UnitSquareMesh(32, 32)
    nullspace = (Constant(1),
                 Expression('x[0]', degree=1),
                 Expression('cos(pi*x[0])', degree=3),
                 # Expression('x[0]*x[1]', degree=2),
                 Expression('x[1]', degree=1))

    V = FunctionSpace(mesh, 'CG', 2)
    P = Projector(V, nullspace)
     
    f = Expression('sin(pi*x[0])*x[1]', degree=3)
    assert all(abs(assemble(inner(f, z)*dx(domain=mesh))) > 0
               for z in nullspace)
    
    fh = interpolate(f, V)
    Pfh = Function(V)
    Pfh.vector()[:] = P*fh.vector()
    assert all(abs(assemble(inner(Pfh, z)*dx(domain=mesh))) < 1E-10
               for z in nullspace), [abs(assemble(inner(Pfh, z)*dx(domain=mesh))) for z in nullspace]

    L20 = L20_inner_product(V, nullspace)
    matL20 = L20.collapse()
    assert (norm(L20*fh.vector() - matL20*fh.vector())) < 1E-13

    x = L20*Pfh.vector()
    assert norm(x) > 0
    iL20 = L20_riesz_map(V, nullspace)
    y = iL20*x  # Apply some operator that is "blind" on the nullspace
    # Check inverse
    assert norm(x-y) < 1E-13
