from dolfin import *
from petsc4py import PETSc
from scipy.linalg import eigh
import numpy as np


def eig(a, m=None, bcs=[], eps=1E-8):
    '''
    Solve the (generalized) eigenvalue problem a(u, v) = lmbda m(u, v) where a,
    m are symmetric bilinear form over V with bcs.
    '''
    # Assumptions
    V = a.arguments()[0].function_space()
    assert V == a.arguments()[1].function_space()
    if m is not None: assert all(V == v.function_space() for v in m.arguments())
    n = V.dim()

    v = TestFunction(V)

    L = inner(v, Constant(np.zeros(v.ufl_shape)))*dx

    A, _ = assemble_system(a, L, bcs)
    if m is not None:
        M, _ = assemble_system(m, L, bcs)
    else:
        M = None

    Amat = as_backend_type(A).mat()
    assert Amat.isHermitian(eps*ln(V.dim()))
    A = A.array()

    if M is not None:
        Mmat = as_backend_type(M).mat()
        assert Mmat.isHermitian(eps*ln(V.dim()))
        M = M.array()

    # Now the eigenvalue problem
    info('\tSolving (G)EVP of size %d' % V.dim())
    timer = Timer('(G)EVP')
    if M is not None:
        eigw, eigv = eigh(A, M)
    else:
        eigw, eigv = eigh(A)
    info('\tDone in %g' % timer.stop())

    return (eigw, eigv) if M is None else (eigw, eigv, M)


def interpolation_norm_eig(a, s, m=None, bcs=[]):
    '''
    Given symmetric [positive-definite] bilinear forms a (m) over V considered 
    with bcs construct the matrix which representes and interpolation norm [m,
    a]_s.

    By spectral computations
    '''
    if m is None:
        lmbda, U = eig(a, m, bcs)
        Lmbda = np.diag(lmbda**s)
        Aarray = U.dot(Lmbda.dot(U.T))
    else:
        lmbda, U, M = eig(a, m, bcs)
        Lmbda = np.diag(lmbda**s)
        U = M.dot(U)
        Aarray = U.dot(Lmbda.dot(U.T))

    A = PETSc.Mat().createDense(size=Aarray.shape, array=Aarray)
    return PETScMatrix(A)


def inv_interpolation_norm_eig(a, s, m=None, bcs=[]):
    '''
    Inverse of the interpolation norm
    By spectral computations
    '''
    if m is None:
        return interpolation_norm_eig(a, -s, bcs=bcs)
    else:
        lmbda, U, M = eig(a, m, bcs)
        Lmbda = np.diag(lmbda**(-s))
        Aarray = U.dot(Lmbda.dot(U.T))

        A = PETSc.Mat().createDense(size=Aarray.shape, array=Aarray)

        return PETScMatrix(A)

# ----------------------------------------------------------------------------

if __name__ == '__main__':
    mesh = UnitSquareMesh(10, 10)
    V = FunctionSpace(mesh, 'CG', 1)
    u = TrialFunction(V)
    v = TestFunction(V)

    a = inner(grad(u), grad(v))*dx
    m = inner(u, v)*dx
    L = inner(Constant(0), v)*dx
    bc = DirichletBC(V, Constant(0), 'on_boundary')

    A, _ = assemble_system(a, L, bc)
    M, _ = assemble_system(m, L, bc)

    N1 = interpolation_norm_eig(a, s=1., m=m, bcs=bc)
    N0 = interpolation_norm_eig(a, s=0, m=m, bcs=bc)

    n = np.linalg.norm(N1.array() - A.array()); assert n < 1E-10
    n = np.linalg.norm(N0.array() - M.array()); assert n < 1E-10

    Nhalf = interpolation_norm_eig(a, s=0.5, m=m, bcs=bc).array()
    invNhalf = inv_interpolation_norm_eig(a, s=0.5, m=m, bcs=bc).array()

    n = np.linalg.norm(Nhalf.dot(invNhalf) - np.eye(V.dim())); assert n < 1E-10
    n = np.linalg.norm(invNhalf.dot(Nhalf) - np.eye(V.dim())); assert n < 1E-10
