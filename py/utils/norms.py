from dolfin import TrialFunction, TestFunction, inner, dx, grad, CellSize, avg, dot, jump, ds, dS
from dolfin import Cell, Facet, cells, facets, Constant, assemble_system, Timer
from dolfin import as_backend_type, PETScMatrix, sqrt, ln, info
from scipy.linalg import eigh
from petsc4py import PETSc
import numpy as np


def DGT_DG_error(pg, po, gamma):
    '''Compare DGT(Omega) and DG(Gamma) solutions'''
    # Assumptions + robust w.r.t arg order
    allowed = set(['Discontinuous Lagrange Trace', 'Discontinuous Lagrange'])
    g = pg.function_space().ufl_element().family()
    allowed.remove(g)
    assert allowed.pop() == po.function_space().ufl_element().family()
    if not(g == 'Discontinuous Lagrange'): pg, po = po, pg

    assert pg.function_space().ufl_element().degree() == po.function_space().ufl_element().degree()

    c2f = gamma.entity_map[1]
    mesh = po.function_space().mesh()

    Sdofm = pg.function_space().dofmap()
    Qdofm = po.function_space().dofmap()

    s_values = pg.vector().array()
    q_values = po.vector().array()
    errors = []
    for cell in cells(gamma.mesh):
        index = cell.index()
        Sdof = Sdofm.cell_dofs(index)[0]

        facet_index = c2f[index]

        mesh_cell = Cell(mesh, Facet(mesh, facet_index).entities(2)[0])
        Qdofs = Qdofm.cell_dofs(mesh_cell.index())
        for i, facet in enumerate(facets(mesh_cell)):
            if facet.index() == facet_index:
                assert cell.midpoint().distance(Facet(mesh, facet_index).midpoint()) < 1E-13
                break
        Qdof = Qdofs[i]
        e = s_values[Sdof]-q_values[Qdof]
        errors.append(e)
    # print errors
    return max(errors)


def reduced_gevp(a, bcs, m=None, eps=1E-8):
    '''
    Let a, m be symmetric bilinear forms over V which is considered with bcs. 
    We solve Find u, lmbda such that a(u, v) = lmbda*m(u, v) for each v in V
    and return the building blocks for interpolation norms.
    '''
    # First check that the problem is symmetric - this is need e.g. because then
    # the complete set of eigenvectors exists
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

    assert as_backend_type(A).mat().isHermitian(eps*ln(V.dim()))

    if M is not None:
        assert as_backend_type(M).mat().isHermitian(eps*ln(V.dim()))

    # Solve the reduced eigenvalue problem
    info('Solving GEVP of size %d' % A.size(0))
    timer = Timer('GEVP')

    if M is not None:
        eigw, eigv = eigh(A.array(), M.array())
    else:
        eigw, eigv = eigh(A.array())
    
    t = timer.stop()
    info('Done in %g' % t)

    if M is None:
        return V, (eigw, eigv)
    else:
        return V, (eigw, eigv, M.array())


class InterpolationNorm(object):
    '''
    Let a, m symmetric forms over V which is considered with bcs. Suppose these
    assemble respectively into matrices A and M. The interpolation norm of order
    s is then induced by H_s = (inv(M)*A)^s
    '''
    def __init__(self, a, bcs, m=None):
        V, args = reduced_gevp(a, bcs, m)
        self.lmbda = args[0]
        self.U = args[1]
        self.M = None if len(args) < 3 else args[2]
        self.V = V

    def get_s_norm(self, s, as_type):
        '''
        H_s norm is returned as np.ndarray/PETScMatrix or function that can be
        allied to dolfin.Functions yieldind sqrt((u, H_s U)).
        '''
        if isinstance(s, (float, int)): s = [s]

        n = len(self.lmbda)
        Lambda = np.diag(sum((self.lmbda**si for si in s), np.zeros(n)))
        U, M = self.U, self.M
        if M is not None:
            W = M.dot(U)
        else:
            W = U

        array = W.dot(Lambda.dot(W.T))

        if as_type == np.ndarray: return array

        mat = PETSc.Mat().createDense(size=len(array), array=array)
        
        if as_type == PETScMatrix: return PETScMatrix(mat)

        def norm_foo(f):
            assert f.function_space() == self.V
            
            x = as_backend_type(f.vector()).vec()
            Hs = mat.createVecLeft()
            mat.mult(x, Hs)
            return sqrt(Hs.dot(x))
        return norm_foo

    def get_s_norm_inv(self, s, as_type):
        '''inv(H_s).'''
        if isinstance(s, (float, int)): s = np.array([s])

        n = len(self.lmbda)
        Lambda = np.diag(sum((self.lmbda**si for si in s), np.zeros(n))**(-1.))
        U = self.U
        array = U.dot(Lambda.dot(U.T))

        if as_type == np.ndarray: return array

        mat = PETSc.Mat().createDense(size=len(array), array=array)
        
        if as_type == PETScMatrix: return PETScMatrix(mat)

        def norm_foo(f):
            assert f.function_space() == self.V
            
            x = as_backend_type(f.vector()).vec()
            Hs = mat.createVecLeft()
            mat.mult(x, Hs)
            return sqrt(Hs.dot(x))
        return norm_foo


class H1_L2_InterpolationNorm(InterpolationNorm):
    def __init__(self, space, f=None):
        if space.ufl_element().family() == 'Lagrange':
            u, v = TrialFunction(space), TestFunction(space)
            m = inner(u, v)*dx
            
            # Include the coefficient
            if f is None:
                a = inner(grad(u), grad(v))*dx + m
            else:
                a = inner(f*grad(u), grad(v))*dx + m

            bcs = []
        else:
            assert space.ufl_element().family() == 'Discontinuous Lagrange', space.ufl_element().family()
            assert space.ufl_element().degree() == 0

            mesh = space.mesh()
            u, v = TrialFunction(space), TestFunction(space)
            h = CellSize(mesh)
            h_avg = avg(h)

            m = inner(u, v)*dx
            
            if f is None:
                a = h_avg**(-1)*dot(jump(v), jump(u))*dS + h**(-1)*dot(u, v)*ds
            else:
                assert False, 'How should the coefs be included?'
            a = a + m
            bcs = []
        InterpolationNorm.__init__(self, a, bcs=bcs, m=m)

# ----------------------------------------------------------------------------

if __name__ == '__main__':
    from dolfin import *

    mesh = UnitIntervalMesh(100)
    V = FunctionSpace(mesh, 'CG', 1)
    u = TrialFunction(V)
    v = TestFunction(V)

    a = inner(grad(u), grad(v))*dx
    m = inner(u, v)*dx
    L = inner(Constant(0), v)*dx
    bc = DirichletBC(V, Constant(0), 'on_boundary')
    
    A, _ = assemble_system(a, L, bc)
    M, _ = assemble_system(m, L, bc)

    I = InterpolationNorm(a, bcs=bc, m=m)

    # s=1
    ## as mat
    print np.linalg.norm(A.array() - I.get_s_norm(s=1, as_type=np.ndarray)) 
    ## as PETScMatrix
    A0 = I.get_s_norm(s=1, as_type=PETScMatrix)
    f = Function(V)
    x = f.vector(); x[:] = np.random.rand(V.dim())
    y, y0 = Function(V).vector(), Function(V).vector()

    A.mult(x, y)
    A0.mult(x, y0)
    print (y-y0).norm('linf')
    ## as foo
    value0 = sqrt(y.inner(x))
    value = I.get_s_norm(s=1, as_type=None)(f)
    print abs(value - value0)

    # s=0
    ## as mat
    print np.linalg.norm(M.array() - I.get_s_norm(s=0, as_type=np.ndarray)) 
    ## as PETScMatrix
    M0 = I.get_s_norm(s=0, as_type=PETScMatrix)
    x = Function(V).vector(); x[:] = np.random.rand(V.dim())
    y, y0 = Function(V).vector(), Function(V).vector()

    M.mult(x, y)
    M0.mult(x, y0)
    print (y-y0).norm('linf')

    # s=0.5
    ## as mat inv
    B = I.get_s_norm(s=0.5, as_type=np.ndarray)
    Binv = I.get_s_norm_inv(s=0.5, as_type=np.ndarray)
    print np.linalg.norm(B.dot(Binv) - np.eye(len(B)))
    # Check action and inverse action
    B = I.get_s_norm(s=0.5, as_type=PETScMatrix)
    Binv = I.get_s_norm_inv(s=0.5, as_type=PETScMatrix)
    x = Function(V).vector(); x.set_local(np.random.rand(V.dim()))
    BinvBx = x.copy()
    print x.norm('linf'), BinvBx.norm('linf')

    Bx = Function(V).vector()
    B.mult(x, Bx)
    Binv.mult(Bx, x)

    print (BinvBx - x).norm('linf')

    H1_L2_InterpolationNorm(V)
    H1_L2_InterpolationNorm(FunctionSpace(mesh, 'DG', 0))

    # Sums
    B1 = I.get_s_norm(s=0.5, as_type=np.ndarray)
    B2 = I.get_s_norm(s=0.25, as_type=np.ndarray)
    B12 = I.get_s_norm(s=[0.5, 0.25], as_type=np.ndarray)
    print np.linalg.norm(B1 + B2 - B12)

    B12inv = I.get_s_norm_inv(s=[0.5, 0.25], as_type=np.ndarray)
    print np.linalg.norm(B12.dot(B12inv) - np.eye(len(B1)))

