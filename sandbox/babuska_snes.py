# Nonlinear Babuska
# -div(nl(u)*grad(u)) = f in [0, 1]^2
#               T(u) = h on boundary
#

from dolfin import *
from xii import *
from xii.linalg.convert import set_lg_map
from ulfy import Expression
from petsc4py import PETSc


def nonlinear_babuska(N, u_exact, p_exact):
    '''MMS for the problem'''
    
    mesh = UnitSquareMesh(N, N)
    bmesh = BoundaryMesh(mesh, 'exterior')

    V = FunctionSpace(mesh, 'CG', 1)
    Q = FunctionSpace(bmesh, 'CG', 1)
    W = (V, Q)

    up = ii_Function(W)
    u, p = up  # Split

    v, q = list(map(TestFunction, W))
    Tu, Tv = (Trace(x, bmesh) for x in (u, v))

    dxGamma = Measure('dx', domain=bmesh)

    # Nonlinearity
    nl = lambda u: (1+u)**2

    f_exact = -div(nl(u)*grad(u))
    g_exact = u

    f = interpolate(Expression(f_exact, subs={u: u_exact}, degree=1), V)
    h = interpolate(Expression(g_exact, subs={u: u_exact}, degree=4), Q)

    # The nonlinear functional
    F = [inner(nl(u)*grad(u), grad(v))*dx + inner(p, Tv)*dxGamma - inner(f, v)*dx,
         inner(Tu, q)*dxGamma - inner(h, q)*dxGamma]

    dF = block_jacobian(F, up)

    return F, dF, up


class NLProblem(NonlinearProblem):
    '''Object for SNES solver'''
    def __init__(self, F, dF):
        NonlinearProblem.__init__(self)
        self.lhs_form = F
        self.jac_form = dF
        self.Fcount, self.Jcount = 0, 0

    def F(self, b, x):
        print('F, enter', x.size())
        b_ = ii_convert(ii_assemble(self.lhs_form))
        self.Fcount += 0
        if b.empty():
            b = as_backend_type(b).vec()
            b.setSizes(b_.size())
            b.setUp()
            b.assemble()

        print('F, enter', b_.size(), b.size)
                    
        b.zeroEntries()
        b.axpy(1, as_backend_type(b_).vec())
        b.assemble()
        print('6', self.Fcount)

    def J(self, A, x):
        print('J, enter', x.size())
        A_ = ii_convert(ii_assemble(self.jac_form))
        print('1')
        self.Jcount += 0
        if A.empty():
            A = as_backend_type(A).mat()
            A.setSizes((A_.size(0), A_.size(1)))
            print('2')
            A.setUp()
            A.assemble()
        print('3')
        A.zeroEntries()
        print('4')
        A.axpy(1, as_backend_type(A_).mat(), PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
        A.assemble()

        print((PETScMatrix(A).array() - A_.array()))
        print('5', self.Jcount)

        
# --------------------------------------------------------------------

if __name__ == '__main__':
    import sys, petsc4py
    petsc4py.init(sys.argv)
    
    import sympy as sp

    # Setup the test case
    x, y = sp.symbols('x y')
    u_exact = sp.cos(sp.pi*x*(1-x)*y*(1-y))
    p_exact = sp.S(0)

    u_expr = Expression(u_exact, degree=4)
    p_expr = Expression(p_exact, degree=4)

    eu0, ep0, h0 = -1, -1, -1
    for N in (8, ):#16, 32, 64, 128, 256):
        F, dF, w = nonlinear_babuska(N, u_exact, p_exact)

        problem = NLProblem(F, dF)

        solver = NewtonSolver()
        solver.solve(problem, w.vector())
        # w_vec = as_backend_type(w.vector()).vec()

        # feed = SNESFeed(F, dF)

        # snes = PETSc.SNES().create()

        # b = as_backend_type(w.vector()).vec().copy()
        # snes.setFunction(feed.formFunction, b)

        # A = PETSc.Mat().createAIJ(size=(b.size, b.size))
        # A.setUp()
        # snes.setJacobian(feed.formJacobian, A)
        
        #snes.getKSP().setType('preonly')
        #snes.getKSP().getPC().setType('lu')
        #snes.getKSP().getPC().setFactorSolverPackage('umfpack')
        #snes.setFromOptions()

        #snes.solve(None, w_vec)

        
    #     Vh = uh.function_space()
    #     Qh = ph.function_space()

    #     eu = errornorm(uh, u_expr, 'H1', degree_rise=0)
    #     ep = errornorm(ph, p_expr, 'L2', degree_rise=0)
    #     h = Vh.mesh().hmin()
        
    #     if eu0 > 0:
    #         rate_u = ln(eu/eu0)/ln(h/h0)
    #         rate_p = ln(ep/ep0)/ln(h/h0)
    #     else:
    #         rate_u, rate_p = -1, -1
            
    #     eu0, ep0, h0 = eu, ep, h
        
    #     data = (eu, rate_u, ep, rate_p, Vh.dim() + Qh.dim())
        
    #     print('|e|_1 = %.4E[%.2f] |p|_0 = %.4E[%.2f] | ndofs = %d' % data)
    
    # File('./nl_results/babuska_uh.pvd') << uh
    # File('./nl_results/babuska_ph.pvd') << ph

