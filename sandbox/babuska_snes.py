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

    v, q = map(TestFunction, W)
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

    return ii_convert(ii_assemble(F)), ii_convert(ii_assemble(dF)), up


class SNESFeed(object):
    '''Object for SNES solver'''
    def __init__(self, F, dF, up):
        '''
        F = Nonlinear functional which assembles into vector
        dF = Jacobian of F which assembles into matrix
        up = Function which vector F == 0 is solved into
        '''
        self.F = F
        self.dF = dF
        self.up = up

        self.__mat = None
        self.__vec = None

    def formJacobian(self, snes, X, A, B):
        A_ = ii_convert(ii_assemble(self.dF))
        set_lg_map(A_)
        #A_ = as_backend_type(A_).mat()
        ## Assembled Jacobian
        #A.zeroEntries()
        #A.axpy(1., A_, None)
        #A.assemble()
        
        # Potentially different matrix for the preconditioner
        #B.assemblyBegin()
        #B.assemblyEnd()
        #B.zeroEntries()
        #B.axpy(1., A_, None)

        return PETSc.Mat.Structure.SAME_NONZERO_PATTERN

    def formFunction(self, snes, X, b):
        b_ = as_backend_type(ii_convert(ii_assemble(self.F))).vec()

        b.assemblyBegin()
        b.assemblyEnd()
        
        b.zeroEntries()
        b.axpy(1., b_)
        b.assemble()

    def mat(self):
        '''Allocate matrix'''
        if self.__mat is None:
            A = ii_convert(ii_assemble(self.dF))
            set_lg_map(A)
            self.__mat = as_backend_type(A).mat()
        return self.__mat

    def vec(self):
        '''Allocate vector'''
        if self.__vec is None:
            b = ii_convert(ii_assemble(self.F))
            self.__vec = as_backend_type(b).vec()
        return self.__vec
        
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
        set_lg_map(dF)
        #feed = SNESFeed(F, dF, w)
        F = as_backend_type(F).vec()
        dF = as_backend_type(dF).mat()

        print dF.getValuesCSR()

        A = PETSc.Mat().createAIJ(size=dF.size,
                                  csr=dF.getValuesCSR())
        A.assemble()
        
        x = A.createVecLeft()
        y = A.createVecRight()

        ksp = PETSc.KSP().create()
        #ksp.setType('gmres')
        #ksp.getPC().setType('lu')
        ksp.setOperators(A)
        ksp.solve(y, x)

        #snes = PETSc.SNES().create()

        #b = feed.vec()
        #snes.setFunction(feed.formFunction, b)

        #A = feed.mat()
        #snes.setJacobian(feed.formJacobian, A)
        
        #snes.getKSP().setType('gmres')
        #snes.getKSP().getPC().setType('lu')
        #snes.setFromOptions()

        #snes.solve(None, b.copy())

        
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

