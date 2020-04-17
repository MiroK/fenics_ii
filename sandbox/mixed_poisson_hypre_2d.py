from dolfin import *
from petsc4py import PETSc
from mpi4py import MPI as pyMPI
from sympy.printing import ccode
import sympy as sp
import numpy as np

from block import block_assemble, block_mat
from block.iterative import MinRes
from block.algebraic.petsc import LU, LumpedInvDiag
from block.block_base import block_base

# MMS utils
def expr_body(expr, **kwargs):
    if not hasattr(expr, '__len__'):
        # Defined in terms of some coordinates
        xyz = set(sp.symbols('x[0], x[1], x[2]'))
        xyz_used = xyz & expr.free_symbols
        assert xyz_used <= xyz
        # Expression params which need default values
        params = (expr.free_symbols - xyz_used) & set(kwargs.keys())
        # Body
        expr = ccode(expr).replace('M_PI', 'pi')
        # Default to zero
        kwargs.update(dict((str(p), 0.) for p in params))
        # Convert
        return expr
    # Vectors, Matrices as iterables of expressions
    else:
        return [expr_body(e, **kwargs) for e in expr]


def as_expression(expr, degree=4, **kwargs):
    '''Turns sympy expressions to Dolfin expressions.'''
    return Expression(expr_body(expr), degree=degree, **kwargs)


def vec(x):
    return as_backend_type(x).vec()


def mat(A):
    return as_backend_type(A).mat()


class HypreAMS(block_base):
    '''AMG auxiliary space preconditioner for Hdiv(0) norm'''
    def __init__(self, V, hdiv0=False, bc=None):
        # FIXME: lift
        assert V.ufl_element().family() == 'Raviart-Thomas'
        assert V.ufl_element().degree() == 1

        mesh = V.mesh()
        assert mesh.geometry().dim() == 2

        sigma, tau = TrialFunction(V), TestFunction(V)
        
        a = inner(div(sigma), div(tau))*dx
        if not hdiv0:
            a += inner(sigma, tau)*dx

        f = Constant(np.zeros(V.ufl_element().value_shape()))
        L = inner(tau, f)*dx

        A, _ = assemble_system(a, L, bc)

        # AMS setup
        Q = FunctionSpace(mesh, 'CG', 1)
        G = DiscreteOperators.build_gradient(V, Q)

        pc = PETSc.PC().create(mesh.mpi_comm().tompi4py())
        pc.setType('hypre')
        pc.setHYPREType('ams')

        # Attach gradient
        pc.setHYPREDiscreteGradient(mat(G))

        # Constant nullspace (in case not mass and bcs)
        constants = [vec(interpolate(c, V).vector())
                     for c in (Constant((1, 0)), Constant((0, 1)))]

        pc.setHYPRESetEdgeConstantVectors(*constants)

        # NOTE: term mass term is accounted for automatically by Hypre
        # unless pc.setPoissonBetaMatrix(None)
        if hdiv0: pc.setHYPRESetBetaPoissonMatrix(None)

        pc.setOperators(mat(A))
        # FIXME: some defaults
        pc.setFromOptions()
        pc.setUp()

        self.pc = pc
        self.A = A   # For creating vec

    def matvec(self, b):
        if not isinstance(b, GenericVector):
            return NotImplemented

        x = self.A.create_vec(dim=1)
        if len(x) != len(b):
            raise RuntimeError(
                'incompatible dimensions for PETSc matvec, %d != %d'%(len(x),len(b)))

        self.pc.apply(vec(b), vec(x))
        return x


def main(n):
    '''Solves grad-div problem in 2d with HypreAMS preconditioning'''
    # Exact solution
    x, y = sp.symbols('x[0] x[1]')
    
    u = sp.sin(pi*x*(1-x)*y*(1-y))

    sp_div = lambda f: f[0].diff(x, 1) + f[1].diff(y, 1)

    sp_grad = lambda f: sp.Matrix([f.diff(x, 1), f.diff(y, 1)])

    sigma = sp_grad(u)    
    f = -sp_div(sigma) + u

    sigma_expr, u_expr, f_expr = list(map(as_expression, (sigma, u, f)))

    # The discrete problem
    mesh = UnitSquareMesh(n, n)
    
    V = FunctionSpace(mesh, 'RT', 1)
    Q = FunctionSpace(mesh, 'DG', 0)
    W = (V, Q)

    sigma, u = list(map(TrialFunction, W))
    tau, v = list(map(TestFunction, W))

    a00 = inner(sigma, tau)*dx
    a01 = inner(div(tau), u)*dx
    a10 = inner(div(sigma), v)*dx
    a11 = -inner(u, v)*dx

    L0 = inner(Constant((0, 0)), tau)*dx
    L1 = inner(-f_expr, v)*dx

    AA = block_assemble([[a00, a01], [a10, a11]])
    bb = block_assemble([L0, L1])

    # b00 = inner(sigma, tau)*dx + inner(div(sigma), div(tau))*dx
    # B00 = LU(assemble(b00))
    B00 = HypreAMS(V)
    
    b11 = inner(u, v)*dx
    B11 = LumpedInvDiag(assemble(b11))

    BB = block_mat([[B00, 0], [0, B11]])
    
    AAinv = MinRes(AA, precond=BB, tolerance=1e-10, maxiter=500, show=2)

    # Compute solution
    sigma_h, u_h = AAinv * bb
    sigma_h, u_h = Function(V, sigma_h), Function(Q, u_h)

    niters = len(AAinv.residuals) - 1
    # error = sqrt(errornorm(sigma_expr, sigma_h, 'Hdiv', degree_rise=1)**2 +
    #              errornorm(u_expr, u_h, 'L2', degree_rise=1)**2)

    hmin = mesh.mpi_comm().tompi4py().allreduce(mesh.hmin(), pyMPI.MIN)
    error = 1.

    return hmin, V.dim()+Q.dim(), niters, error


# --------------------------------------------------------------------------

if __name__ == '__main__':
    msg = 'hmin = %g #dofs = %d, niters = %d, error = %g(%.2f)'

    h0, error0 = None, None
    for n in (8, 16, 32, 64, 128, 256, 512, 1024): 
        h, ndofs, niters, error = main(n)

        if error0 is not None:
            rate = ln(error/error0)/ln(h/h0)
        else:
            rate = -1
        h0, error0 = h, error

        print((msg % (h, ndofs, niters, error, rate)))
