from __future__ import absolute_import
from __future__ import print_function
from dolfin import *
from petsc4py import PETSc
from mpi4py import MPI as pyMPI
from sympy.printing import ccode
import sympy as sp
from six.moves import map

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


def main(n):
    '''Solves grad-div problem in 2d with HypreAMS preconditioning'''
    # Exact solution
    x, y = sp.symbols('x[0] x[1]')
    
    sigma = sp.Matrix([sp.cos(pi*y**2), sp.sin(pi*x**2)])

    sp_div = lambda f: f[0].diff(x, 1) + f[1].diff(y, 1)

    sp_grad = lambda f: sp.Matrix([f.diff(x, 1), f.diff(y, 1)])
    
    f = -sp_grad(sp_div(sigma)) + sigma

    sigma_expr, f_expr = list(map(as_expression, (sigma, f)))

    # The discrete problem
    mesh = UnitSquareMesh(n, n)
    
    V = FunctionSpace(mesh, 'RT', 1)
    u, v = TrialFunction(V), TestFunction(V)

    a = inner(div(u), div(v))*dx + inner(u, v)*dx
    L = inner(f_expr, v)*dx

    A, b = assemble_system(a, L)

    # Solve
    Q = FunctionSpace(mesh, 'CG', 1)
    G = DiscreteOperators.build_gradient(V, Q)

    ksp = PETSc.KSP().create(PETSc.COMM_WORLD)
    ksp.setType('cg')
    ksp.setTolerances(rtol=1E-8, atol=1E-12, divtol=1.0E10, max_it=300)

    opts = PETSc.Options()
    opts.setValue("ksp_monitor_true_residual", None)
    ksp.setFromOptions()

    # AMS preconditioner
    pc = ksp.getPC()
    pc.setType('hypre')
    pc.setHYPREType('ams')

    vec = lambda x: as_backend_type(x).vec()
    mat = lambda A: as_backend_type(A).mat()

    # Attach gradient
    pc.setHYPREDiscreteGradient(mat(G))

    # Constant nullspace (in case not mass and bcs)
    constants = [vec(interpolate(c, V).vector())
                 for c in (Constant((1, 0)), Constant((0, 1)))]

    pc.setHYPRESetEdgeConstantVectors(*constants)
    # NOTE: term mass term is accounted for automatically by Hypre
    # unless pc.setPoissonBetaMatrix(None)

    # Set operator for the linear solver
    ksp.setOperators(mat(A))

    uh = Function(V)

    ksp.solve(vec(b), vec(uh.vector()))
    
    niters = ksp.getIterationNumber()
    error = errornorm(sigma_expr, uh, 'Hdiv', degree_rise=1)
    hmin = mesh.mpi_comm().tompi4py().allreduce(mesh.hmin(), pyMPI.MIN)

    return hmin, V.dim(), niters, error

# --------------------------------------------------------------------------

if __name__ == '__main__':
    msg = 'hmin = %g #dofs = %d, niters = %d, error = %g(%.2f)'

    h0, error0 = None, None
    for n in (8, 16, 32, 64, 128, 256): 
        h, ndofs, niters, error = main(n)

        if error0 is not None:
            rate = ln(error/error0)/ln(h/h0)
        else:
            rate = -1
        h0, error0 = h, error

        print((msg % (h, ndofs, niters, error, rate)))
