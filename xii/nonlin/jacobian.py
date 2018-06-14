from dolfin import *
from xii import *
import ufl


def is_okay_functional(L):
    '''Functional over V (cannot have other test functions but in V)'''
    args = set(L.arguments())
    assert len(args) == 1
    # All are test functions
    assert not any(arg.number() == 1 for arg in args)
    # That one test function
    return args.pop()

    
def block_jacobian(bf, foos):
    '''
    Jacobian of block functional is a block matrix with entries d bf[i] / d foos[j]
    '''
    # Want a square matrix
    assert len(bf) == len(foos)
    # Things in a row are forms
    assert all(isinstance(L, ufl.Form) for L in bf)
    # Diff w.r.t functions
    assert all(isinstance(f, Coefficient) for f in foos)

    # Each row is supposed to be a function in a non-mixed function space
    test_functions = map(is_okay_functional, bf)
    # Have them ordered the same way as foo
    assert all(t.function_space() == f.function_space()
               for t, f in zip(test_functions, foos))

    return [[ii_derivative(bfi, fj) for fj in foos] for bfi in bf]


def ii_derivative(f, x):
    '''DOLFIN's derivative df/dx extended to handle trace stuff'''
    assert is_okay_functional(f)

    # The case that dolfin can handle out of the box - no restrictions
    if not_has_restricted_args(f) 

    if not is_restricted(v): return derivative(f, x)

    # (Tu, q) wrt q

    #

# --------------------------------------------------------------------

if __name__ == '__main__':

    mesh = UnitSquareMesh(10, 10)
    bmesh = BoundaryMesh(mesh, 'exterior')
    
    V = FunctionSpace(mesh, 'CG', 1)
    Q = FunctionSpace(mesh, 'CG', 1)
    W = [V, Q]
    
    v, q = map(TestFunction, W)

    up = ii_Function(W)
    u, p = up

    Tu, Tv = (Trace(x, bmesh) for x in (u, v))
    dxGamma = Measure('dx', domain=bmesh)

    L = inner(p, Tv)*dxGamma



    exit()
    f = interpolate(Expression('sin(pi*(x[0]+x[1]))', degree=1), V)
    g = interpolate(Expression('x[0]*x[0]+x[1]*x[1]', degree=1), Q)

    F = [inner((1 + u)**2*grad(u), grad(v))*dx + inner(u, v)*dx - inner(f, v)*dx,
         inner((1 + p)**2*grad(p), grad(q))*dx + inner(p, q)*dx - inner(g, q)*dx]

    dF = block_jacobian(F, (u, p))

    # Newton
    omega = 1.0       # relaxation parameter
    eps = 1.0
    tol = 1.0E-5
    iter = 0
    maxiter = 25

    dup = ii_Function(W)
    while eps > tol and iter < maxiter:
        iter += 1
        A, b = map(ii_assemble, (dF, F))

        A, b = map(ii_convert, (A, b))
        
        solve(A, dup.vector(), b)
        
        eps = sqrt(sum(x.norm('l2')**2 for x in dup.vectors()))
        
        print 'Norm:', eps, A.norm('linf'), b.norm('l2')

        for i in range(len(W)):
            up[i].vector().axpy(-omega, dup[i].vector())

    for i, x in enumerate(up):
        File('%s_%d.pvd' % ('test_nlin', i)) << x


    
