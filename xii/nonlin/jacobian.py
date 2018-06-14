from xii.assembler.ufl_utils import replace as ii_replace
from itertools import ifilter
import dolfin as df
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


def is_restricted(x):
    '''Is x marked for restriction (in some way we support)'''
    for rtype in ('trace_', 'average_', 'restriction_'):
        if hasattr(x, rtype):
            return True
        
    return False


def restriction_type(x):
    '''How is x restricted'''
    assert is_restricted(x)

    for rtype in ('trace_', 'average_', 'restriction_'):
        if hasattr(x, rtype):
            return rtype


def get_trialfunction(f):
    '''Extract (pointer) to trial function of the object'''
    return next(ifilter(lambda x: x.number() == 1, f.arguments()))


def ii_derivative(f, x):
    '''DOLFIN's derivative df/dx extended to handle trace stuff'''
    test_f = is_okay_functional(f)

    # FIXME: for now don't allow diffing wrt compound expressions, in particular
    # restricted args. However, since trace u is just annotating u we
    # can make the distinction here
    assert isinstance(x, Coefficient)
    if is_restricted(x):
        df.warning('Disregarding restriction in diffing')

    # The case that dolfin can handle out of the box
    if not any(map(is_restricted, f.coefficients())):
        # There are no restrictied args in the definition of f
        if not is_restricted(test_f):
            return df.derivative(f, x)
        # Since diff only manips of coeffs restrictions to test_f are preserved
        # so we're always good
        return df.derivative(f, x)

    # If some was them was unrestricted then they all should be - otherwise
    # we'd be missing terms with different dimensionality (most likely)
    assert all(map(is_restricted, f.coefficients()))

    # So now we have L(arg, v) where arg = (T[u], Pi[u], ...) and the idea
    # is to define derivative w.r.t to x by doing
    # sum_{arg} (partial L / partial arg){arg=T[u]}(partial arg / partial x). 
    J = 0
    for fi in f.coefficients():
        rtype = restriction_type(fi)
        fi_sub = df.Function(fi.function_space())
        
        # To recreate the form for partial we sub in every integral of f
        sub_form = []
        for integral in f.integrals():
            integrand = ii_replace(integral.integrand(), fi, fi_sub, attributes=(rtype, ))
            sub_form.append(integral.reconstruct(integrand=integrand))
        sub_form = ufl.Form(sub_form)
        
        # Partial wrt to substituated argument
        df_dfi = df.derivative(sub_form, fi_sub)

        sub_form = []
        # An subback for trace!
        for integral in df_dfi.integrals():
            integrand = ii_replace(integral.integrand(), fi_sub, fi)
            sub_form.append(integral.reconstruct(integrand=integrand))
        df_dfi = ufl.Form(sub_form)

        # As d Tu / dx = T(x) we now need to restrict the trial function
        trial_f = get_trialfunction(df_dfi)
        setattr(trial_f, rtype, getattr(fi, rtype))
        
        J += df_dfi
    # Done
    return J

# --------------------------------------------------------------------

if __name__ == '__main__':
    from xii import ii_Function, Trace, ii_assemble, ii_convert
    from dolfin import *
    import numpy as np


    def is_linear(L, u):
        '''Is L linear in u'''
        # Compute the deriative
        dL = ii_convert(ii_assemble(ii_derivative(L, u)))
        # Random guy
        w = Function(u.function_space()).vector()
        w.set_local(np.random.rand(w.local_size()))
        # Where we evaluate the direction
        dw = PETScVector(as_backend_type(dL).mat().getVecLeft())
        dL.mult(w, dw)

        # Now L(u) + dw
        Lu_dw = ii_assemble(L) + dw
        # And if the thing is linear then L(u+dw) should be the same
        u.vector().axpy(1, w)
        Lu_dw0 = ii_assemble(L)

        return (Lu_dw - Lu_dw0).norm('linf'), Lu_dw0.norm('linf')

    # ---------------------------------------------------------------
    
    mesh = UnitSquareMesh(10, 10)
    bmesh = BoundaryMesh(mesh, 'exterior')
    
    V = FunctionSpace(mesh, 'CG', 1)
    Q = FunctionSpace(bmesh, 'CG', 1)
    W = [V, Q]

    u, p = map(Function, W)
    v, q = map(TestFunction, W)
    Tu, Tv = (Trace(x, bmesh) for x in (u, v))

    dxGamma = Measure('dx', domain=bmesh)
    # These should be linear
    L = inner(Tu, q)*dxGamma
    print is_linear(L, u)

    L = inner(Tu+Tu, q)*dxGamma
    print is_linear(L, u)

    # Some simple nonlinearity where I can check things
    L = inner(Tu**2, q)*dxGamma
    dL0 = inner(2*Tu*Trace(TrialFunction(V), bmesh), q)*dxGamma
    A0 = ii_convert(ii_assemble(dL0)).array()

    dL = ii_derivative(L, u)
    A = ii_convert(ii_assemble(dL)).array()

    print np.linalg.norm(A - A0, np.inf), np.linalg.norm(A, np.inf)

    L = inner(2*Tu**3 - sin(Tu), q)*dxGamma
    dL0 = inner((6*Tu**2 - cos(Tu))*Trace(TrialFunction(V), bmesh), q)*dxGamma
    A0 = ii_convert(ii_assemble(dL0)).array()

    dL = ii_derivative(L, u)
    A = ii_convert(ii_assemble(dL)).array()

    print np.linalg.norm(A - A0, np.inf), np.linalg.norm(A, np.inf)




    exit()
    #dL = ii_convert(ii_assemble(dL))

    x = Function(V).vector()
    x.set_local(np.random.rand(x.local_size()))

    y = Function(Q).vector()
    dL.mult(x, y)
    y.axpy(-1, ii_assemble(L))
    print y.norm('linf')
    

    exit()
    up = ii_Function(W)
    u, p = up

    Tu, Tv = (Trace(x, bmesh) for x in (u, v))


    L = inner(Tu, q)*dxGamma

    print ii_derivative(L, u)

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


    
