from __future__ import absolute_import
from xii.assembler.ufl_utils import replace as ii_replace
from six.moves import map
from six.moves import zip

import dolfin as df
import ufl
from six.moves import filter


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
    assert all(isinstance(f, df.Coefficient) for f in foos)

    # Each row is supposed to be a function in a non-mixed function space
    test_functions = list(map(is_okay_functional, bf))
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
    return next(filter(lambda x: x.number() == 1, f.arguments()))


def ii_derivative(f, x):
    '''DOLFIN's derivative df/dx extended to handle trace stuff'''
    test_f = is_okay_functional(f)

    # FIXME: for now don't allow diffing wrt compound expressions, in particular
    # restricted args. 
    assert isinstance(x, df.Coefficient) and not is_restricted(x)

    # So now we have L(arg, v) where arg = (u, ..., T[u], Pi[u], ...) and the idea
    # is to define derivative w.r.t to x by doing
    # J = sum_{arg} (partial L / partial arg){arg=T[u]}(partial arg / partial x). 
    J = 0
    # NOTE: in the following I try to avoid assembly of zero forms because
    # these might not be well-defined for xii assembler. Also, assembling
    # zeros is useless
    for fi in [c for c in f.coefficients() if not isinstance(c, df.Constant)]:
        # Short circuit if (partial fi)/(partial x) is 0
        if not ((fi == x) or fi.vector().id() == x.vector().id()): continue

        if is_restricted(fi):
            rtype = restriction_type(fi)
            attributes = (rtype, )
        else:
            rtype = ''
            attributes = None
        fi_sub = df.Function(fi.function_space())
        
        # To recreate the form for partial we sub in every integral of f
        sub_form_integrals = []
        for integral in f.integrals():
            
            integrand = ii_replace(integral.integrand(), fi, fi_sub, attributes)
            # If the substitution is do nothing then there's no need to diff
            if integrand != integral.integrand():                
                sub_form_integrals.append(integral.reconstruct(integrand=integrand))
        sub_form = ufl.Form(sub_form_integrals)

        # Partial wrt to substituated argument
        df_dfi = df.derivative(sub_form, fi_sub)

        # Substitue back the original form argument
        sub_form_integrals = []
        for integral in df_dfi.integrals():
            integrand = ii_replace(integral.integrand(), fi_sub, fi)
            assert integrand != integral.integrand()
            sub_form_integrals.append(integral.reconstruct(integrand=integrand))
        df_dfi = ufl.Form(sub_form_integrals)

        # As d Tu / dx = T(x) we now need to restrict the trial function
        if rtype:
            trial_f = get_trialfunction(df_dfi)
            setattr(trial_f, rtype, getattr(fi, rtype))
        # Since we only allos diff wrt to coef then in the case rtype == ''
        # we have dfi/dx = 1
        
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

        if dL == 0:
            info('dL/du is zero')
            return None
        
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

        return (Lu_dw - Lu_dw0).norm('linf'), Lu_dw0.norm('linf')#, dw.norm('linf')

    # ---------------------------------------------------------------
    # Let's check som functional => match the hand computed value while
    # the value of the functional is also > 0
    def test(a, b):
        assert a < 1E-14 and b > 0, (a, b)
        return True
    
    mesh = UnitSquareMesh(10, 10)
    bmesh = BoundaryMesh(mesh, 'exterior')
    
    V = FunctionSpace(mesh, 'CG', 1)
    Q = FunctionSpace(bmesh, 'CG', 1)
    W = [V, Q]

    u, p = interpolate(Constant(2), V), interpolate(Constant(1), Q)
    v, q = list(map(TestFunction, W))
    Tu, Tv = (Trace(x, bmesh) for x in (u, v))

    # NOTE in the tests below `is_linear` assigns to u making it nonzero
    # so forms where based on u = Function(V) you expect zero are nz.
    # Also, since the assignment is rand this looks differently every time
    dxGamma = Measure('dx', domain=bmesh)
    # These should be linear
    L = inner(Tu, q)*dxGamma
    assert is_linear(L, Function(V)) == None
    test(*is_linear(L, u))
    
    # ---------------------------------------------------------------
    
    L = inner(Tu+Tu, q)*dxGamma
    test(*is_linear(L, u))

    # ---------------------------------------------------------------

    # # Some simple nonlinearity where I can check things
    L = inner(Tu**2, q)*dxGamma
    dL0 = inner(2*Tu*Trace(TrialFunction(V), bmesh), q)*dxGamma
    A0 = ii_convert(ii_assemble(dL0)).array()

    dL = ii_derivative(L, u)
    A = ii_convert(ii_assemble(dL)).array()

    test(np.linalg.norm(A - A0, np.inf), np.linalg.norm(A0, np.inf))

    # ---------------------------------------------------------------

    L = inner(2*Tu**3 - sin(Tu), q)*dxGamma
    dL0 = inner((6*Tu**2 - cos(Tu))*Trace(TrialFunction(V), bmesh), q)*dxGamma
    A0 = ii_convert(ii_assemble(dL0)).array()

    dL = ii_derivative(L, u)
    A = ii_convert(ii_assemble(dL)).array()

    test(np.linalg.norm(A - A0, np.inf), np.linalg.norm(A0, np.inf))

    # ---------------------------------------------------------------

    # Something that can be encounted in babuska
    L = inner(u**2*grad(u), grad(v))*dx + inner(p**2, Tv)*dxGamma

    du = TrialFunction(V)
    dL0 = inner(2*u*du*grad(u) + u**2*grad(du), grad(v))*dx
    A0 = ii_convert(ii_assemble(dL0)).array()

    dL = ii_derivative(L, u)
    A = ii_convert(ii_assemble(dL)).array()

    test(np.linalg.norm(A - A0, np.inf), np.linalg.norm(A, np.inf))

    # ---------------------------------------------------------------

    dp = TrialFunction(Q)
    dL0 = inner(2*p*dp, Tv)*dxGamma
    A0 = ii_convert(ii_assemble(dL0)).array()

    dL = ii_derivative(L, p)
    A = ii_convert(ii_assemble(dL)).array()

    test(np.linalg.norm(A - A0, np.inf), np.linalg.norm(A, np.inf))

    # ---------------------------------------------------------------

    L = inner(Tu*p, Tv)*dxGamma

    du = TrialFunction(V)
    dL0 = inner(Trace(du, bmesh)*p, Tv)*dxGamma
    A0 = ii_convert(ii_assemble(dL0)).array()

    dL = ii_derivative(L, u)
    A = ii_convert(ii_assemble(dL)).array()

    test(np.linalg.norm(A - A0, np.inf), np.linalg.norm(A, np.inf))

    # ---------------------------------------------------------------

    L = inner(Tu*p, Tv)*dxGamma

    dp = TrialFunction(Q)
    dL0 = inner(Tu*dp, Tv)*dxGamma
    A0 = ii_convert(ii_assemble(dL0)).array()

    dL = ii_derivative(L, p)
    A = ii_convert(ii_assemble(dL)).array()

    test(np.linalg.norm(A - A0, np.inf), np.linalg.norm(A, np.inf))

    # ---------------------------------------------------------------

    L = inner(Tu*Tu + p*p, Tv)*dxGamma

    du = TrialFunction(V)
    dL0 = inner(2*Tu*Trace(du, bmesh), Tv)*dxGamma
    A0 = ii_convert(ii_assemble(dL0)).array()

    dL = ii_derivative(L, u)
    A = ii_convert(ii_assemble(dL)).array()

    test(np.linalg.norm(A - A0, np.inf), np.linalg.norm(A, np.inf))

    # ---------------------------------------------------------------

    dp = TrialFunction(Q)
    dL0 = inner(2*p*dp, Tv)*dxGamma
    A0 = ii_convert(ii_assemble(dL0)).array()

    dL = ii_derivative(L, p)
    A = ii_convert(ii_assemble(dL)).array()

    test(np.linalg.norm(A - A0, np.inf), np.linalg.norm(A, np.inf))
