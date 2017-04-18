from dolfin import *
from petsc4py import PETSc
from scipy.linalg import eigh
import numpy as np


def reduced_gevp(a, bcs, m=None, eps=1E-8):
    '''
    Let a, m be symmetric bilinear forms over V which is considered with bcs 
    and it is assumed that card(bcs) is close to the dim(V). The generalized 
    eigenvalue problem: Find u, lmbda such that
        
        a(u, v) = lmbda*m(u, v) for each v in V

    is here solved by reducing the problem to dofs which are not constrained by
    bcs
    '''
    # First check that the problem is symmetric - this is need e.g. because then
    # the complete set of eigenvectors exists
    V = a.arguments()[0].function_space()
    assert V == a.arguments()[1].function_space()
    if m is not None: assert all(V == v.function_space() for v in m.arguments())
    n = V.dim()

    v = TestFunction(V)

    if V.ufl_element().family() == 'HDiv Trace':
        L = inner(v, Constant(np.zeros(v.ufl_shape)))*ds# +\
    else:
        L = inner(v, Constant(np.zeros(v.ufl_shape)))*dx

    A, _ = assemble_system(a, L, bcs)
    if m is not None:
        M, _ = assemble_system(m, L, bcs)
    else:
        M = None

    A = as_backend_type(A).mat()
    assert A.isHermitian(eps*ln(V.dim()))

    if M is not None:
        M = as_backend_type(M).mat()
        assert M.isHermitian(eps*ln(V.dim()))

    # Build the reduction matrix == basis of the column space which is the
    # orthogonal complement of the column space of R^n basis vectors due to bcs
    if isinstance(bcs, DirichletBC): bcs = [bcs]
    I = sum((bc.get_boundary_values().keys() for bc in bcs), [])
    m = len(I)

    I_perp = set(map(long, range(0, n))).difference(set(I))
    k = len(I_perp)
    assert m + k == n

    E_perp = PETSc.Mat()
    E_perp.createAIJ(size=(n, k), nnz=1)
    E_perp.setUp()

    for col, row in enumerate(I_perp):
        rows = [row]
        cols = [col]
        values = [1.] 

        E_perp.setValues(rows, cols, values, PETSc.InsertMode.INSERT_VALUES)
    E_perp.assemble()

    # Reduce as Ep.T.(A.Ep)
    R0, RA = PETSc.Mat(), PETSc.Mat()
    A.matMult(E_perp, R0)
    E_perp.transposeMatMult(R0, RA)
    RA = PETScMatrix(RA)

    if M is not None:
        RM = PETSc.Mat()
        M.matMult(E_perp, R0)
        E_perp.transposeMatMult(R0, RM)
        RM = PETScMatrix(RM)

    # Solve the reduced eigenvalue problem
    info('Solving reduced GEVP of size %d' % RA.size(0))
    timer = Timer('GEVP')
    
    if M is not None:
        eigw_Ip, eigv_Ip = eigh(RA.array(), RM.array())
    else:
        eigw_Ip, eigv_Ip = eigh(RA.array())
    
    t = timer.stop()
    info('Done in %g' % t)

    # It remains to solve the eigenvalue problem for the dofs constrained by bcs
    # For Ae_i = lambda e_i, the single diag entry of A is the eigenvalue
    eigw_I = A.getDiagonal()
    eigw_I = eigw_I.array_r[I]
    
    # For Ae_i = lambda Me_i, note that M_ei = beta e_i is not necessarily such
    # that beta == 1 so we need to M-othonormalize the vectors. Moreover
    # scaling of the eigenvalue of A is required
    if M is not None:
        # So here we collect normalization factors
        M_normalize = M.getDiagonal()
        M_normalize = M_normalize.array_r[I]
        # And adjust the eigenvalues
        eigw_I = eigw_I/M_normalize
    else:
        M_normalize = np.ones_like(eigw_I)

    # Results are separated into I and I_perp part, for each the minimum stuff is
    # given to give the solution
    if M is None:
        return (eigw_I, I, M_normalize), (eigw_Ip, eigv_Ip, E_perp)
    else:
        # RM is necessary here for we do know if the data will be used to
        # constract the norm or its inverse
        return (eigw_I, I, M_normalize), (eigw_Ip, eigv_Ip, E_perp, M)


def interp_norm(a, bcs, s, m=None):
    '''
    Let a, m symmetric forms over V which is considered with bcs. Suppose these
    assemble respectively into matrices A and M. The interpolation norm of order
    s is then induced by H_s = (inv(M)*A)^s

    Here a function f is returned such that H_s*vec = f(vec)
    '''
    # with EVP
    if m is None:
        (eigw_I, I, _), (eigw_Ip, eigv_Ip, E_perp) = reduced_gevp(a, bcs)
        
        # The action of the I part can be described in terms of point operations
        # between a vector and a filter which only has the (powers of)
        # eigenvalues in a right place
        filter_I = np.zeros(len(eigw_I)+len(eigw_Ip))
        filter_I[I] = eigw_I**s
        filter_I = PETSc.Vec().createWithArray(filter_I)
        # Alloc for I action
        y_I = filter_I.copy()

        # The action of the Ip part is E_perp (C*Lambda*C^t) *E_perp.T*v
        core = eigv_Ip.dot(np.diag(eigw_Ip**s).dot(eigv_Ip.T))
        # Alloc for intermediate product
        z = PETSc.Vec().createSeq(len(eigw_Ip))
        # And the final output
        y = filter_I.copy()
    
        # Here vec is Vector class object
        def f(vec):
            x = as_backend_type(vec).vec()

            # The action of the constrained part
            y_I.pointwiseMult(x, filter_I)  # Note that this zeros

            # The rest
            E_perp.multTranspose(x, z)
            core_z = core.dot(z.array)
            z.setArray(core_z)
            E_perp.mult(z, y)

            y.axpy(1, y_I)

            return PETScVector(y)
    # With GEVP
    else:
        (eigw_I, I, M_normalize), (eigw_Ip, eigv_Ip, E_perp, M) = reduced_gevp(a, bcs, m=m)
        
        # The action of the I part can be described in terms of point operations
        # between a vector and a filter which only has the (powers of)
        # eigenvalues in a right place
        filter_I = np.zeros(len(eigw_I)+len(eigw_Ip))
        filter_I[I] = M_normalize*(eigw_I**s)
        filter_I = PETSc.Vec().createWithArray(filter_I)
        # Alloc for I action
        y_I = filter_I.copy()

        # The action ...
        ME_perp = PETSc.Mat()
        M.matMult(E_perp, ME_perp)
        core = eigv_Ip.dot(np.diag(eigw_Ip**s).dot(eigv_Ip.T))
        # Alloc for intermediate product
        z = PETSc.Vec().createSeq(len(eigw_Ip))
        # And the final output
        y = filter_I.copy()
    
        # Here vec is Vector class object
        def f(vec):
            x = as_backend_type(vec).vec()

            # The action of the constrained part
            y_I.pointwiseMult(x, filter_I)  # Note that this zeros

            # The rest
            ME_perp.multTranspose(x, z)
            core_z = core.dot(z.array)
            z.setArray(core_z)
            ME_perp.mult(z, y)

            y.axpy(1, y_I)

            return PETScVector(y)

    return f


def interp_norm_inv(a, bcs, s, m=None):
    '''Inverse of the operator that is the interpolation norm with s=s.'''
    if m is None:
        return interp_norm(a, bcs, -s)
    else:
        (eigw_I, I, M_normalize), (eigw_Ip, eigv_Ip, E_perp, _) = reduced_gevp(a, bcs, m=m)
        
        # The action of the I part can be described in terms of point operations
        # between a vector and a filter which only has the (powers of)
        # eigenvalues in a right place
        filter_I = np.zeros(len(eigw_I)+len(eigw_Ip))
        filter_I[I] = (eigw_I**(-s))/M_normalize
        filter_I = PETSc.Vec().createWithArray(filter_I)
        # Alloc for I action
        y_I = filter_I.copy()

        # The action ...
        core = eigv_Ip.dot(np.diag(eigw_Ip**(-s)).dot(eigv_Ip.T))
        # Alloc for intermediate product
        z = PETSc.Vec().createSeq(len(eigw_Ip))
        # And the final output
        y = filter_I.copy()
    
        # Here vec is Vector class object
        def f(vec):
            x = as_backend_type(vec).vec()

            # The action of the constrained part
            y_I.pointwiseMult(x, filter_I)  # Note that this zeros

            # The rest
            E_perp.multTranspose(x, z)
            core_z = core.dot(z.array)
            z.setArray(core_z)
            E_perp.mult(z, y)

            y.axpy(1, y_I)

            return PETScVector(y)

        return f

# -----------------------------------------------------------------------------

if __name__ == '__main__':

    gamma = ['near((x[0]-0.25)*(x[0]-0.75), 0) && (0.25-DOLFIN_EPS < x[1]) && (x[1] < 0.75+DOLFIN_EPS)',
             'near((x[1]-0.25)*(x[1]-0.75), 0) && (0.25-DOLFIN_EPS < x[0]) && (x[0] < 0.75+DOLFIN_EPS)']
    gamma = map(lambda x: '('+x+')', gamma)
    gamma = ' || '.join(gamma)
    gamma = CompiledSubDomain(gamma)

    n = 8
    n *= 4
    mesh = UnitSquareMesh(n, n)
    facet_f = FacetFunction('size_t', mesh, 0)
    gamma.mark(facet_f, 1)

    V = FunctionSpace(mesh, 'CG', 1)
    u = TrialFunction(V)
    v = TestFunction(V)

    not_gamma = lambda x, on_boundary: not gamma.inside(x, on_boundary)
    bc = DirichletBC(V, Constant(0), not_gamma, 'pointwise')
    n = V.dim()
    
    a_form = inner(grad(u), grad(v))*dx
    L_form = inner(Constant(0), v)*dx
    m_form = inner(u, v)*dx

    ##################
    # Full EVP problem
    ##################
    A, _ = assemble_system(a_form, L_form, bc)
    A = A.array()

    I, Ip = reduced_gevp(a_form, bcs=bc)

    eigw = np.sort(np.r_[I[0], Ip[0]])

    info('Solving EVP of size %d' % len(A))
    timer = Timer('EVP')
    eigw0, eigv0 = eigh(A)
    t0 = timer.stop()
    info('Done in %g' % t0)

    _TOL = 1E-10*np.log(V.dim())
    e = np.linalg.norm(eigw-np.sort(eigw0)); assert e < _TOL

    f = Function(V)
    vec = f.vector()
    as_backend_type(vec).vec().setRandom()

    # With s = 1 we have hs_norm == A
    hs_norm = interp_norm(a=a_form, bcs=bc, s=1.0)
    y = A.dot(vec.array())
    y0 = hs_norm(vec)
    e = np.linalg.norm(y - y0.array()); assert e < _TOL

    # With s = -1 we have inv(A)
    hs_norm = interp_norm_inv(a=a_form, bcs=bc, s=1.0)
    vec0 = hs_norm(y0)
    e = np.linalg.norm(vec.array() - vec0.array()); assert e < _TOL

    # With s = 0 we have the identity
    hs_norm = interp_norm(a=a_form, bcs=bc, s=0.0)
    y0 = hs_norm(vec)
    e = np.linalg.norm(vec.array() - y0.array()); assert e < _TOL 

    ###################
    # Full GEVP problem
    ###################
    M, _ = assemble_system(m_form, L_form, bc)
    M = M.array()

    I, Ip = reduced_gevp(a_form, bcs=bc, m=m_form)

    eigw = np.sort(np.r_[I[0], Ip[0]])

    info('Solving GEVP of size %d' % len(A))
    timer = Timer('GEVP')
    eigw0, eigv0 = eigh(A, M)
    t0 = timer.stop()
    info('Done in %g' % t0)

    e = np.linalg.norm(eigw-np.sort(eigw0)); assert e < _TOL, e

    f = Function(V)
    vec = f.vector()
    as_backend_type(vec).vec().setRandom()

    # With s = 1 we have hs_norm == A
    hs_norm = interp_norm(a=a_form, bcs=bc, s=1.0, m=m_form)
    y = A.dot(vec.array())
    y0 = hs_norm(vec)
    e = np.linalg.norm(y - y0.array()); assert e < _TOL

    # With s = -1 we have inv(A)
    hs_norm = interp_norm_inv(a=a_form, bcs=bc, s=1.0, m=m_form)
    vec0 = hs_norm(y0)
    e = np.linalg.norm(vec.array() - vec0.array()); assert e < _TOL, e

    # With s = 0 we have the identity
    hs_norm = interp_norm(a=a_form, bcs=bc, s=0.0, m=m_form)
    y = M.dot(vec.array())
    y0 = hs_norm(vec)
    e = np.linalg.norm(y - y0.array()); assert e < _TOL 

    # FIXME: compare this Hs norm with that of the pure trace space (CG1)
    #        Poisson
    #        (start on mixed Poisson)
    # ----------------------------------
    #        begin MINRES paper review
    #        ibsc
