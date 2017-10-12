from dolfin import *
from trace_tools.trace_assembler import trace_assemble
from trace_tools.norms import inv_interpolation_norm_eig, interpolation_norm_eig
from trace_tools.embedded_mesh import EmbeddedMesh
from trace_tools.pcstr import PointConstraintData
from utils.block_eig import identity_matrix
from utils.direct import dolfin_solve
from block import block_mat, block_vec
from block.iterative import MinRes
from block.algebraic.petsc import AMG
import numpy as np

def add(*subdomains):
    '''OR the domains.'''
    body = lambda x, on_boundary:\
            any(subdomain.inside(x, on_boundary) for subdomain in subdomains)
    return AutoSubDomain(body)

left = CompiledSubDomain('near(x[0], 0) && on_boundary')
right = CompiledSubDomain('near(x[0], 1) && on_boundary')
top = CompiledSubDomain('near(x[1], 1) && on_boundary')
bottom = CompiledSubDomain('near(x[1], 0) && on_boundary')

lrtb = add(left, right, top, bottom)
lrt = add(left, right, top)
lb, tr = add(left, bottom), add(top, right)

# FIXME: which of these gives bounded iters, why?
# FIXME: hook upu to direct solver

cases = {'all': ([lrtb], []),
         'lrt, b': ([lrt, bottom], [(0, 1, np.array([0., 0])), 
                                    (0, 1, np.array([1., 0]))]),
         'lb, rt': ([lb, tr], [(0, 1, np.array([0., 1])), 
                               (0, 1, np.array([1., 0]))]),
         'lb, r, t': ([lb, right, top], [(0, 1, np.array([1., 0])), 
                                         (1, 2, np.array([1., 1])),
                                         (2, 0, np.array([0., 1.]))]),
         'l, b, r, t': ([left, bottom, right, top], [(0, 1, np.array([0., 0])), 
                                                     (1, 2, np.array([1., 0])),
                                                     (2, 3, np.array([1., 1.])),
                                                     (0, 3, np.array([0., 1.]))])}

case = 'lb, r, t'
gamma, pcstr = cases[case]

u_exact = Expression('sin(pi*(x[0]-x[1]))', degree=4)
g = u_exact
f = Expression('A*u', u=u_exact, A=(2*pi**2+1), degree=3)


def main(n, solver='iterative', eigs=False):
    '''
    Minimize 0.5*inner(grad(u), grad(u))*dx+0.5*inner(u, u)*dx-inner(f, v)*dx over 
    H^1(Omega) with Neumann bcs subject to Tu = g on Gamma
    '''
    n *= 4
    omega_mesh = UnitSquareMesh(n, n)
    # Isolate 2d things
    V = FunctionSpace(omega_mesh, 'CG', 1)
    u, v = TrialFunction(V), TestFunction(V)
    a00 = inner(grad(u), grad(v))*dx + inner(u, v)*dx
    L0 = inner(f, v)*dx
    # System
    A00 = assemble(a00)
    b0 = assemble(L0)
    # Preconditioner
    P00 = AMG(A00)

    # Coupling
    A01s, A10s, b1s, P11s = [], [], [], []
    Qs = []
    for domain in gamma:
        facet_f = FacetFunction('size_t', omega_mesh, 0)
        domain.mark(facet_f, 1)

        gamma_mesh = EmbeddedMesh(omega_mesh, facet_f, 1)

        # Space of u and the Lagrange multiplier
        Q = FunctionSpace(gamma_mesh.mesh, 'CG', 1)
        Qs.append(Q)
        p, q = TrialFunction(Q), TestFunction(Q)
        dxGamma = Measure('dx', domain=gamma_mesh.mesh)

        a01 = inner(p, v)*dxGamma
        a10 = inner(u, q)*dxGamma
        L1 = inner(g, q)*dxGamma
        # System blocks
        A01 = trace_assemble(a01, gamma_mesh)
        A10 = trace_assemble(a10, gamma_mesh)
        b1 = assemble(L1)
        # Preconditioner
        # Trace of H^1 is H^{1/2} and the dual is H^{-1/2}
        m = inner(p, q)*dxGamma
        a = inner(grad(p), grad(q))*dxGamma + m
        if not eigs:
            P11 = inv_interpolation_norm_eig(a, s=-0.5, m=m)
        else:
            P11 = interpolation_norm_eig(a, s=-0.5, m=m)
        # And collect
        A01s.append(A01)
        A10s.append(A10)
        b1s.append(b1)
        P11s.append(P11)
    # Transform for bmat
    has_pcstr = True
    # No point constraints
    if len(gamma) == 1:
        has_pcstr = False
        # First row
        A01s = [A00] + A01s
        # Remaining rows
        AA = [A01s]
        for col, A10 in enumerate(A10s):
            row = [0]*len(A01s)
            row[0] = A10
            AA.append(row)

        bb = [b0] + b1s

        mats = [P00] + P11s
        BB = []
        for col, mat in enumerate(mats):
            row = [0]*len(mats)
            row[col] = mat
            BB.append(row)
    else:
        assert len(gamma) == len(pcstr)
        # Request the matrices
        Ts, Tts, bX = PointConstraintData(pcstr, Qs)

        # First row
        A01s = [A00] + A01s + [0]
        # Remaining rowhttps://www.google.no/webhp?sourceid=chrome-instant&ion=1&espv=2&ie=UTF-8#q=trygve+berland+uio+duo&*s
        AA = [A01s]
        for col, A10 in enumerate(A10s):
            row = [0]*len(A01s)
            row[0] = A10
            row[-1] = Tts[col]
            AA.append(row)
        AA.append([0] + Ts + [0])

        bb = [b0] + b1s + [bX]

        I = identity_matrix(len(pcstr))
        mats = [P00] + P11s + [I]
        BB = []
        for col, mat in enumerate(mats):
            row = [0]*len(mats)
            row[col] = mat
            BB.append(row)

    # System
    AA = block_mat(AA)
    bb = block_vec(bb)
    # The preconditioner
    BB = block_mat(BB)

    if eigs:
        BB[0][0] = A00
        return AA, BB

    if solver == 'iterative':
        AAinv = MinRes(AA, precond=BB, tolerance=1e-13, maxiter=250, show=0)
        # Compute solution
        x = AAinv * bb
        
        niters = len(AAinv.residuals) - 1
        
        uh = Function(V, x[0])
        phs = [Function(Qi, xi) for Qi, xi in zip(Qs, x[1:])]
    else:
        x = dolfin_solve(AA, bb)
        
        niters = -1
        
        uh = Function(V); uh.vector().set_local(x[0])
        phs = []
        for Qi, xi in zip(Qs, x[1:]):
            ph = Function(Qi)
            ph.vector().set_local(xi)
            phs.append(ph)

    if has_pcstr:
        for index0, index1, point in pcstr:
            values = phs[index0](*point), phs[index1](*point)
            print 'Constraint', values[0], values[1], abs(values[0]-values[1])

    size = [b.size() for b in bb]
    size = ' + '.join(map(str, size))
    h = omega_mesh.hmin()

    return size, niters, uh, h

# ---------------------------------------------------------------------------

if __name__ == '__main__':
    from trace_tools.plotting import line_plot
    import matplotlib.pyplot as plt
    from utils.block_eig import eig, eig_zero
    from numpy import nan
    import numpy as np

    record = open('%s.txt' % case, 'w')
    
    np.set_printoptions(precision=4)
    # Show that this is not a singular system
    record.write('EIGENVALUES\n')
    for n in [2, 4, 8, 16]:
        AA, BB = main(n, eigs=True)

        size, eigenvalues = eig(AA, BB)
        eigenvalues = np.sort(np.abs(eigenvalues))
        lmin, lmax = eigenvalues[[0, -1]]

        print size, lmin, lmax, lmax/lmin, eigenvalues[:10]

        msg = 'size = %d, lmin = %g, lmax = %g' % (size, lmin, lmax)
        msg = msg + 'eigs = ' + ','.join(map(str, eigenvalues[:4]))
        print msg
        record.write(msg+'\n')

        pairs = eig_zero(AA, BB)
        
        for p, vecs in pairs:
            print p, map(lambda x: np.linalg.norm(x)/len(x), vecs)
            # print vecs[1]
            # print vecs[2]
            print
    
    # Show that the system is okay in a sense that it produces converging
    # approaximations (at least to u)
    rate, h0, e0 = -1, -1, -1
    record.write('\nDIRECT\n')
    for n in [2, 4, 8, 16, 32, 64]: # 128, 256, 512, 1024]:
        size, niters, uh, h = main(n, solver='direct')
        e = errornorm(u_exact, uh, 'H1')
        if h0 > 0 and n <= 128:
            rate = ln(e/e0)/ln(h/h0)
        else:
            rate = nan
        e0, h0 = e, h

        msg = 'size = %s, niters = %d, e(rate) = %.4E(%.2f)' % (size, niters, e, rate)
        record.write(msg+'\n')
        print '\033[1;37;31m%s\033[0m' % msg
    # V = uh.function_space()
    # e = interpolate(u_exact, V)
    # e.vector().axpy(-1, uh.vector())
    # plot(uh)
    # plot(e)

    # Show properties of the iterations
    rate, h0, e0 = -1, -1, -1
    record.write('\nITERATIVE\n')
    for n in [2, 4, 8, 16, 32, 64, 128, 256, 512]:
        size, niters, uh, h = main(n, solver='iterative')
        if h0 > 0 and n <= 128:
            e = errornorm(u_exact, uh, 'H1')
            rate = ln(e/e0)/ln(h/h0)
        else:
            e = nan
            rate = nan
        e0, h0 = e, h

        msg = 'size = %s, niters = %d, e(rate) = %.4E(%.2f)' % (size, niters, e, rate)
        record.write(msg+'\n')
        print '\033[1;37;31m%s\033[0m' % msg
    record.close()
    # interactive()
