from dolfin import *
from trace_tools.trace_assembler import trace_assemble
from trace_tools.norms import inv_interpolation_norm_eig
from trace_tools.embedded_mesh import EmbeddedMesh
from block import block_mat, block_vec
from block.iterative import MinRes
from block.algebraic.petsc import AMG


gamma = ['near((x[0]-0.25)*(x[0]-0.75), 0) && (0.25-DOLFIN_EPS < x[1]) && (x[1] < 0.75+DOLFIN_EPS)',
         'near((x[1]-0.25)*(x[1]-0.75), 0) && (0.25-DOLFIN_EPS < x[0]) && (x[0] < 0.75+DOLFIN_EPS)']
gamma = map(lambda x: '('+x+')', gamma)
gamma = ' || '.join(gamma)
gamma = CompiledSubDomain(gamma)

f = Expression('sin(pi*(x[0]-x[1]))', degree=3)
g = Expression('cos(pi*(x[0]+x[1]))', degree=3)


def main(n):
    '''
    Minimize 0.5*inner(grad(u), grad(u))*dx+0.5*inner(u, u)*dx-inner(f, v) over 
    H^1(Omega) with Neumann bcs subject to Tu = g on Gamma
    '''
    n *= 4
    omega_mesh = UnitSquareMesh(n, n)
    facet_f = FacetFunction('size_t', omega_mesh, 0)
    gamma.mark(facet_f, 1)

    gamma_mesh = EmbeddedMesh(omega_mesh, facet_f, 1)

    # Space of u and the Lagrange multiplier
    V = FunctionSpace(omega_mesh, 'CG', 1)
    Q = FunctionSpace(gamma_mesh.mesh, 'CG', 1)

    u, p = TrialFunction(V), TrialFunction(Q)
    v, q = TestFunction(V), TestFunction(Q)

    dxGamma = Measure('dx', domain=gamma_mesh.mesh)

    a00 = inner(grad(u), grad(v))*dx + inner(u, v)*dx
    a01 = inner(p, v)*dxGamma
    a10 = inner(u, q)*dxGamma
    L0 = inner(f, v)*dx
    L1 = inner(g, q)*dxGamma

    # Blocks
    A00 = assemble(a00)
    A01 = trace_assemble(a01, gamma_mesh)
    A10 = trace_assemble(a10, gamma_mesh)
    b0 = assemble(L0)
    b1 = assemble(L1)
    # System
    AA = block_mat([[A00, A01], [A10, 0]])
    bb = block_vec([b0, b1])

    # Preconditioner blocks
    P00 = AMG(A00)
    # Trace of H^1 is H^{1/2} and the dual is H^{-1/2}
    m = inner(p, q)*dxGamma
    a = inner(grad(p), grad(q))*dxGamma + m
    P11 = inv_interpolation_norm_eig(a, s=-0.5, m=m)
    # The preconditioner
    BB = block_mat([[P00, 0], [0, P11]])

    AAinv = MinRes(AA, precond=BB, tolerance=1e-10, maxiter=500, show=0)
    # Compute solution
    U, P = AAinv * bb
    uh, ph = Function(V, U), Function(Q, P)
    niters = len(AAinv.residuals) - 1
    size = '%d + %d' % (V.dim(), Q.dim())

    return size, niters, uh, ph

# ---------------------------------------------------------------------------

if __name__ == '__main__':
    from trace_tools.plotting import line_plot
    import matplotlib.pyplot as plt

    for n in [2, 4, 8, 16, 32, 64, 128]:
        size, niters, uh, ph = main(n)
        print '\033[1;37;31m%s\033[0m' % ('size = %s, niters = %d' % (size, niters))

    plot(uh, interactive=True)
    fig = line_plot(ph)
    plt.show()
