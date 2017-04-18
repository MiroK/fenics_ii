import sys
sys.path.append('..')
from trace_tools.trace_assembler import trace_assemble
from trace_tools.embedded_mesh import EmbeddedMesh
from norms import H1_L2_InterpolationNorm
from block import block_mat, block_vec
from block_eig import identity_matrix
from dolfin import *


def mp_system(ncells, precond, full=False):
    '''Mixed Poisson.'''
    mesh = UnitSquareMesh(ncells, ncells)
    V = FunctionSpace(mesh, 'BDM', 1)  # sigma
    Q = FunctionSpace(mesh, 'DG', 0)   # u
    W = [V, Q]

    sigma, u = map(TrialFunction, W)
    tau, v = map(TestFunction, W)

    # System
    a00 = inner(sigma, tau)*dx
    a01 = inner(u, div(tau))*dx
    a10 = inner(div(sigma), v)*dx

    A00 = assemble(a00)
    A01 = assemble(a01)
    A10 = assemble(a10)

    AA = block_mat([[A00, A01], [A10, 0]])

    # Preconditioner for L2 x H1 formulation
    if precond == 'L2':
        p00 = inner(sigma, tau)*dx

        n = FacetNormal(mesh)
        h = CellSize(mesh)
        h_avg = avg(h)
        p11 = h_avg**(-1)*dot(jump(v, n), jump(u, n))*dS +\
              h**(-1)*dot(u, v)*ds +\
              inner(u, v)*dx

        P00 = assemble(p00)
        P11 = assemble(p11)
    # Preconditioner for Hdiv x L2
    elif precond == 'Hdiv':
        p00 = inner(sigma, tau)*dx + inner(div(sigma), div(tau))*dx
        p11 = inner(u, v)*dx

        P00 = assemble(p00)
        P11 = assemble(p11)
    else:
        P00 = identity_matrix(V.dim(), comm=mesh.mpi_comm().tompi4py())
        P11 = identity_matrix(Q.dim(), comm=mesh.mpi_comm().tompi4py())

    BB = block_mat([[P00, 0], [0, P11]])

    if not full:
        return AA, BB
    # Add the rhs and spaces:
    else:
        L0 = inner(Constant((0, 0)), tau)*dx
        f = Expression('sin(pi*(x[0]-x[1]))', degree=3)
        L1 = -inner(f, v)*dx
        b = map(assemble, (L0, L1))
        b = block_vec(b)

        return AA, BB, b, W


def t_system(ncells, precond, full=False):
    '''Min 0.5*(sigma, sigma) - (f, sigma) in Omega with sigma.n = g on Gamma.'''
    Gamma = ['near((x[0]-0.25)*(x[0]-0.75), 0) && (0.25-DOLFIN_EPS < x[1]) && (x[1] < 0.75+DOLFIN_EPS)',
             'near((x[1]-0.25)*(x[1]-0.75), 0) && (0.25-DOLFIN_EPS < x[0]) && (x[0] < 0.75+DOLFIN_EPS)']
    Gamma = map(lambda x: '('+x+')', Gamma)
    Gamma = ' || '.join(Gamma)
    Gamma = CompiledSubDomain(Gamma)

    intGamma = AutoSubDomain(lambda x, on_boundary:\
                             between(x[0], (0.25, 0.75)) and between(x[1], (0.25, 0.75)))

    omega = UnitSquareMesh(4*ncells, 4*ncells)
    facet_f = FacetFunction('size_t', omega, 0)
    Gamma.mark(facet_f, 1)
    gamma = EmbeddedMesh(omega, facet_f, 1, normal=Point(0.5, 0.5))
    cell_f = CellFunction('size_t', omega, 0)
    intGamma.mark(cell_f, 1)

    dx = Measure('dx', domain=omega, subdomain_data=cell_f)
    dxGamma = Measure('dx', domain=gamma.mesh)
    n = gamma.normal

    V = FunctionSpace(omega, 'BDM', 1)  # sigma
    Q = FunctionSpace(gamma.mesh, 'DG', 0)   # u
    W = [V, Q]

    sigma, u = map(TrialFunction, W)
    tau, v = map(TestFunction, W)

    # System
    a00 = inner(sigma, tau)*dx
    a01 = inner(dot(tau('+'), n('+')), u)*dxGamma
    a10 = inner(dot(sigma('+'), n('+')), v)*dxGamma

    A00 = assemble(a00)
    A01 = trace_assemble(a01, gamma)
    A10 = trace_assemble(a10, gamma)

    AA = block_mat([[A00, A01], [A10, 0]])

    if precond == 'one':
        n = FacetNormal(omega)
        # NOTE: zero why?
        # p00 = inner(dot(tau('+'), n('+')), dot(sigma('+'), n('+')))*dS(1)
        
        p00 = inner(tau, sigma)*dx
        P00 = assemble(p00)

        p11 = inner(u, v)*dxGamma
        P11 = H1_L2_InterpolationNorm(W[1]).get_s_norm(s=0, as_type=PETScMatrix)
        # L2 norm gives
        # 120 1 3.85224 3.85224
        # 432 1 5.1719 5.1719
        # 1632 1 7.08771 7.08771
        # 6336 1 9.80301 9.80301
        # 24960 1 13.6469 13.6469
        # 99072 1 19.0858 19.0858

        P11 = identity_matrix(W[1].dim())
        # And this guy!!!
        # 120 1 2.23115 2.23115
        # 432 1 2.21671 2.21671
        # 1632 1 2.21661 2.21661
        # 6336 1 2.21661 2.21661
        # 24960 1 2.21661 2.21661
        # 99072 1 2.21661 2.21661
    else:
        comm = omega.mpi_comm().tompi4py()
        P00 = identity_matrix(V.dim(), comm=comm)
        P11 = identity_matrix(Q.dim(), comm=comm)

    BB = block_mat([[P00, 0], [0, P11]])

    if not full:
        return AA, BB
    else:
        f0 = Expression(('cos(pi*x[0])', 'cos(pi*x[1])'), degree=3)
        L0 = inner(f0, tau)*dx

        f1 = Expression('sin(pi*(x[0]-x[1]))', degree=3)
        L1 = inner(f1, v)*dx
        
        b = map(assemble, (L0, L1))
        b = block_vec(b)

        return AA, BB, b, W, gamma


def t_system_meg(ncells):
    '''
    Min 0.5*(sigma, sigma) - (f, sigma) in Omega with sigma.n = g on Gamma, is
    solved here with DGT
    '''
    Gamma = ['near((x[0]-0.25)*(x[0]-0.75), 0) && (0.25-DOLFIN_EPS < x[1]) && (x[1] < 0.75+DOLFIN_EPS)',
             'near((x[1]-0.25)*(x[1]-0.75), 0) && (0.25-DOLFIN_EPS < x[0]) && (x[0] < 0.75+DOLFIN_EPS)']
    Gamma = map(lambda x: '('+x+')', Gamma)
    Gamma = ' || '.join(Gamma)
    Gamma = CompiledSubDomain(Gamma)

    intGamma = AutoSubDomain(lambda x, on_boundary:\
                             between(x[0], (0.25, 0.75)) and between(x[1], (0.25, 0.75)))

    omega = UnitSquareMesh(4*ncells, 4*ncells)

    facet_f = FacetFunction('size_t', omega, 0)
    Gamma.mark(facet_f, 1)

    cell_f = CellFunction('size_t', omega, 0)
    intGamma.mark(cell_f, 1)

    n = FacetNormal(omega)

    V = FiniteElement('BDM', omega.ufl_cell(), 1) 
    Q = FiniteElement('DGT', omega.ufl_cell(), 0)
    W = MixedElement([V, Q])
    W = FunctionSpace(omega, W)

    sigma, u = TrialFunctions(W)
    tau, v = TestFunctions(W)

    dx = Measure('dx', subdomain_data=cell_f)
    dS = Measure('dS', subdomain_data=facet_f)
    # System
    a = inner(sigma, tau)*dx() + \
        inner(dot(tau('+'), n('+')), u('+'))*dS(1) +\
        inner(dot(sigma('+'), n('+')), v('+'))*dS(1)

    f0 = Expression(('cos(pi*x[0])', 'cos(pi*x[1])'), degree=3)
    f1 = Expression('sin(pi*(x[0]-x[1]))', degree=3)

    L = inner(f0, tau)*dx() + inner(f1('+'), v('+'))*dS(1)

    bc = DirichletBC(W.sub(1), Constant(0), facet_f, 0)

    wh = Function(W)
    solve(a == L, wh, bc)

    sigmah, uh = wh.split(deepcopy=True)

    return sigmah, uh

# ----------------------------------------------------------------------------

if __name__ == '__main__':
    from direct import dolfin_solve
    from trace_tools.plotting import line_plot
    import matplotlib.pyplot as plt
    from norms import DGT_DG_error, H1_L2_InterpolationNorm
    from collections import defaultdict

    sigma0, p0 = t_system_meg(ncells=32)
    hs = defaultdict(list)
    hs_inv = defaultdict(list)

    for ncells in [4, 8, 16, 32]:
        AA, _, b, W, gamma = t_system(ncells, precond=None, full=True)
        sigmah, ph = dolfin_solve(AA, b, method='default', spaces=W)

        es = inner(sigmah, sigmah)*dx
        ep = inner(ph, ph)*dx  # Probably not the right norm

        print map(sqrt, map(abs, map(assemble, (es, ep))))

        # Lets see if we can find a norm in which ph converges
        I = H1_L2_InterpolationNorm(W[-1])
        s = [-1, -0.5, 0, 0.5, 1]
        hs_norms = map(lambda si: I.get_s_norm(si, as_type=None), s)
        hs_norms_inv = map(lambda si: I.get_s_norm_inv(si, as_type=None), s)

        for si, norm_f in zip(s, hs_norms): 
            print 'Hs_norm', si, norm_f(ph)
            hs[si].append(norm_f(ph))

        for si, norm_f in zip(s, hs_norms_inv): 
            print 'Hs_norm_inv', si, norm_f(ph)
            hs_inv[si].append(norm_f(ph))
    
    mesh = sigma0.function_space().mesh()
    dx = dx(domain=mesh)
    print assemble(inner(sigmah-sigma0, sigmah-sigma0)*dx)
    print DGT_DG_error(p0, ph, gamma)

    # NOTE Seems like all the inv guys
    print 
    print 'HS'
    for s in hs: print s, hs[s]

    print 'inv(HS)'
    for s in hs: print s, hs_inv[s]
    # plot(sigmah)
    # plot(sigma0)
    # interactive()
