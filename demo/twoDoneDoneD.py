# |---|---| Omega is (0, 1)^2, the vertical line is Gamma
# |   |   |
# |   |   | 
# |---|---|
#
# With some handwaving we consider the following coupled problem 
#
#     -div(kappa*grad u) + p*delta_Gamma = f  in Omega
#     -div(kappa1*grad u1) - p           = f1 on Gamma
#     u - u1 = g                              on Gamma
#
# closed by Dirichlet boundary conditions on u, u1 on their respective
# boundaries
from utils import PiecewiseExpression
import sympy as sp
from dolfin import *
from xii import *
import ulfy


def get_geometry(i):
    '''
    Get cell function with (0, 1)^2 divided by Gamma, and facet function
    marking boundaries and Gamma
    '''
    n = 4*2**i

    mesh = UnitSquareMesh(n, n)
    cell_f = MeshFunction('size_t', mesh, 2, 1)  # 1 is the left piece
    CompiledSubDomain('x[0] > 0.5-DOLFIN_EPS').mark(cell_f, 2)

    facet_f = MeshFunction('size_t', mesh, 1, 0)
    boundaries = {1: 'near(x[0], 0)',
                  2: 'near(x[0], 1)',
                  3: 'near(x[1], 0) && x[0] < 0.5+DOLFIN_EPS',
                  4: 'near(x[1], 0) && x[0] > 0.5-DOLFIN_EPS',
                  5: 'near(x[1], 1) && x[0] < 0.5+DOLFIN_EPS',
                  6: 'near(x[1], 1) && x[0] > 0.5-DOLFIN_EPS',
                  7: 'near(x[0], 0.5)'}
    [CompiledSubDomain(bdry).mark(facet_f, tag) for tag, bdry in boundaries.items()]
    
    return cell_f, facet_f


def setup_mms(material_parameters):
    # We create a discontinuity on the Gamma by considering
    # differnt solutions on left and right side of it
    # u0 is upper one with boundaries 1, 4, 2, 6
    # u1 is the lower one 3, 5
    omega = UnitSquareMesh(4, 4)
    x, y = SpatialCoordinate(omega)
    # Want continuity on the inteface
    u0 = sin(pi*y*(x-0.5))
    u1 = -sin(pi*y*(x-0.5))

    kappai = Constant(1)
    f0 = -div(kappai*grad(u0))
    f1 = -div(kappai*grad(u1))

    n = Constant((1, 0))
    lm = -(kappai*dot(grad(u0), n) - kappai*dot(grad(u1), n))

    kappa = Constant(1)
    # u will be the 1d solution
    u = sin(pi*y*(1-y))
    # FIXME: Laplace-Beltrami here?
    f = -kappa*u.dx(1).dx(1) - lm

    kappai_, kappa_ = sp.symbols('kappa kappa1')
    subs = {kappai: kappai_, kappa: kappa_}

    print(material_parameters['kappa'],
          material_parameters['kappa1'])
    to_expr = lambda f: ulfy.Expression(f, subs=subs, degree=4,
                                        kappa=material_parameters['kappa'],
                                        kappa1=material_parameters['kappa1'])

    u0_, u1_, u_ = map(to_expr, (u0, u1, u))

    subdomains = {1: CompiledSubDomain('x[0] < 0.5+DOLFIN_EPS'),
                  2: CompiledSubDomain('x[0] > 0.5-DOLFIN_EPS')}
    
    return {
        'solution': {'u': PiecewiseExpression(subdomains, {1: u0_, 2: u1_}),
                     'u1': u_,
                     'lm': to_expr(lm)},
        'f0': to_expr(f0), 'f1': to_expr(f1), 'f': to_expr(f), 
        'dirichlet_omega': {1: u0_, 3: u0_, 5: u0_,
                            2: u1_, 4: u1_, 6: u1_},
        'dirichlet_gamma': u_,
        'g_u': to_expr(u0 - u),
        'get_geometry': get_geometry
    }
    

def setup_problem(i, mms, parameters):
    '''Solution of the 2d-1d-1d'''
    cell_f, facet_f = mms['get_geometry'](i)

    omega = cell_f.mesh()
    dX = Measure('dx', domain=omega, subdomain_data=cell_f)

    gamma = EmbeddedMesh(facet_f, 7)
    dx_ = Measure('dx', domain=gamma)
    
    V = FunctionSpace(omega, 'CG', 1)   # 2d
    V1 = FunctionSpace(gamma, 'CG', 1)  # 1d
    Q = FunctionSpace(gamma, 'CG', 1)  # Multiplier 1d
    W = [V, V1, Q]

    u, u1, p = map(TrialFunction, W)
    v, v1, q = map(TestFunction, W)
    
    Tu, Tv = (Trace(x, gamma) for x in (u, v))

    # Material parameters
    kappa, kappa1 = (Constant(parameters[key]) for key in ('kappa', 'kappa1'))

    a = block_form(W, 2)
    a[0][0] = kappa*inner(grad(u), grad(v))*dx
    a[0][2] = inner(p, Tv)*dx_
    a[1][1] = kappa1*inner(grad(u1), grad(v1))*dx
    a[1][2] = -inner(p, v1)*dx_
    a[2][0] = inner(q, Tu)*dx_
    a[2][1] = -inner(q, u1)*dx_
    
    L = block_form(W, 1)
    L[0] = inner(mms['f0'], v)*dX(1) + inner(mms['f1'], v)*dX(2)
    L[1] = inner(mms['f'], v1)*dx
    L[2] = inner(mms['g_u'], q)*dx

    # Dirichlet on all outside
    V_bcs = [DirichletBC(V, value, facet_f, tag)
             for tag, value in mms['dirichlet_omega'].items()]
    V1_bcs = [DirichletBC(V1, mms['dirichlet_gamma'], 'on_boundary')]
    Q_bcs = [DirichletBC(Q, Constant(0), 'on_boundary')]

    W_bcs = [V_bcs, V1_bcs, Q_bcs]
    
    return a, L, W, W_bcs

# --------------------------------------------------------------------

if __name__ == '__main__':
    from common import ConvergenceLog, H1_norm, L2_norm, broken_norm
    import sys, argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Decide material parameters ...
    parser.add_argument('--param_kappa', type=float, default=1, help='2d diffusion')
    parser.add_argument('--param_kappa1', type=float, default=1, help='1d diffusion')
    args, _ = parser.parse_known_args()

    # Reduce verbosity
    set_log_level(40)
    # For checking convergence we pick the solution of the test case ...
    material_params = {k.split('_')[-1] : v for k, v in vars(args).items() if k.startswith('param_')}
    mms = setup_mms(material_params)
    u_true, u1_true, lm_true = (mms['solution'][k] for k in ('u', 'u1', 'lm'))

    clog = ConvergenceLog({'u': (u_true.expressions, broken_norm(u_true.subdomains, H1_norm), '1'),
                           'u1': (u1_true, H1_norm, '1'),
                           'lm': (lm_true, L2_norm, '0')})

    print(clog.header())
    for i in range(6):
        a, L, W, bcs = setup_problem(i, mms, parameters=material_params)
        # Use direct solver to get the solution
        A, b = map(ii_assemble, (a, L))
        A, b = apply_bc(A, b, bcs)
        A, b = map(ii_convert, (A, b))

        wh = ii_Function(W)
        LUSolver(A, 'mumps').solve(wh.vector(), b)

        uh, u1h, lmh = wh
        
        clog.add((uh, u1h, lmh))
        print(clog.report_last(with_name=False))

    urates = tuple(clog[var].get_rate()[0] for var in ('u', 'u1'))
    prate, _ = clog['lm'].get_rate()
    
    expected = (1, 1)
    passed = all(abs(r-e) < 0.1 for r, e in zip(urates, expected)) and prate > 0.4

    sys.exit(int(passed))
