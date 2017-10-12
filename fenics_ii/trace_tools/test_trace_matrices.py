from collections import namedtuple
from embedded_mesh import EmbeddedMesh
from dolfin import *


def test(tmat, args, which):
    '''The trace matrix is defined with tmat(args) and checked'''
    for case in [c2d_1d, c3d_2d, c3d_1d]:
        setup = case(which)

        # Check Lagrange
        V = args[0][0](setup.omega_mesh, *args[0][1:])
        TV = args[1][0](setup.gamma_mesh.mesh, *args[1][1:])   # What would be the trace space

        if len(args) > 2:
            Tmat = tmat(V, TV, *args[2])
        else:
            Tmat = tmat(V, TV)

        f = setup.f
        x = as_backend_type(interpolate(f, V).vector()).vec()
        y0 = as_backend_type(interpolate(f, TV).vector()).vec()

        y = y0.copy()
        Tmat.mult(x, y)

        y0.axpy(-1, y)
        n = y0.norm(0)
        print Tmat.size
        assert n < 1E-10, n

        # plot(f, mesh=setup.gamma_mesh.mesh)

        # Tf = Function(TV, PETScVector(y))
        # plot(Tf)
        # interactive()


# Cases ----------------------------------------------------------------------

Setup = namedtuple('Setup', ['omega_mesh', 'gamma_mesh', 'f'])

def c2d_1d(which):
    gamma = ['near((x[0]-0.25)*(x[0]-0.75), 0) && (0.25-DOLFIN_EPS < x[1]) && (x[1] < 0.75+DOLFIN_EPS)',
             'near((x[1]-0.25)*(x[1]-0.75), 0) && (0.25-DOLFIN_EPS < x[0]) && (x[0] < 0.75+DOLFIN_EPS)']
    gamma = map(lambda x: '('+x+')', gamma)
    gamma = ' || '.join(gamma)
    gamma = CompiledSubDomain(gamma)

    n = 1
    n *= 4
    omega_mesh = UnitSquareMesh(n, n)
    facet_f = FacetFunction('size_t', omega_mesh, 0)
    gamma.mark(facet_f, 1)

    gamma_mesh = EmbeddedMesh(omega_mesh, facet_f, 1)

    if which == 'scalar':
        f = Expression('sin(pi*(x[0]+x[1]))', degree=1)
    elif which == 'vector':
        #f = Expression(('sin(pi*(x[0]+x[1]))',
        #                'cos(pi*(x[0]+x[1]))'), degree=1)
        f = Constant((1, 2))

    return Setup(omega_mesh, gamma_mesh, f)


def c3d_2d(which):
    base = UnitCubeMesh(10, 10, 10)

    gamma2 = CompiledSubDomain('near(x[2], 0.5) || near(x[0], 0.)')
    domain_f2 = FacetFunction('size_t', base, 0)
    gamma2.mark(domain_f2, 1)

    emesh2 = EmbeddedMesh(base, domain_f2, 1)

    if which == 'scalar':
        f = Expression('sin(pi*(x[0]+x[1]+x[2]))', degree=1)
    elif which == 'vector':
        f = Expression(('sin(pi*(x[0]+x[1]+x[2]))', 
                        'cos(pi*(x[0]+x[1]+x[2]))',
                        'exp(pi*(x[0]+x[1]+x[2]))'), degree=1)

        f = Constant((1, 2, 3))

    return Setup(base, emesh2, f)


def c3d_1d(which):
    base = UnitCubeMesh(10, 10, 10)

    gamma1 = ['(near(x[0], x[1]) && near(x[1], x[2]))',
              '(near(x[0], 1) && near(x[1], 1))',
              '(near(x[0], x[1]) && near(x[2], 0))']
    gamma1 = ' || '.join(gamma1)
    gamma1 = CompiledSubDomain(gamma1)

    domain_f1 = EdgeFunction('size_t', base, 0)
    gamma1.mark(domain_f1, 1)

    emesh1 = EmbeddedMesh(base, domain_f1, 1)

    if which == 'scalar':
        f = Expression('sin(pi*(x[0]+x[1]+x[2]))', degree=1)
    elif which == 'vector':
        f = Expression(('sin(pi*(x[0]+x[1]+x[2]))', 
                        'cos(pi*(x[0]+x[1]+x[2]))',
                        'exp(pi*(x[0]+x[1]+x[2]))'), degree=1)

    return Setup(base, emesh1, f)

# ----------------------------------------------------------------------------

def Hdiv_test(tmat, args):
    '''The trace matrix is defined with tmat(args) and checked'''
    for ii, case in enumerate([one], 1):
        print '-'*79
        print ii
        print '-'*79
        for which in (0, 1):
            e0_L2, e0_L8, h0 = -1, -1, -1
            rate_L2, rate_L8 = -1, -1
            for n in [10, 20, 40, 80]:
                setup = case(n, which)

                # Check Lagrange
                V = args[0][0](setup.omega_mesh, *args[0][1:])
                TV = args[1][0](setup.gamma_mesh.mesh, *args[1][1:])   # What would be the trace space

                Tmat = tmat(V, TV, 
                            (setup.gamma_mesh.normal,
                             setup.gamma_mesh.entity_map,
                             '+'))

                f = setup.f
                Tf = setup.Tf
                x = as_backend_type(interpolate(f, V).vector()).vec()
                y0 = as_backend_type(interpolate(Tf, TV).vector()).vec()

                y = y0.copy()
                Tmat.mult(x, y)

                y = Function(TV, PETScVector(y))
                y0 = Function(TV, PETScVector(y0))
                e_L2 = sqrt(abs(assemble((y-y0)**2*dx)))

                y0.vector().axpy(-1, y.vector())
                e_L8 = y0.vector().norm('linf')

                h = setup.gamma_mesh.mesh.hmin()

                if e0_L2 > -1:
                    try:
                        rate_L2 = ln(e_L2/e0_L2)/ln(h/h0)
                    except:
                        rate_L2 = -1
                    
                    try:
                        rate_L8 = ln(e_L8/e0_L8)/ln(h/h0)
                    except:
                        rate_L8 = -1
                e0_L2, e0_L8, h0 = e_L2, e_L8, h
                
                print which, '-->', Tmat.size, (e_L2, rate_L2), (e_L8, rate_L8)
        # assert n < 1E-10, n
                

Hdiv_setup = namedtuple('SSetup', ['omega_mesh', 'gamma_mesh', 'f', 'Tf'])


def one(n, which):
    base = UnitSquareMesh(n, n)
    gamma = CompiledSubDomain('near(x[0], 0.5)')
    domain_f = FacetFunction('size_t', base, 0)
    gamma.mark(domain_f, 1)
    normal = Expression(('1', '0'), degree=1)
    emesh = EmbeddedMesh(base, domain_f, 1, normal)
    # Exact
    if which == 0:
        f = Expression(('x[0]', 'x[1]'), degree=1)
        Tf = Expression('0.5', degree=1)
    elif which == 1:
        f = Expression(('sin(x[0])', 'sin(x[1])'), degree=1)
        Tf = Expression('sin(x[0])', degree=1)

    return Hdiv_setup(base, emesh, f, Tf)


def two(n, which):
    base = UnitSquareMesh(n, n)
    gamma = CompiledSubDomain('near(x[0], x[1])')
    domain_f = FacetFunction('size_t', base, 0)
    gamma.mark(domain_f, 1)
    normal = Expression(('-1', '1'), degree=1)
    emesh = EmbeddedMesh(base, domain_f, 1, normal)

    if which == 0:
        f = Expression(('-x[0]', 'x[1]'), degree=1)
        Tf = Expression('2*x[0]', degree=1)
    else:
        f = Expression(('-x[0]*x[1]', 'x[1]*x[0]'), degree=2)
        Tf = Expression('2*x[0]*x[0]', degree=2)

    return Hdiv_setup(base, emesh, f, Tf)
