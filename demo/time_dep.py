from dolfin import *
from xii import *


def heat(n, dt, f, u0, gD):
    '''BE u_t - (u_xx + u_yy) = f with u = gD on bdry and u(0, x) = u0'''
    mesh = UnitSquareMesh(n, n)
    facet_f = MeshFunction('size_t', mesh, 1, 0)
    CompiledSubDomain('near(x[0], 0)').mark(facet_f, 1)
    CompiledSubDomain('near(x[0], 1)').mark(facet_f, 2)
    CompiledSubDomain('near(x[1], 0)').mark(facet_f, 3)
    CompiledSubDomain('near(x[1], 1)').mark(facet_f, 4)

    bmesh = EmbeddedMesh(facet_f, (1, 2, 3))

    V = FunctionSpace(mesh, 'CG', 1)
    Q = FunctionSpace(bmesh, 'CG', 1)
    W = [V, Q]

    u, p = list(map(TrialFunction, W))
    v, q = list(map(TestFunction, W))

    dx_ = Measure('dx', domain=bmesh)
    Tu, Tv = Trace(u, bmesh), Trace(v, bmesh)

    dt = Constant(dt)

    a = block_form(W, 2)
    a[0][0] = inner(grad(u), grad(v))*dx + (1/dt)*inner(u, v)*dx
    a[0][1] = inner(p, Tv)*dx_
    a[1][0] = inner(q, Tu)*dx_

    u0 = interpolate(u0, V)

    L = block_form(W, 1)
    L[0] = (1/dt)*inner(u0, v)*dx + inner(f, v)*dx
    L[1] = inner(gD, q)*dx_

    bcs = [[DirichletBC(V, gD, facet_f, 4)],
           [DirichletBC(Q, Constant(0), 'on_boundary')]]

    AA, bb = list(map(ii_assemble, (a, L)))
    AA, bb, apply_b = apply_bc(AA, bb, bcs, return_apply_b=True)

    wh = ii_Function(W)

    A = ii_convert(AA)
    print(('Symmetry', as_backend_type(A).mat().isHermitian(1E-4)))
    solver = LUSolver(A)
    
    t = 0
    while t < 1:
        t += dt(0)
        gD.t = t
        f.t = t

        bb = ii_assemble(L)
        apply_b(bb)

        solver.solve(wh.vector(), ii_convert(bb))
        u0.assign(wh[0])


    return t, u0

# --------------------------------------------------------------------

if __name__ == '__main__':
    import sympy as sp

    x, y, t = sp.symbols('x[0] x[1] t')

    u = sp.sin(sp.pi*x*(x**2 + y**2)*t)
    f = u.diff(t, 1) - (u.diff(x, 2) + u.diff(y, 2))
    # Wrap
    u = Expression(sp.printing.ccode(u).replace('M_PI', 'pi'), t=0, degree=4)
    f = Expression(sp.printing.ccode(f).replace('M_PI', 'pi'), t=0, degree=4)

    table = []
    dt = 1E-1
    for nrefs in range(4):
        dt_row = []
        for n in (8, 16, 32, 64):
            u.t, f.t = 0., 0.
            
            t, uh = heat(n, dt, f=f, u0=u, gD=u)
            
            u.t = t
            dt_row.append(errornorm(u, uh, 'H1'))
        table.append(dt_row)
        print((dt, '->', dt_row))
        dt /= 2.
