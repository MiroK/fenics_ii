from dolfin import *
from xii import *
from xii.assembler.eval_dlt_trace_matrix import embedding_mat

if True:
    mesh = UnitSquareMesh(5, 5)

    facet_f = MeshFunction('size_t', mesh, 1, 0)
    CompiledSubDomain('near(x[0]*(1-x[0]), 0)').mark(facet_f, 1)
    trace_mesh = EmbeddedMesh(facet_f, 1)

    V = FunctionSpace(mesh, 'HDiv Trace', 1)

    TV = FunctionSpace(trace_mesh, 'DG', 1)

    E = embedding_mat(V, TV, trace_mesh=trace_mesh)

    f = Expression('3*x[0] + 2*x[1]', degree=1)

    u, v = TrialFunction(V), TestFunction(V)
    a = inner(u, v)*ds + inner(u('+'), v('+'))*dS
    L = inner(f, v)*ds + inner(f, v('+'))*dS

    uh = Function(V)
    solve(a == L, uh)

    e = assemble(inner(f-uh, f-uh)*ds + inner(f-uh('+'), f-uh('+'))*dS)
    print('projection error', e)

    Tv_vec = E*uh.vector()
    Tv = Function(TV)
    Tv.vector()[:] = Tv_vec

    Tv0 = interpolate(f, TV)

    print(assemble(inner(f-Tv0, f-Tv0)*dx))
    print(assemble(inner(f-Tv, f-Tv)*dx))
    print(assemble(inner(Tv, Tv)*dx))
    print((Tv0.vector() - Tv.vector()).norm('l2'))

# ---------

mesh = UnitSquareMesh(32, 32)

cell_f = MeshFunction('size_t', mesh, 2, 0)
CompiledSubDomain('x[0] > 0.25-DOLFIN_EPS').mark(cell_f, 1)

facet_f = MeshFunction('size_t', mesh, 1, 0)
CompiledSubDomain('near(x[0], 0.25)').mark(facet_f, 1)

left = EmbeddedMesh(cell_f, 0)
left_boundaries = left.translate_markers(facet_f, (1, ))

bmesh = EmbeddedMesh(left_boundaries, 1)
print(bmesh.parent_meshes)

right = EmbeddedMesh(cell_f, 1)
right_boundaries = right.translate_markers(facet_f, (1, ))
bmesh.compute_embedding(right_boundaries, 1)

if True:
    Vl = VectorFunctionSpace(left, 'CG', 2)
    Vr = VectorFunctionSpace(right, 'CG', 2)

    ul = TrialFunction(Vl)

    n = Constant((-1, 0))

    foo = lambda arg: dot(sym(grad(arg)), n)
    Tul = Trace(foo(ul), bmesh)

    vr = TestFunction(Vr)
    Tvr = Trace(vr, bmesh)

    dx_ = Measure('dx', domain=bmesh)

    form = inner(Tvr, Tul)*dx_

    M = ii_assemble(form)

    import ulfy

    x, y = SpatialCoordinate(mesh)

    f = as_vector((x**2+x - 3*y**2, 2*x**2 + x*y + y**2))
    g = as_vector((x, y))
    u_expr = ulfy.Expression(foo(f), degree=2)
    v_expr = ulfy.Expression(g, degree=2)
    f_expr, g_expr = (ulfy.Expression(arg, degree=2) for arg in (f, g))

    true = assemble(inner(v_expr, u_expr)*dx_)

    mine = (M*interpolate(f_expr, Vl).vector()).inner(interpolate(g_expr, Vr).vector())

    print(true, mine, abs(true - mine))            

if True:
    Vl = FunctionSpace(left, 'CG', 2)
    Vr = FunctionSpace(right, 'CG', 2)

    ul = TrialFunction(Vl)

    n = Constant((-1, 0))

    foo = lambda arg: dot(grad(arg), n)
    Tul = Trace(foo(ul), bmesh)

    vr = TestFunction(Vr)
    Tvr = Trace(vr, bmesh)

    dx_ = Measure('dx', domain=bmesh)

    form = inner(Tvr, Tul)*dx_

    M = ii_assemble(form)

    import ulfy

    x, y = SpatialCoordinate(mesh)

    f = 2*x**2 - y**2
    g = x + 2*x*y
    u_expr = ulfy.Expression(foo(f), degree=2)
    v_expr = ulfy.Expression(g, degree=2)
    f_expr, g_expr = (ulfy.Expression(arg, degree=2) for arg in (f, g))

    true = assemble(inner(v_expr, u_expr)*dx_)

    mine = (M*interpolate(f_expr, Vl).vector()).inner(interpolate(g_expr, Vr).vector())

    print(true, mine, abs(true - mine))    
