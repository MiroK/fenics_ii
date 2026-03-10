# FIXME: scaling wrt to radius of the different inner products!
from dolfin import *
from xii import *
import numpy as np

from gmshnics import msh_gmsh_model, mesh_from_gmsh
import gmsh


def unit_cube_with_mesh(A, B, r, resolution=None):
    gmsh.initialize()

    model = gmsh.model
    fac = model.occ

    box = fac.addBox(0, 0, 0, 1, 1, 1)
    cyl = fac.addCylinder(A[0], A[1], A[2], (B-A)[0], (B-A)[1], (B-A)[2], r)

    fac.fragment([(3, box)], [(3, cyl)])
    fac.synchronize()

    volumes = model.getEntities(3)
    bdry0, bdry1 = [model.getBoundary([vol]) for vol in volumes]

    if len(bdry0) < len(bdry1):
        _, cylinder = volumes[0]
        cylinder_boundary = bdry0

        _, box = volumes[1]
        box_boundary = bdry1
    else:
        _, cylinder = volumes[1]
        cylinder_boundary = bdry1

        _, box = volumes[0]
        box_boundary = bdry0
    model.addPhysicalGroup(3, [box], 1)
    model.addPhysicalGroup(3, [cylinder], 2)    

    for (dim, tag) in cylinder_boundary:
        c = fac.getCenterOfMass(dim, abs(tag))
        if near(c[2], A[2]):
            model.addPhysicalGroup(2, [tag], 1)
        elif near(c[2], B[2]):
            model.addPhysicalGroup(2, [tag], 2)
        else:
            model.addPhysicalGroup(2, [tag], 3)

    external_boundaries = ((np.array([0.0, 0.5, 0.5]), 10),
                           (np.array([1.0, 0.5, 0.5]), 11),
                           (np.array([0.5, 0.0, 0.5]), 12),
                           (np.array([0.5, 1.0, 0.5]), 13),
                           (np.array([0.5, 0.5, 0.0]), 14),
                           (np.array([0.5, 0.5, 1.0]), 15))
    for (dim, tag) in box_boundary:
        c = fac.getCenterOfMass(dim, abs(tag))
        for (center, phystag) in external_boundaries:
            if np.linalg.norm(c-center) < 1E-10:
                model.addPhysicalGroup(2, [tag], phystag)
            
    fac.synchronize()

    #gmsh.fltk.initialize()
    #gmsh.fltk.run()
    
    if resolution is None:
        resolution = 0.5*r
    gmsh.option.setNumber('Mesh.MeshSizeMax', resolution)

    nodes, topologies = msh_gmsh_model(model, 3)
    mesh, entity_fs = mesh_from_gmsh(nodes, topologies)
    gmsh.finalize()
    
    return entity_fs


def cylinder_mesh(A, B, r, resolution=None):
    gmsh.initialize()

    model = gmsh.model
    fac = model.occ

    tau = (B - A)/np.linalg.norm(B-A)
    Projection = np.eye(3) - np.outer(tau, tau)
    vals, vecs = np.linalg.eigh(Projection)
    n1, n2 = vecs[:, 1], vecs[:, 2]

    O = fac.addPoint(*A)
    (N, S, E, W) = (A+r*n1, A-r*n1, A+r*n2, A-r*n2)
    # Base
    N = fac.addPoint(*N)
    S = fac.addPoint(*S)
    E = fac.addPoint(*E)
    W = fac.addPoint(*W)

    base_lines = [fac.addCircleArc(E, O, N),
                  fac.addCircleArc(N, O, W),
                  fac.addCircleArc(W, O, S),
                  fac.addCircleArc(S, O, E)]

    fac.extrude([(1, tag) for tag in base_lines], *(B-A))
    fac.synchronize()

    wall = [tag for (dim, tag) in model.getEntities(2)]

    base, top = [], []
    for (dim, tag) in model.getEntities(1):
        x = np.array(fac.getCenterOfMass(dim, tag))

        if abs(np.dot(x - A, tau)) < 1E-10:
            base.append(tag)
        elif abs(np.dot(x - B, tau)) < 1E-10:
            top.append(tag)
        else:
            pass

    base = [fac.addPlaneSurface([fac.addCurveLoop(base)])]
    top = [fac.addPlaneSurface([fac.addCurveLoop(top)])]
    fac.synchronize()
    
    model.addPhysicalGroup(2, base, 1)
    model.addPhysicalGroup(2, top, 2)
    model.addPhysicalGroup(2, wall, 3)    
    
    fac.synchronize()            

    if resolution is None:
        resolution = 0.5*r
    gmsh.option.setNumber('Mesh.MeshSizeMax', resolution)

    nodes, topologies = msh_gmsh_model(model, 2)
    mesh, entity_fs = mesh_from_gmsh(nodes, topologies)
    gmsh.finalize()
    
    cell_f = entity_fs[2]
    # Now we want to compute normal vector field
    V = VectorFunctionSpace(mesh, 'DG', 0)
    dm = V.dofmap()

    n = Function(V)
    values = n.vector().get_local()
    for cell in SubsetIterator(cell_f, 1):
        dofs = dm.cell_dofs(cell.index())
        values[dofs] = -tau

    for cell in SubsetIterator(cell_f, 2):
        dofs = dm.cell_dofs(cell.index())
        values[dofs] = tau

    for cell in SubsetIterator(cell_f, 3):
        dofs = dm.cell_dofs(cell.index())

        n_ = Projection@(cell.midpoint().array() - A)
        values[dofs] = n_/np.linalg.norm(n_)

    n.vector().set_local(values)

    return (cell_f, n)


def extract_P0foo_from(parent_f, submesh):
    '''Restriction'''
    V = parent_f.function_space()
    parent_mesh = V.mesh()

    dim = parent_mesh.topology().dim()
    assert dim == submesh.topology().dim()
    
    c2pc = submesh.parent_entity_map[parent_mesh.id()][dim]

    R = FunctionSpace(submesh, V.ufl_element())
    f = Function(R)
    values = f.vector().get_local()
    
    Vdm, Rdm = V.dofmap(), R.dofmap()
    parent_values = parent_f.vector().get_local()

    for (c, pc) in c2pc.items():
        values[Rdm.cell_dofs(c)] = parent_values[Vdm.cell_dofs(pc)]
    f.vector().set_local(values)

    return f


def test(diff, val):
    assert diff < 1E-8, diff


def mark_cube_boundaries(facet_f):
    '''10 and above'''
    mesh = facet_f.mesh()
    assert mesh.geometry().dim() == 3
    # assert mesh.topology().dim() == facet_f.dim() == 2

    CompiledSubDomain('near(x[0], 0)').mark(facet_f, 10)
    CompiledSubDomain('near(x[0], 1)').mark(facet_f, 11)
    CompiledSubDomain('near(x[1], 0)').mark(facet_f, 12)
    CompiledSubDomain('near(x[1], 1)').mark(facet_f, 13)
    CompiledSubDomain('near(x[2], 0)').mark(facet_f, 14)
    CompiledSubDomain('near(x[2], 1)').mark(facet_f, 15)

    return facet_f


def get_full_system_solution(cell_f, facet_f, Ks, fs, pressure_bcs):
    '''Reference'''
    mesh = cell_f.mesh()
    # Remeber 1 is the box\cylinder, 2 is cylinder
    dx = Measure('dx', domain=mesh, subdomain_data=cell_f)
    ds = Measure('ds', domain=mesh, subdomain_data=facet_f)

    cell = mesh.ufl_cell()
    Velm = FiniteElement('RT', cell, 1)
    Qelm = FiniteElement('DG', cell, 0)
    Welm = MixedElement([Velm, Qelm])
    W = FunctionSpace(mesh, Welm)

    u, p = TrialFunctions(W)
    v, q = TestFunctions(W)

    K1, K2 = Ks
    a = ((1/K1)*inner(u, v)*dx(1) + (1/K2)*inner(u, v)*dx(2) - inner(p, div(v))*dx
         - inner(q, div(u))*dx)

    f1, f2 = fs
    L = -inner(f1, q)*dx(1) - inner(f2, q)*dx(2)

    n = FacetNormal(mesh)
    for (tag, value) in pressure_bcs.items():
        print(tag, value(0))
        L += -inner(value, dot(v, n))*ds(tag)

    print('Dim full system', W.dim())
    wh = Function(W)
    solve(a == L, wh)

    uh, ph = wh.split(deepcopy=True)

    return uh, ph
    
# Setting up pressure bcs for the full simulation


# --------------------------------------------------------------------

if __name__ == '__main__':
    from xii.meshing.generation import StraightLineMesh
    n = 2**4


    test_wall = False
    test_port = False
    test_mean = False
    test_lambda = False

    # -----------
    
    Omega = UnitCubeMesh(n-1, n-1, 2*n)
    Omega_facet_f = MeshFunction('size_t', Omega, Omega.topology().dim()-1, 0)
    Omega_facet_f = mark_cube_boundaries(Omega_facet_f)
    dsOmega = Measure('ds', domain=Omega, subdomain_data=Omega_facet_f)
    
    radius = 0.05           # Averaging radius for cyl. surface
    quadrature_degree = 10  # Quadraure degree for that integration

    # Want something fully inside
    A, B = np.array([[0.5, 0.5, 0.1], [0.5, 0.5, 0.9]])

    # Materials
    K1 = Constant(1E0)   # Outside of the cylinder
    K2 = Constant(1E4)   # Inside

    f1 = Constant(0)     # Outer forcing
    f2 = Constant(1)

    # On the outer boundary
    pressure_bcs = {tag: Constant(0) for tag in (15, 11, 12, 13, 14)}
    pressure_bcs[10] = Constant(0)

    # ---- Reference solution
    full_entity_fs = unit_cube_with_mesh(A=A, B=B, r=radius, resolution=radius)

    full_cell_f, full_facet_f = full_entity_fs[3], full_entity_fs[2]

    uh_full, ph_full = get_full_system_solution(full_cell_f, full_facet_f, Ks=(K1, K2), fs=(f1, f2),
                                                pressure_bcs=pressure_bcs)
    

    with XDMFFile(f'uh_full.xdmf') as out:
        uh_full.rename('uh_full', '')
        out.write(uh_full)

    with XDMFFile(f'ph_full.xdmf') as out:
        ph_full.rename('ph_full', '')        
        out.write(ph_full)

    # ----- Reduced model
    
    Lambda = StraightLineMesh(A, B, ncells=2*n)
    dL = Measure('dx', domain=Lambda)

    # Averaging surface
    cylinder = Circle(radius=radius, degree=quadrature_degree)
    
    # TODO: these meshes should be computed just based on the `cylinder`
    cylinder_f, cylinder_normal = cylinder_mesh(A, B, r=radius, resolution=radius)

    File('full_cylinder.pvd') << full_entity_fs[3]
    File('cylinder.pvd') << cylinder_f
    File('Omega.pvd') << Omega
    File('Lambda.pvd') << Lambda


    
    base, top, wall = (EmbeddedMesh(cylinder_f, tag)  for tag in (1, 2, 3))
    # Get normal fields of the piecese
    n_base, n_top, n_wall = (extract_P0foo_from(cylinder_normal, subd) for subd in (base, top, wall))

    dWall = Measure('dx', domain=wall)
    
    dBase = Measure('dx', domain=base)
    base_area = assemble(Constant(1)*dBase)

    dTop = Measure('dx', domain=top)
    top_area = assemble(Constant(1)*dTop)
    
    # For mass conservation on the in/out flow we need
    line_boundaries = MeshFunction('size_t', Lambda, Lambda.topology().dim()-1, 0)
    CompiledSubDomain('near(x[0], A) && near(x[1], B) && near(x[2], C)', A=A[0], B=A[1], C=A[2]).mark(line_boundaries, 1)
    CompiledSubDomain('near(x[0], A) && near(x[1], B) && near(x[2], C)', A=B[0], B=B[1], C=B[2]).mark(line_boundaries, 2)    
    dsL = Measure('ds', domain=Lambda, subdomain_data=line_boundaries)
    
    tau = TangentCurve(Lambda)
    Div = lambda arg, t=tau: dot(grad(arg), tau)

    V = FunctionSpace(Omega, 'BDM', 1)
    VL = FunctionSpace(Lambda, 'CG', 1)

    Q = FunctionSpace(Omega, 'DG', 0)
    QL = FunctionSpace(Lambda, 'DG', 0)

    # Multipliers for the outflow
    Qb = FunctionSpace(base, 'DG', 0)
    Qt = FunctionSpace(top, 'DG', 0)
    
    W = (V, VL, Q, QL, Qb, Qt)

    u, uL, p, pL, pb, pt = map(TrialFunction, W)
    v, vL, q, qL, qb, qt = map(TestFunction, W)

    Tu_w, Tv_w = (Trace(arg, wall) for arg in (u, v))
    Tu_b, Tv_b = (Trace(arg, base) for arg in (u, v))
    Tu_t, Tv_t = (Trace(arg, top) for arg in (u, v))    

    Pi_u, Pi_v = (FluxAverage(arg, Lambda, cylinder, normalize=False) for arg in (u, v))

    Mpb, Mqb = (Mean(arg, weight=Constant(1/base_area), measure=dBase) for arg in (pb, qb))
    Mpt, Mqt = (Mean(arg, weight=Constant(1/top_area), measure=dTop) for arg in (pt, qt))    
    
    if test_wall:
        File('results/wall.pvd') << wall
        File('results/n_wall.pvd') << n_wall
        
        op = inner(dot(Tu_w, n_wall), dot(Tv_w, n_wall))*dWall
        Op = ii_assemble(op)

        exprs = (Expression(('x[0]', 'x[1]', '0'), degree=1),
                 Expression(('x[1]', '-x[0]', '0'), degree=1),
                 Expression(('0', '0', 'x[2]'), degree=1),
                 Expression(('2*x[0]', '3*x[1]', '3*x[2]'), degree=1))

        for u0_expr in exprs:
            this = assemble(inner(dot(u0_expr, n_wall), dot(u0_expr, n_wall))*dWall)

            u0 = interpolate(u0_expr, V).vector()
            that = u0.inner(Op*u0)

            test(abs(this-that), this)

    if test_port:
        File('results/base.pvd') << base
        File('results/n_base.pvd') << n_base
    

        op = inner(pb, dot(Tv_b, n_base))*dBase
        Op = ii_assemble(op)

        v_exprs = (Expression(('2*x[0]', '3*x[1]', '3*x[2]'), degree=1),
                   Expression(('x[0]', 'x[1]', '3*x[2]'), degree=1))
        
        p_exprs = (Expression('x[0]+2*x[1]+3*x[2]', degree=1),
                   Expression('5*x[0]+x[1]+3*x[2]', degree=1))

        for (v_expr, p_expr) in zip(v_exprs, p_exprs):
            this = assemble(inner(p_expr, dot(v_expr, n_base))*dBase)

            v_foo = interpolate(v_expr, V).vector()
            p_foo = interpolate(p_expr, Qb).vector()
            that = v_foo.inner(Op*p_foo)

            test(abs(this-that), this)

    if test_mean:

        op = inner(Mqb, uL)*dsL(1)
        Op = ii_assemble(op)
        
        u_exprs = (Expression('2*x[0]+3*x[1]+3*x[2]', degree=1),
                   Expression('x[0]+x[1]+3*x[2]', degree=1))
        
        p_exprs = (Expression('x[0]+2*x[1]+3*x[2]', degree=1),
                   Expression('5*x[0]+x[1]+3*x[2]', degree=1))

        for (v_expr, p_expr) in zip(u_exprs, p_exprs):
            mean = assemble(p_expr*dBase)/base_area
            this = assemble(inner(Constant(mean), v_expr)*dsL(1))

            u_foo = interpolate(v_expr, VL).vector()
            q_foo = interpolate(p_expr, Qb).vector()
            that = q_foo.inner(Op*u_foo)

            test(abs(this-that), this)

    if test_lambda:

        op = ii_assemble(inner(pL, Pi_v)*dL)

        # pL_expr = Expression('x[0] + 2*x[1] + 3*x[2]', degree=1)
        pL_expr = Constant(10)
        pL_foo = interpolate(pL_expr, QL)

        Pi_v_exprs = (Constant(2*pi*radius**2), Constant(0), Constant(5*pi*radius**2))
        v_exprs = (Expression(('x[0]-0.5', 'x[1]-0.5', '0'), degree=1),
                   Expression(('(x[1]-0.5)', '-(x[0]-0.5)', '0'), degree=1),
                   Expression(('2*x[0]', '3*x[1]', '0'), degree=1))
        for (v_expr, Pi_v_expr) in zip(v_exprs, Pi_v_exprs):
            this = assemble(inner(pL_expr, Pi_v_expr)*dL)

            that = interpolate(v_expr, V).vector().inner(op*pL_foo.vector())
            test(abs(this-that), that)
        

    K2_hat, f2_hat = K2*pi*radius**2, f2*pi*radius**2
    K_n = Constant(K1/radius)
    # Parts without the coupling
    a, L = block_form(W, 2), block_form(W, 1)
    a[0][0] = (1/K1)*inner(u, v)*dx + (1/K_n)*inner(dot(Tu_w, n_wall), dot(Tv_w, n_wall))*dWall
    a[0][2] = -inner(p, div(v))*dx
    a[0][4] = inner(pb, dot(Tv_b, n_base))*dBase
    a[0][5] = inner(pt, dot(Tv_t, n_top))*dTop    
    
    a[1][1] = (1/K2_hat)*inner(uL, vL)*dL
    a[1][3] = -inner(pL, Div(vL))*dL
    a[1][4] = inner(Mpb, vL)*dsL(1)
    a[1][5] = -inner(Mpt, vL)*dsL(2)    

    a[2][0] = -inner(q, div(u))*dx
    a[3][1] = -inner(qL, Div(uL))*dL

    a[4][0] = inner(qb, dot(Tu_b, n_base))*dBase
    a[4][1] = inner(Mqb, uL)*dsL(1)

    a[5][0] = inner(qt, dot(Tu_t, n_top))*dTop
    a[5][1] = inner(Mqt, uL)*dsL(2)
    
    # ---
    
    n = FacetNormal(Omega)
    for (tag, value) in pressure_bcs.items():
        print(tag, '->', assemble(Constant(1)*dsOmega(tag)), value(0))
        L[0] += -inner(value, dot(v, n))*dsOmega(tag)
    
    L[2] = -inner(f1, q)*dx
    L[3] = -inner(f2_hat, qL)*dL
    
    A, b = map(ii_assemble, (a, L))
    
    # Add coupling with the line
    A[0][3] = ii_assemble(inner(pL, Pi_v)*dL)
    A[3][0] = ii_assemble(inner(qL, Pi_u)*dL)
    
    wh = ii_Function(W)
    A_, b_ = map(monolithic, (A, b))
    print('Linear system of size', A_.size(0))

    solver = PETScLUSolver('mumps')
    solver.solve(A_, wh.vector(), b_)

    uh, ubh, ph, pbh = wh[:4]
    wh.rename([('uh', ''), ('ubh', ''), ('ph', ''), ('pbh', '')])

    for whi in wh[:4]:
        print(whi.vector().norm('l2'))
        with XDMFFile(f'{whi.name()}.xdmf') as out:
            out.write(whi)
