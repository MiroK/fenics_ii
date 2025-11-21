# FIXME: scaling wrt to radius of the different inner products!
from dolfin import *
from xii import *
import numpy as np

from gmshnics import msh_gmsh_model, mesh_from_gmsh
import gmsh


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

# --------------------------------------------------------------------

if __name__ == '__main__':
    from xii.meshing.generation import StraightLineMesh
    n = 2**4

    test_wall = False
    test_port = False
    test_mean = False
    test_lambda = True

    # -----------
    
    Omega = UnitCubeMesh(n-1, n-1, 2*n)
    radius = 0.01           # Averaging radius for cyl. surface
    quadrature_degree = 10  # Quadraure degree for that integration

    K = Constant(1E0)
    Kb = Constant(1E0)

    # Want something fully inside
    A, B = np.array([[0.5, 0.5, 0.1], [0.5, 0.5, 0.9]])
    Lambda = StraightLineMesh(A, B, ncells=2*n)
    dL = Measure('dx', domain=Lambda)

    # Averaging surface
    cylinder = Circle(radius=radius, degree=quadrature_degree)
    
    # TODO: these meshes should be computed just based on the `cylinder`
    cylinder_f, cylinder_normal = cylinder_mesh(A, B, r=radius, resolution=radius)

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
    # TODO:
    # - check that dWall is convergent
    #
    #
    #
    
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

            print(abs(this-that), this)

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

            print(abs(this-that), this)

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

            print(abs(this-that), this)

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
            print(abs(this-that))
        
    exit()
        
    # Parts without the coupling
    a, L = block_form(W, 2), block_form(W, 1)
    a[0][0] = (1/K)*inner(u, v)*dx + inner(dot(Tu_w, n_wall), dot(Tv_w, n_wall))*dWall
    a[0][2] = -inner(p, div(v))*dx
    a[0][4] = inner(pb, dot(Tv_b, n_base))*dBase
    a[0][5] = inner(pt, dot(Tv_t, n_top))*dTop    
    
    a[1][1] = (1/Kb)*inner(uL, vL)*dL
    a[1][3] = -inner(pL, Div(vL))*dL
    a[1][4] = inner(Mpb, vL)*dsL(1)
    a[1][5] = inner(Mpt, vL)*dsL(2)    

    a[2][0] = -inner(q, div(u))*dx
    a[3][1] = -inner(qL, Div(uL))*dL

    a[4][0] = inner(qb, dot(Tu_b, n_base))*dBase
    a[4][1] = inner(Mqb, uL)*dsL(1)

    a[5][0] = inner(qt, dot(Tu_t, n_top))*dTop
    a[5][1] = inner(Mqt, uL)*dsL(2)
    # ---
    L[3] = inner(Constant(1), qL)*dL
    
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
