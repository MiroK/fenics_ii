from dolfin import *
import numpy as np
from numpy.polynomial.legendre import leggauss
from petsc4py import PETSc
from ufl import UFLException


# FIXME: this should be done in C
def pipe_surface_average(u, Q, R=0.2, deg=4, collect_xq=False):
    '''
    Let Q be the function space over Gamma which is a manifold of codimension 2
    in 3d. Futher let S be a surface of a pipe with radius R and Gamma the
    centerline. Given R^3->R^1 function u here we compute its average value over
    S as a function represented in Q.
    '''
    info('\tComputing surface average')
    timer = Timer('trace matrix')
    assert u.value_rank() == 0
    try:
        assert u.geometric_dimension() == 3
    except UFLException as e:
        pass
    
    mesh = Q.mesh()
    assert mesh.topology().dim() == 1
    assert mesh.geometry().dim() == 3
    assert Q.ufl_element().family() in ('Lagrange', 'Discontinuous Lagrange')

    q = Function(Q)
    # NOTE: a cell of mesh (an edge) degines a normal vector. Let P be the plane
    # that is defined by the normal vector n and some point x on Gamma. Let L
    # be the circle that is the intersect of P and S. The value of q (in Q) at x
    # is defined as
    #
    #                    q(x) = (1/|L|)*\int_{L}g(x)*dL
    #
    # which simplifies to g(x) = (1/(2*pi*R))*\int_{-pi}^{pi}u(L)*R*d(theta) and
    # or                       = (1/2) * \int_{-1}^{1} u (L(pi*s)) * ds
    # This can be integrated no problemo once we figure out L. To this end, let
    # t_1 and t_2 be two unit mutually orthogonal vectors that are orthogonal to
    # n. Then L(pi*s) = p + R*t_1*cos(pi*s) + R*t_2*sin(pi*s) can be seen to be
    # such that i) |x-p| = R and ii) x.n = 0 [i.e. this the suitable
    # parametrization]
    # Note that for L, the radius R is constant but it can vary along the
    # centerline
    if isinstance(R, (int, float)): R = lambda x, R=R: R

    mesh_x = mesh.coordinates().reshape((-1, 3))
    dofmap = Q.dofmap()
    dofs_x = Q.tabulate_dof_coordinates().reshape((-1, 3))

    xq, wq = leggauss(deg)
    xq *= pi

    surface_points = []
    for cell in cells(mesh):
        # Up to the shift x, L can be defined for entiry segment
        v0, v1 = mesh_x[cell.entities(0)]
        n = v0 - v1

        t1 = np.array([n[1]-n[2], n[2]-n[0], n[0]-n[1]])
    
        t2 = np.cross(n, t1)
        t1 /= np.linalg.norm(t1)
        t2 = t2/np.linalg.norm(t2)
   
        # FIXME: parallel?
        values = q.vector().get_local()
        # For each dof of the segment the circle is shifted
        cell_dofs = dofmap.cell_dofs(cell.index())

        # print t1, t2, xq
        for x, dof in zip(dofs_x[cell_dofs], cell_dofs):
            # Composite; map points
            rad = R(x)
            Lxq = [x + rad*t1*sin(s) + rad*t2*cos(s) for s in xq]
            surface_points.append(Lxq)
            # Integrate
            value = (wq*np.array(map(u, Lxq))).sum()/2
            values[dof] = value
        q.vector().set_local(values)
        q.vector().apply('insert')
    info('\tDone in %g' % timer.stop())

    if collect_xq:
        return q, surface_points
    else:
        return q


def pipe_surface_average_operator(V, Q, R=0.2, deg=4):
    '''
    pipe_surface_average functionality as matrix A, i.e. q_vec=T*u_vec = where
    represents the function in Q returned by pipe_surface_average(u, Q, R, deg)
    for u represented in V by u_vec.
    '''
    info('\tComputing averaging matrix')
    timer = Timer('trace matrix')
    # NOTE: for now only scalars and the end result must be in *Lagrange space.
    assert V.ufl_element().value_size() == Q.ufl_element().value_size() == 1
    assert Q.ufl_element().family() in ('Lagrange', 'Discontinuous Lagrange')
    assert V.element().geometric_dimension() == 3
    assert Q.element().geometric_dimension() == 3
    assert Q.element().topological_dimension() == 1

    # FIXME: parallel/vector valued functions?

    # We are building a dim(Q) x dim(V) matrix where for each row(dof of Q) the
    # entries in that row are such that row.u_vec is the numerical quadrature.
    # In this process evaluation of basis functions of V will be needed
    Vmesh = V.mesh()
    tree = Vmesh.bounding_box_tree()                     # cell find
    limit = Vmesh.topology().size_global(Vmesh.topology().dim())
 
    Vel = V.element()       
    Vdofm = V.dofmap()                                   # dofs of cell (cols)
    basis_values = np.zeros(Vel.space_dimension())       # V basis eval

    Qmesh = Q.mesh()
    Qdofm = Q.dofmap()                                       # rows
    Qdofs_x = Q.tabulate_dof_coordinates().reshape((-1, 3))  # determines L
    Qmesh_x = Qmesh.coordinates().reshape((-1, 3))           # for normal

    # Define the mat object ---------------------------------------------------
    comm = Vmesh.mpi_comm().tompi4py()
    mat = PETSc.Mat()
    mat.create(comm)
    mat.setSizes([[Qdofm.index_map().size(IndexMap.MapSize_OWNED),
                   Qdofm.index_map().size(IndexMap.MapSize_GLOBAL)],
                  [Vdofm.index_map().size(IndexMap.MapSize_OWNED),
                   Vdofm.index_map().size(IndexMap.MapSize_GLOBAL)]])
    mat.setType('aij')
    mat.setUp()
    # Local to global
    row_lgmap = PETSc.LGMap().create(map(int, Qdofm.tabulate_local_to_global_dofs()), comm=comm)
    col_lgmap = PETSc.LGMap().create(map(int, Vdofm.tabulate_local_to_global_dofs()), comm=comm)
    mat.setLGMap(row_lgmap, col_lgmap)
    # -------------------------------------------------------------------------
    if isinstance(R, (int, float)): R = lambda x, R=R: R

    xq, wq = leggauss(deg)
    xq *= pi    # pi*(-1, 1)
    wq /= 2.    # And the end we want 0.5*\int{}...
    mat.assemblyBegin()
    for cell in cells(Qmesh):
        # Up to the shift x, L can be defined for entire segment
        v0, v1 = Qmesh_x[cell.entities(0)]
        n = v0 - v1

        t1 = np.array([n[1]-n[2], n[2]-n[0], n[0]-n[1]])
    
        t2 = np.cross(n, t1)
        t1 /= np.linalg.norm(t1)
        t2 = t2/np.linalg.norm(t2)
        # For each dof of the segment the circle is shifted
        rows = Qdofm.cell_dofs(cell.index())
        # Now build rows
        for x, row in zip(Qdofs_x[rows], rows):
            # Composite; map points
            rad = R(x)
            Lxq = [x + rad*t1*sin(s) + rad*t2*cos(s) for s in xq]
            # Now at each point the basis functions of V must be evaluated
            # (those that are active determine the columns)
            cols = {}
            for k, xqk in enumerate(Lxq):
                c = tree.compute_first_entity_collision(Point(*xqk))
                if c >= limit: continue

                Vcell = Cell(Vmesh, c)
                vertex_coordinates = Vcell.get_vertex_coordinates()
                cell_orientation = Vcell.orientation()
                Vel.evaluate_basis_all(basis_values, xqk, vertex_coordinates, cell_orientation)

                cols_k = Vdofm.cell_dofs(c)
                # The corresponding value is is the basis value scaled by a
                # weight
                values_k = basis_values*wq[k]
                # Add
                for col, value in zip(cols_k, values_k):
                    if col in cols:
                        cols[col] += value
                    else:
                        cols[col] = value
            # Time to assign
            column_indices, column_values = [], []
            for c, v in cols.items(): 
                column_indices.append(c)
                column_values.append(v)
            # Convert
            column_indices = np.array(column_indices, dtype='int32')
            column_values = np.array(column_values, dtype='double')
            row_indices = [row]
            mat.setValues(row_indices, column_indices, column_values, PETSc.InsertMode.INSERT_VALUES)
    mat.assemblyEnd()
    info('\tDone in %g' % timer.stop())

    return PETScMatrix(mat)

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    from embedded_mesh import EmbeddedMesh

    for n in [10, 20, 40, 80, 160]:
        mesh = BoxMesh(Point(-1, -1, -1), Point(1, 1, 1), 4, 4, n)

        gamma = CompiledSubDomain('near(x[0], 0) && near(x[1], 0)')
        f = EdgeFunction('size_t', mesh, 0)
        gamma.mark(f, 1)

        gamma = EmbeddedMesh(mesh, f, 1).mesh
        Q = FunctionSpace(gamma, 'DG', 0)

        u = Expression('sqrt(x[0]*x[0]+x[1]*x[1])*fabs(x[0])*sin(pi*x[2])', degree=2)

        R = 0.2
        q = pipe_surface_average(u, Q, R=R, deg=12)
        q0 = Expression('(2/pi)*R*R*sin(pi*x[2])', R=R, degree=3)
        print '>>>', errornorm(q0, q, 'L2')

    # ------------------------------------------------------------------------

    u = Expression('sqrt(x[0]*x[0]+x[1]*x[1])*sin(pi*x[2])', degree=2)
    deg = 8
    V = FunctionSpace(mesh, 'CG', 1)
    u = interpolate(u, V)

    q0 = pipe_surface_average(u, Q, R=R, deg=deg)

    A = pipe_surface_average_operator(V, Q, R=R, deg=deg)
    q = Function(Q)
    A.mult(u.vector(), q.vector())
    print q.vector().norm('linf')
    q.vector().axpy(-1, q0.vector())
    print q.vector().norm('linf')

    # ------------------------------------------------------------------------

    # 0.5*(0.2)**4*0.25
    u = Expression('(x[0]*x[0])*(x[1]*x[1])', degree=4)
    deg = 15
    V = FunctionSpace(mesh, 'CG', 4)
    u = interpolate(u, V)

    q0 = pipe_surface_average(u, Q, R=R, deg=deg)

    A = pipe_surface_average_operator(V, Q, R=R, deg=deg)
    q = Function(Q)
    A.mult(u.vector(), q.vector())
    # As before check that the average and operator give same
    print q.vector().norm('linf')
    q.vector().axpy(-1, q0.vector())
    print 'avg vs operator', q.vector().norm('linf')

    exact = interpolate(Constant(0.5*(0.2)**4*0.25), Q)
    # The quality of approximation
    q0.vector().axpy(-1, exact.vector())
    print 'error', q0.vector().norm('linf')

    # A test for changing radius
    # R is given in terms of coordinates on Gamma
    Q = FunctionSpace(gamma, 'DG', 4)
    q, points = pipe_surface_average(u, Q, 
                                     R=lambda x: 0.2 + (1+x[2])*0.1,
                                     deg=15, collect_xq=True)

    exact = interpolate(Expression('pow(0.2 + (1+x[2])*0.1, 4)/8.', degree=4), Q)
    q.vector().axpy(-1, exact.vector())
    print 'error', q.vector().norm('linf')

    # JUST FOR FUN -----------------------------------------------------------
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Individual circles
    for circle_points in points:
        pts = np.array(circle_points)
        ax.plot(np.r_[pts[:, 0], pts[0, 0]],
                np.r_[pts[:, 1], pts[0, 1]],
                np.r_[pts[:, 2], pts[0, 2]], '-ko')
    plt.show()
