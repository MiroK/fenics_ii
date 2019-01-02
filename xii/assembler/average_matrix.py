from xii.linalg.matrix_utils import petsc_serial_matrix, is_number
from xii.assembler.average_form import average_cell

from numpy.polynomial.legendre import leggauss
from dolfin import PETScMatrix, cells, Point, Cell
from petsc4py import PETSc
import numpy as np


def memoize_average(average_mat):
    '''Cached average'''
    cache = {}
    def cached_average_mat(V, TV, reduced_mesh, data):
        key = ((V.ufl_element(), V.mesh().id()),
               (TV.ufl_element(), TV.mesh().id()),
               data['radius'], data['quad_degree'], data['surface'])

        if key not in cache:
            cache[key] = average_mat(V, TV, reduced_mesh, data)
        return cache[key]
    
    return cached_average_mat


@memoize_average
def avg_mat(V, TV, reduced_mesh, data):
    '''
    A mapping for computing the surface averages of function in V in the 
    space TV. Surface averaging is defined as 

    (Pi u)(x) = |C_R(x)|int_{C_R(x)} u(y) dy with C_R(x) the circle of 
    radius R(x) centered at x with a normal parallel with the edge tangent.
    '''
    assert TV.mesh().id() == reduced_mesh.id()
    
    # It is most natural to represent Pi u in a DG space
    assert TV.ufl_element().family() == 'Discontinuous Lagrange'
    
    # Compatibility of spaces
    assert V.dolfin_element().value_rank() == TV.dolfin_element().value_rank()
    assert V.ufl_element().value_shape() == TV.ufl_element().value_shape()
    assert average_cell(V) == TV.mesh().ufl_cell()
    assert V.mesh().geometry().dim() == TV.mesh().geometry().dim()

    radius = data['radius']
    # 3d-1d trace
    if radius is None:
        return PETScMatrix(trace_3d1d_matrix(V, TV, reduced_mesh))

    quad_degree = data['quad_degree']
    # Surface averages
    if data['surface'] == 'cylinder':
        Rmat = cylinder_average_matrix(V, TV, radius, quad_degree)
    # The sphere
    else:
        Rmat = sphere_average_matrix(V, TV, radius, quad_degree)
        
    return PETScMatrix(Rmat)
                

def cylinder_average_matrix(V, TV, radius, quad_degree):
    '''Averaging matrix'''
    mesh = V.mesh()
    line_mesh = TV.mesh()
    # We are going to perform the integration with Gauss quadrature at
    # the end (PI u)(x):
    # A cell of mesh (an edge) defines a normal vector. Let P be the plane
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
    
    # Clearly we can scale the weights as well as precompute
    # cos and sin terms.
    xq, wq = leggauss(quad_degree)
    wq *= 0.5
    cos_xq = np.cos(np.pi*xq).reshape((-1, 1))
    sin_xq = np.sin(np.pi*xq).reshape((-1, 1))

    if is_number(radius):
         radius = lambda x, radius=radius: radius 

    mesh_x = TV.mesh().coordinates()
    # The idea for point evaluation/computing dofs of TV is to minimize
    # the number of evaluation. I mean a vector dof if done naively would
    # have to evaluate at same x number of component times.
    value_size = TV.ufl_element().value_size()

    # Eval at points will require serch
    tree = mesh.bounding_box_tree()
    limit = mesh.num_cells()

    TV_coordinates = TV.tabulate_dof_coordinates().reshape((TV.dim(), -1))
    TV_dm = TV.dofmap()
    V_dm = V.dofmap()
    # For non scalar we plan to make compoenents by shift
    if value_size > 1:
        TV_dm = TV.sub(0).dofmap()

    Vel = V.element()               
    basis_values = np.zeros(V.element().space_dimension()*value_size)
    with petsc_serial_matrix(TV, V) as mat:

        for line_cell in cells(line_mesh):
            # Get the tangent => orthogonal tangent vectors

            v0, v1 = mesh_x[line_cell.entities(0)]
            n = v0 - v1

            t1 = np.array([n[1]-n[2], n[2]-n[0], n[0]-n[1]])
    
            t2 = np.cross(n, t1)
            t1 /= np.linalg.norm(t1)
            t2 = t2/np.linalg.norm(t2)

            # The idea is now to minimize the point evaluation
            scalar_dofs = TV_dm.cell_dofs(line_cell.index())
            scalar_dofs_x = TV_coordinates[scalar_dofs]
            for scalar_row, avg_point in zip(scalar_dofs, scalar_dofs_x):
                # Get radius and integration points
                rad = radius(avg_point)
         
                integration_points = avg_point + rad*t1*sin_xq + rad*t2*cos_xq

                data = {}
                for index, ip in enumerate(integration_points):
                    c = tree.compute_first_entity_collision(Point(*ip))
                    if c >= limit: continue

                    Vcell = Cell(mesh, c)
                    vertex_coordinates = Vcell.get_vertex_coordinates()
                    cell_orientation = Vcell.orientation()
                    Vel.evaluate_basis_all(basis_values, ip, vertex_coordinates, cell_orientation)

                    cols_ip = V_dm.cell_dofs(c)
                    values_ip = basis_values*wq[index]
                    # Add
                    for col, value in zip(cols_ip, values_ip.reshape((-1, value_size))):
                        if col in data:
                            data[col] += value
                        else:
                            data[col] = value
                            
                # The thing now that with data we can assign to several
                # rows of the matrix
                column_indices = np.array(data.keys(), dtype='int32')
                for shift in range(value_size):
                    row = scalar_row + shift
                    column_values = np.array([data[col][shift] for col in column_indices])
                    mat.setValues([row], column_indices, column_values, PETSc.InsertMode.INSERT_VALUES)
            # On to next avg point
        # On to next cell
    return PETScMatrix(mat)


def trace_3d1d_matrix(V, TV, reduced_mesh):
    '''Trace from 3d to 1d. Makes sense only for CG space'''
    assert reduced_mesh.id() == TV.mesh().id()
    assert V.ufl_element().family() == 'Lagrange'
    
    mesh = V.mesh()
    line_mesh = TV.mesh()
    
    # The idea for point evaluation/computing dofs of TV is to minimize
    # the number of evaluation. I mean a vector dof if done naively would
    # have to evaluate at same x number of component times.
    value_size = TV.ufl_element().value_size()

    # We use the map to get (1d cell -> [3d edge) -> 3d cell]
    if hasattr(reduced_mesh, 'parent_entity_map'):
        # ( )
        mapping = reduced_mesh.parent_entity_map[mesh.id()][1]
        # [ ]
        mesh.init(1)
        mesh.init(1, 3)
        e2c = mesh.topology()(1, 3)
        # From 1d cell (by index)
        get_cell3d = lambda c, d1d3=mapping, d3d3=e2c: d3d3(d1d3[c.index()])[0]
    # Tree collision by midpoint
    else:
        tree = mesh.bounding_box_tree()
        limit = mesh.num_cells()

        get_cell3d = lambda c, tree=tree, bound=limit: (
            lambda index: index if index<bound else None
        )(tree.compute_first_entity_collision(c.midpoint()))
  
    TV_coordinates = TV.tabulate_dof_coordinates().reshape((TV.dim(), -1))
    TV_dm = TV.dofmap()
    V_dm = V.dofmap()
    # For non scalar we plan to make compoenents by shift
    if value_size > 1:
        TV_dm = TV.sub(0).dofmap()

    Vel = V.element()               
    basis_values = np.zeros(V.element().space_dimension()*value_size)
    with petsc_serial_matrix(TV, V) as mat:

        for line_cell in cells(line_mesh):
            # Get the tangent => orthogonal tangent vectors
            # The idea is now to minimize the point evaluation
            scalar_dofs = TV_dm.cell_dofs(line_cell.index())
            scalar_dofs_x = TV_coordinates[scalar_dofs]

            # Let's get a 3d cell to use for getting the V values
            # CG assumption allows taking any
            tet_cell = get_cell3d(line_cell)
            if tet_cell is None: continue
            
            Vcell = Cell(mesh, tet_cell)
            vertex_coordinates = Vcell.get_vertex_coordinates()
            cell_orientation = 0
            # Columns are determined by V cell! I guess the sparsity
            # could be improved if for x_dofs of TV only x_dofs of V
            # were considered
            column_indices = np.array(V_dm.cell_dofs(tet_cell), dtype='int32')

            for scalar_row, avg_point in zip(scalar_dofs, scalar_dofs_x):
                # 3d at point
                Vel.evaluate_basis_all(basis_values, avg_point, vertex_coordinates, cell_orientation)
                # The thing now is that with data we can assign to several
                # rows of the matrix. Shift determines the (x, y, ... ) or
                # (xx, xy, yx, ...) component of Q
                data = basis_values.reshape((-1, value_size)).T
                for shift, column_values in enumerate(data):
                    row = scalar_row + shift
                    mat.setValues([row], column_indices, column_values, PETSc.InsertMode.INSERT_VALUES)
            # On to next avg point
        # On to next cell
    return PETScMatrix(mat)


def sphere_average_matrix(V, TV, radius, quad_degree):
    '''Averaging matrix over the sphere'''
    mesh = V.mesh()
    line_mesh = TV.mesh()
    # Lebedev below need off degrees
    if quad_degree % 2 == 0: quad_degree += 1
    # NOTE: this is a dependency
    from quadpy.sphere import Lebedev

    integrator = Lebedev(quad_degree)
    xq = integrator.points
    wq = integrator.weights
    
    if is_number(radius):
         radius = lambda x, radius=radius: radius 

    mesh_x = TV.mesh().coordinates()
    # The idea for point evaluation/computing dofs of TV is to minimize
    # the number of evaluation. I mean a vector dof if done naively would
    # have to evaluate at same x number of component times.
    value_size = TV.ufl_element().value_size()

    # Eval at points will require serch
    tree = mesh.bounding_box_tree()
    limit = mesh.num_cells()

    TV_coordinates = TV.tabulate_dof_coordinates().reshape((TV.dim(), -1))
    TV_dm = TV.dofmap()
    V_dm = V.dofmap()
    # For non scalar we plan to make compoenents by shift
    if value_size > 1:
        TV_dm = TV.sub(0).dofmap()

    Vel = V.element()               
    basis_values = np.zeros(V.element().space_dimension()*value_size)
    with petsc_serial_matrix(TV, V) as mat:

        for line_cell in cells(line_mesh):
            # The idea is now to minimize the point evaluation
            scalar_dofs = TV_dm.cell_dofs(line_cell.index())
            scalar_dofs_x = TV_coordinates[scalar_dofs]
            for scalar_row, avg_point in zip(scalar_dofs, scalar_dofs_x):
                # Get radius and integration points
                rad = radius(avg_point)
                # Scale and shift the unit sphere to the point
                integration_points = xq*rad + avg_point

                data = {}
                for index, ip in enumerate(integration_points):
                    c = tree.compute_first_entity_collision(Point(*ip))
                    if c >= limit: continue

                    Vcell = Cell(mesh, c)
                    vertex_coordinates = Vcell.get_vertex_coordinates()
                    cell_orientation = Vcell.orientation()
                    Vel.evaluate_basis_all(basis_values, ip, vertex_coordinates, cell_orientation)

                    cols_ip = V_dm.cell_dofs(c)
                    values_ip = basis_values*wq[index]
                    # Add
                    for col, value in zip(cols_ip, values_ip.reshape((-1, value_size))):
                        if col in data:
                            data[col] += value
                        else:
                            data[col] = value
                            
                # The thing now that with data we can assign to several
                # rows of the matrix
                column_indices = np.array(data.keys(), dtype='int32')
                for shift in range(value_size):
                    row = scalar_row + shift
                    column_values = np.array([data[col][shift] for col in column_indices])
                    mat.setValues([row], column_indices, column_values, PETSc.InsertMode.INSERT_VALUES)
            # On to next avg point
        # On to next cell
    return PETScMatrix(mat)


# --------------------------------------------------------------------


if __name__ == '__main__':
    from dolfin import *
    from xii import EmbeddedMesh
    
    mesh = UnitCubeMesh(10, 10, 10)

    f = EdgeFunction('size_t', mesh, 0)
    CompiledSubDomain('near(x[0], 0.5) && near(x[1], 0.5)').mark(f, 1)

    bmesh = EmbeddedMesh(f, 1)

    # Trace
    V = FunctionSpace(mesh, 'CG', 2)
    TV = FunctionSpace(bmesh, 'DG', 1)
    
    f = interpolate(Expression('x[0]+x[1]+x[2]', degree=1), V)
    Tf0 = interpolate(f, TV)

    Trace = avg_mat(V, TV, bmesh, {'radius': None, 'surface': 'cylinder'})
    Tf = Function(TV)
    Trace.mult(f.vector(), Tf.vector())
    Tf0.vector().axpy(-1, Tf.vector())
    print('??', Tf0.vector().norm('linf'))

    V = VectorFunctionSpace(mesh, 'CG', 2)
    TV = VectorFunctionSpace(bmesh, 'DG', 1)
    
    f = interpolate(Expression(('x[0]+x[1]+x[2]',
                                'x[0]-x[1]',
                                'x[1]+x[2]'), degree=1), V)
    Tf0 = interpolate(f, TV)

    Trace = avg_mat(V, TV, bmesh, {'radius': None, 'surface': 'cylinder'})
    Tf = Function(TV)
    Trace.mult(f.vector(), Tf.vector())
    Tf0.vector().axpy(-1, Tf.vector())
    print('??', Tf0.vector().norm('linf'))

    # PI
    radius = 0.01
    quad_degree = 10
    data = {'radius': radius, 'quad_degree': quad_degree, 'surface': 'cylinder'}
    # Simple scalar
    V = FunctionSpace(mesh, 'CG', 3)
    Q = FunctionSpace(bmesh, 'DG', 3)


    f = Expression('x[2]*((x[0]-0.5)*(x[0]-0.5) + (x[1]-0.5)*(x[1]-0.5))', degree=3)
    Pif = Expression('x[2]*A*A', A=radius, degree=1)
    
    f = interpolate(f, V)
    Pi_f0 = interpolate(Pif, Q)

    Pi_f = Function(Q)

    Pi = avg_mat(V, Q, bmesh, data)
    Pi.mult(f.vector(), Pi_f.vector())

    Pi_f0.vector().axpy(-1, Pi_f.vector())
    print('>>', Pi_f0.vector().norm('linf'))

    V = VectorFunctionSpace(mesh, 'CG', 3)
    Q = VectorFunctionSpace(bmesh, 'DG', 3)

    f = Expression(('x[2]*((x[0]-0.5)*(x[0]-0.5) + (x[1]-0.5)*(x[1]-0.5))',
                    '2*x[2]*((x[0]-0.5)*(x[0]-0.5) + (x[1]-0.5)*(x[1]-0.5))',
                    '-3*x[2]*((x[0]-0.5)*(x[0]-0.5) + (x[1]-0.5)*(x[1]-0.5))'),
                    degree=3)
    Pif = Expression(('x[2]*A*A',
                      '2*x[2]*A*A',
                      '-3*x[2]*A*A'), A=radius, degree=1)
    
    f = interpolate(f, V)
    Pi_f0 = interpolate(Pif, Q)

    Pi_f = Function(Q)

    Pi = avg_mat(V, Q, bmesh, data)
    Pi.mult(f.vector(), Pi_f.vector())

    Pi_f0.vector().axpy(-1, Pi_f.vector())

    print('>>', Pi_f0.vector().norm('linf'))

    # Ball
    radius = 0.02
    quad_degree = 5
    data = {'radius': radius, 'quad_degree': quad_degree, 'surface': 'sphere'}
    # Simple scalar
    V = FunctionSpace(mesh, 'CG', 4)
    Q = FunctionSpace(bmesh, 'DG', 4)

    # Sanity, NOTE - when the curve exists the domain we have an integration
    # error
    f = Constant(2)
    Pif = Constant(2)

    from quadpy.sphere import integrate
    from quadpy.sphere import Lebedev
    
    xxx = Lebedev(11)

    f = Expression('pow((x[0]-0.5)*(x[0]-0.5), 1) + pow((x[1]-0.5)*(x[1]-0.5), 1)', degree=2)
    Pif = Constant(
        integrate(lambda x: x[0]**2+x[1]**2, np.array([0, 0, 0]), radius, xxx)/(4*pi*radius**2)
    )

    f = Expression('pow((x[0]-0.5)*(x[0]-0.5), 1)', degree=2)
    Pif = Constant(
        integrate(lambda x: x[0]**2, np.array([0, 0, 0]), radius, xxx)/(4*pi*radius**2)
    )

    f = interpolate(f, V)
    Pi_f0 = interpolate(Pif, Q)

    Pi_f = Function(Q)

    Pi = avg_mat(V, Q, bmesh, data)
    Pi.mult(f.vector(), Pi_f.vector())

    Pi_f0.vector().axpy(-1, Pi_f.vector())
    print('>>', Pi_f0.vector().norm('linf'))
    print(Pi_f0.vector().get_local())
