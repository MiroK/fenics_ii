from xii.linalg.matrix_utils import petsc_serial_matrix, is_number
from xii.assembler.normal_average_form import average_cell, average_space
from xii.assembler.fem_eval import DegreeOfFreedom, FEBasisFunction
from numpy.polynomial.legendre import leggauss
from dolfin import (PETScMatrix, cells, Point, Cell, Function, TestFunction, Constant,
                    dot, FunctionSpace)

from petsc4py import PETSc
import numpy as np


def memoize_average(average_mat):
    '''Cached average'''
    cache = {}
    def cached_average_mat(V, TV, reduced_mesh, data):
        key = ((V.ufl_element(), V.mesh().id()),
               (TV.ufl_element(), TV.mesh().id()),
               data['shape'])

        if key not in cache:
            cache[key] = average_mat(V, TV, reduced_mesh, data)
        return cache[key]
    
    return cached_average_mat


@memoize_average
def normal_avg_mat(V, TV, reduced_mesh, data):
    '''
    A mapping for computing the surface averages of function in V in the 
    space TV. Surface averaging is defined as 

    (Pi u)(x) = |C_R(x)|int_{C_R(x)} u(y).n(y) dy with C_R(x) the circle of 
    radius R(x) centered at x with a normal parallel with the edge tangent.
    '''
    assert TV.mesh().id() == reduced_mesh.id()
    
    # It is most natural to represent Pi u in a DG space
    assert TV.ufl_element().family() == 'Discontinuous Lagrange'

    v = TestFunction(V)
    n = Constant((0, )*V.mesh().geometry().dim())
    # Compatibility of spaces
    assert dot(v, n).ufl_shape == TV.ufl_element().value_shape(), (dot(v, n).ufl_shape, TV.ufl_element().value_shape())
    assert average_cell(V) == TV.mesh().ufl_cell()
    assert V.mesh().geometry().dim() == TV.mesh().geometry().dim()

    shape = data['shape']
    # Surface averages
    Rmat = normal_average_matrix(V, TV, shape)
        
    return PETScMatrix(Rmat)
                

def normal_average_matrix(V, TV, shape):
    '''
    Averaging matrix for reduction of g in V to TV by integration over shape.
    '''
    # We build a matrix representation of u in V -> Pi(u) in TV where
    #
    # Pi(u)(s) = |L(s)|^-1*\int_{L(s)} dot(u(t), n) dx(s)
    #
    # Here L is the shape over which u is integrated for reduction.
    # Its measure is |L(s)|.
    
    mesh_x = TV.mesh().coordinates()
    value_size = V.ufl_element().value_size()

    mesh = V.mesh()
    # Eval at points will require serch
    tree = mesh.bounding_box_tree()
    limit = mesh.num_cells()

    TV_coordinates = TV.tabulate_dof_coordinates().reshape((TV.dim(), -1))
    line_mesh = TV.mesh()
    
    TV_dm = TV.dofmap()
    V_dm = V.dofmap()

    TV_dof = DegreeOfFreedom(TV)

    Vel = V.element()
    basis_values = np.zeros(V.element().space_dimension()*value_size)
    with petsc_serial_matrix(TV, V) as mat:

        for line_cell in cells(line_mesh):
            TV_dof.cell = line_cell.index()
            
            # Get the tangent (normal of the plane which cuts the virtual
            # surface to yield the bdry curve
            v0, v1 = mesh_x[line_cell.entities(0)]
            n = v0 - v1

            # The idea is now to minimize the point evaluation
            TV_dofs = TV_dm.cell_dofs(line_cell.index())
            TV_dofs_x = TV_coordinates[TV_dofs]
            for ldof, (row, avg_point) in enumerate(zip(TV_dofs, TV_dofs_x)):
                TV_dof.dof = ldof
                
                # Avg point here has the role of 'height' coordinate
                quadrature = shape.quadrature(avg_point, n)
                normal = shape.normal(avg_point, n)
                
                integration_points = quadrature.points
                wq = quadrature.weights

                # Precompute normals
                normals_ip = [normal(ip) for ip in integration_points]

                curve_measure = sum(wq)

                data = {}
                for index, (ip, normal_ip) in enumerate(zip(integration_points, normals_ip)):
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
                            data[col] += value.dot(normal_ip)/curve_measure
                        else:
                            data[col] = value.dot(normal_ip)/curve_measure
                            
                # The thing now that with data we can assign to several
                # rows of the matrix
                column_indices = np.array(data.keys(), dtype='int32')
                column_values = np.array([TV_dof.eval(Constant(data[col])) for col in column_indices])
                mat.setValues([row], column_indices, column_values, PETSc.InsertMode.INSERT_VALUES)
            # On to next avg point
        # On to next cell
    return PETScMatrix(mat)


def normal_average_matrix_2d(V, center, shape):
    '''
    Averaging matrix for reduction of g in V by integrating shape@center
    '''
    assert V.mesh().geometry().dim() == 2
    # NOTE: the Pi operator is the one which reduces function to manifold
    # over topological gap of 2. Since we are in 2d each basis function is
    # reduced to a (point) value. We are thusbuilding a vector  
    value_size = V.ufl_element().value_size()

    mesh = V.mesh()
    # Eval at points will require serch
    tree = mesh.bounding_box_tree()
    limit = mesh.num_cells()

    TV = FunctionSpace(V.mesh(), 'Real', 0)
    TV_dm = TV.dofmap()
    V_dm = V.dofmap()

    TV_dof = DegreeOfFreedom(TV)

    Vel = V.element()
    basis_values = np.zeros(V.element().space_dimension()*value_size)
    with petsc_serial_matrix(TV, V) as mat:
        # Assume plane normal
        n = np.array([0, 0, 1.])
        # The idea is now to minimize the point evaluation
        TV_dofs = TV_dm.cell_dofs(0)  # That one dof
        TV_dofs_x = [center.array()]  # <-- our auxiliary dof sits at the center 
        for ldof, (row, avg_point) in enumerate(zip(TV_dofs, TV_dofs_x)):
            TV_dof.dof = ldof
            # Avg point here has the role of 'height' coordinate
            quadrature = shape.quadrature(avg_point, n)
            normal = shape.normal(avg_point, n)

            integration_points = quadrature.points
            wq = quadrature.weights

            # Precompute normals
            normals_ip = [normal(ip)[:2] for ip in integration_points]
            curve_measure = sum(wq)

            data = {}
            for index, (ip, normal_ip) in enumerate(zip(integration_points, normals_ip)):
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
                        data[col] += value.dot(normal_ip)/curve_measure
                    else:
                        data[col] = value.dot(normal_ip)/curve_measure

            # The thing now that with data we can assign to several
            # rows of the matrix
            column_indices = np.array(data.keys(), dtype='int32')
            column_values = np.array([TV_dof.eval(Constant(data[col])) for col in column_indices])
            mat.setValues([row], column_indices, column_values, PETSc.InsertMode.INSERT_VALUES)
            # On to next avg point
        # On to next cell
    return PETScMatrix(mat)

# --------------------------------------------------------------------

if __name__ == '__main__':
    from dolfin import *
    from xii import EmbeddedMesh
    from xii.assembler.average_shape import Circle

    
    def is_close(a, b=0): return abs(a - b) < 1E-13


    mesh = UnitSquareMesh(32, 32)
    
    radius = 0.01
    quad_degree = 10
    # PI
    shape = Circle(radius=radius, degree=quad_degree)

    V = VectorFunctionSpace(mesh, 'CG', 2)
    TV = FunctionSpace(mesh, 'R', 0)  # The space that holds the result
    f = interpolate(Expression(('(x[0]-0.5)',
                                '(x[1]-0.5)'), degree=1), V)
    Tf0 = interpolate(Constant(radius), TV)

    
    Trace = normal_average_matrix_2d(V, center=Point(0.5, 0.5), shape=shape)

    Tf = Function(TV)
    Trace.mult(f.vector(), Tf.vector())
    Tf0.vector().axpy(-1, Tf.vector())
    assert is_close(Tf0.vector().norm('linf')), (Tf0.vector().norm('linf'), Tf.vector().norm('linf'))


    # 3d version ---
    
    mesh = UnitCubeMesh(10, 10, 10)

    f = MeshFunction('size_t', mesh, 1, 0)
    CompiledSubDomain('near(x[0], 0.5) && near(x[1], 0.5)').mark(f, 1)
    
    bmesh = EmbeddedMesh(f, 1)

    radius = 0.01
    quad_degree = 10

    # PI
    shape = Circle(radius=radius, degree=quad_degree)
        
    if True:
        V = VectorFunctionSpace(mesh, 'CG', 2)
        TV = FunctionSpace(bmesh, 'DG', 1)
    
        f = interpolate(Expression(('0',
                                    '0',
                                    'x[1]+x[2]'), degree=1), V)
        Tf0 = interpolate(Constant(0), TV)
        
        Trace = normal_avg_mat(V, TV, bmesh, {'shape': shape})
        Tf = Function(TV)
        Trace.mult(f.vector(), Tf.vector())
        Tf0.vector().axpy(-1, Tf.vector())
        assert is_close(Tf0.vector().norm('linf')), Tf0.vector().norm('linf')

    # One more that is in place but orthogonal to normal
    if True:
        V = VectorFunctionSpace(mesh, 'CG', 2)
        TV = FunctionSpace(bmesh, 'DG', 1)
    
        f = interpolate(Expression(('-(x[1]-0.5)',
                                    '(x[0]-0.5)',
                                    'x[1]+x[2]'), degree=1), V)
        Tf0 = interpolate(Constant(0), TV)
        
        Trace = normal_avg_mat(V, TV, bmesh, {'shape': shape})
        Tf = Function(TV)
        Trace.mult(f.vector(), Tf.vector())
        Tf0.vector().axpy(-1, Tf.vector())
        assert is_close(Tf0.vector().norm('linf')), Tf0.vector().norm('linf')
    
    # One when we check quadrature
    if True:
        V = VectorFunctionSpace(mesh, 'CG', 2)
        TV = FunctionSpace(bmesh, 'DG', 1)
    
        f = interpolate(Expression(('(x[0]-0.5)',
                                    '(x[1]-0.5)',
                                    'x[1]+x[2]'), degree=1), V)
        Tf0 = interpolate(Constant(radius), TV)
        
        Trace = normal_avg_mat(V, TV, bmesh, {'shape': shape})
        Tf = Function(TV)
        Trace.mult(f.vector(), Tf.vector())
        Tf0.vector().axpy(-1, Tf.vector())
        assert is_close(Tf0.vector().norm('linf')), (Tf0.vector().norm('linf'), Tf.vector().norm('linf'))

        
    if True:
        V = VectorFunctionSpace(mesh, 'CG', 2)
        TV = FunctionSpace(bmesh, 'DG', 1)
    
        f = interpolate(Expression(('(x[0]-0.5)*x[2]',
                                    '(x[1]-0.5)*x[2]',
                                    'x[1]+x[2]'), degree=1), V)
        Tf0 = interpolate(Expression('x[2]*A', A=radius, degree=1), TV)
        
        Trace = normal_avg_mat(V, TV, bmesh, {'shape': shape})
        Tf = Function(TV)
        Trace.mult(f.vector(), Tf.vector())
        Tf0.vector().axpy(-1, Tf.vector())
        assert is_close(Tf0.vector().norm('linf')), (Tf0.vector().norm('linf'), Tf.vector().norm('linf'))

    if True:
        V = FunctionSpace(mesh, 'RT', 3)
        TV = FunctionSpace(bmesh, 'DG', 1)
    
        f = interpolate(Expression(('(x[0]-0.5)*x[2]',
                                    '(x[1]-0.5)*x[2]',
                                    'x[1]+x[2]'), degree=1), V)
        Tf0 = interpolate(Expression('x[2]*A', A=radius, degree=1), TV)
        
        Trace = normal_avg_mat(V, TV, bmesh, {'shape': shape})
        Tf = Function(TV)
        Trace.mult(f.vector(), Tf.vector())
        Tf0.vector().axpy(-1, Tf.vector())
        assert is_close(Tf0.vector().norm('linf')), (Tf0.vector().norm('linf'), Tf.vector().norm('linf'))
