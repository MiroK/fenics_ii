from xii.linalg.matrix_utils import petsc_serial_matrix, is_number
from xii.assembler.averaging_form import average_cell

from numpy.polynomial.legendre import leggauss
from dolfin import PETScMatrix
from petsc4py import PETSc
import numpy as np


def avg_mat(V, TV, radius, quad_degree):
    '''
    A mapping for computing the surface averages of function in V in the 
    space TV. Surface averaging is defined as 

    (Pi u)(x) = |C_R(x)|int_{C_R(x)} u(y) dy with C_R(x) the circle of 
    radius R(x) centered at x with a normal parallel with the edge tangent.
    '''
    # It is most natural to represent Pi u in a DG space
    assert TV.ufl_element().family() == 'Discontinuous Lagrange'
    
    # Compatibility of spaces
    assert V.dolfin_element().value_rank() == TV.dolfin_element().value_rank()
    assert V.ufl_element().value_shape() == TV.ufl_element().value_shape()
    assert average_cell(V) == TV.mesh().ufl_cell()
    assert V.mesh().geometry().dim() == TV.mesh().geometry().dim()

    Rmat = averaging_matrix(V, TV, radius, quad_degree)
    return PETScMatrix(Rmat)
                

def averaging_matrix(V, TV, radius, quad_degree):
    '''The first cell connected to the facet gets to set the values of TV'''
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
    # So clearly we can scale the weights as well as precompute
    # cos and sin terms.
    xq, wq = leggauss(quad_degree)
    wq *= 0.5
    cos_xq = np.cos(pi*xq).reshape((-1, 1))
    sin_xq = np.sin(pi*xq).reshape((-1, 1))

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

            # On to next point
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

    radius = 0.01
    quad_degree = 10
    # A = 0.4

    # Simple scalar
    V = FunctionSpace(mesh, 'CG', 3)
    Q = FunctionSpace(bmesh, 'DG', 3)

    f = Expression('x[2]*((x[0]-0.5)*(x[0]-0.5) + (x[1]-0.5)*(x[1]-0.5))', degree=3)
    Pif = Expression('x[2]*A*A', A=radius, degree=1)
    
    f = interpolate(f, V)
    Pi_f0 = interpolate(Pif, Q)

    Pi_f = Function(Q)

    Pi = avg_mat(V, Q, radius, quad_degree)
    Pi.mult(f.vector(), Pi_f.vector())

    Pi_f0.vector().axpy(-1, Pi_f.vector())
    print '>>', Pi_f0.vector().norm('linf')

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

    Pi = avg_mat(V, Q, radius, quad_degree)
    Pi.mult(f.vector(), Pi_f.vector())

    Pi_f0.vector().axpy(-1, Pi_f.vector())
    print '>>', Pi_f0.vector().norm('linf')
