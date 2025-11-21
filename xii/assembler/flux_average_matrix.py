from xii.linalg.matrix_utils import petsc_serial_matrix, is_number
from xii.assembler.average_form import average_cell, average_space

from numpy.polynomial.legendre import leggauss
from dolfin import PETScMatrix, cells, Point, Cell, Function
import scipy.sparse as sp
from petsc4py import PETSc
import numpy as np
import tqdm


def memoize_average(average_mat):
    '''Cached average'''
    cache = {}
    def cached_average_mat(V, TV, reduced_mesh, data):
        key = ((V.ufl_element(), V.mesh().id()),
               (TV.ufl_element(), TV.mesh().id()),
               data['shape'],
               data['normalize'])

        if key not in cache:
            cache[key] = average_mat(V, TV, reduced_mesh, data)
        return cache[key]
    
    return cached_average_mat


@memoize_average
def flux_avg_mat(V, TV, reduced_mesh, data):
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
    assert len(V.ufl_element().value_shape()) == 1
    assert TV.ufl_element().value_shape() == (), (TV.ufl_element(), )
    assert average_cell(V) == TV.mesh().ufl_cell()
    assert V.mesh().geometry().dim() == TV.mesh().geometry().dim()

    shape = data['shape']
    assert shape is not None

    # Surface averages
    Rmat = flux_average_matrix(V, TV, shape, data['normalize'])
        
    return PETScMatrix(Rmat)
                

def flux_average_matrix(V, TV, shape, normalize):
    '''
    Averaging matrix for reduction of g in V to TV by integration over shape.
    '''
    # We build a matrix representation of u in V -> Pi(u) in TV where
    #
    # Pi(u)(s) = |L(s)|^-1*\int_{L(s)}u(t) dx(s)
    #
    # Here L is the shape over which u is integrated for reduction.
    # Its measure is |L(s)|.
    
    mesh_x = TV.mesh().coordinates()
    # The idea for point evaluation/computing dofs of TV is to minimize
    # the number of evaluation. I mean a vector dof if done naively would
    # have to evaluate at same x number of component times.
    value_size = V.ufl_element().value_size()

    mesh = V.mesh()
    # Eval at points will require serch
    tree = mesh.bounding_box_tree()
    limit = mesh.num_cells()

    TV_coordinates = TV.tabulate_dof_coordinates().reshape((TV.dim(), -1))
    line_mesh = TV.mesh()
    
    TV_dm = TV.dofmap()
    V_dm = V.dofmap()

    Vel = V.element()               
    basis_values = np.zeros((V.element().space_dimension(), value_size))


    II, JJ, VALS = [], [], []
    nnz = 0
    for line_cell in tqdm.tqdm(cells(line_mesh), desc=f'Averaging over {line_mesh.num_cells()} cells',
                               total=line_mesh.num_cells()):
        # Get the tangent (normal of the plane which cuts the virtual
        # surface to yield the bdry curve
        v0, v1 = mesh_x[line_cell.entities(0)]
        n = v0 - v1

        # The idea is now to minimize the point evaluation
        scalar_dofs = TV_dm.cell_dofs(line_cell.index())
        scalar_dofs_x = TV_coordinates[scalar_dofs]
        for scalar_row, avg_point in zip(scalar_dofs, scalar_dofs_x):
            # Avg point here has the role of 'height' coordinate
            quadrature, quadrature_normal = shape.quadrature_normal(avg_point, n)
            integration_points = quadrature.points
            wq = quadrature.weights

            if normalize:
                curve_measure = sum(wq)
            else:
                curve_measure = 1.0

            data = {}
            for index, (ip, normal_ip) in enumerate(zip(integration_points, quadrature_normal)):
                c = tree.compute_first_entity_collision(Point(*ip))
                if c >= limit:
                    c = None
                    continue

                if c is None:
                    cs = tree.compute_entity_collisions(Point(*ip))[:1]
                else:
                    cs = (c, )
                    
                c = cs[0]
                Vcell = Cell(mesh, c)
                vertex_coordinates = Vcell.get_vertex_coordinates()
                cell_orientation = Vcell.orientation()
                basis_values[:] = Vel.evaluate_basis_all(ip, vertex_coordinates, cell_orientation).reshape(basis_values.shape)


                cols_ip = V_dm.cell_dofs(c)
                # This v.n@xq * wq
                values_ip = np.sum(basis_values*normal_ip, axis=1)*wq[index]
                # Add
                for col, value in zip(cols_ip, values_ip):
                    if col in data:
                        data[col] += value/curve_measure
                    else:
                        data[col] = value/curve_measure

            # All points outside it seems
            if not data: continue
            
            column_indices, column_values = zip(*data.items())
            rows = [scalar_row]*len(column_indices)
            nnz = max(nnz, len(column_indices))

            II.extend(rows)
            JJ.extend(column_indices)
            VALS.extend(column_values)
        # On to next avg point
    # On to next cell
    csr = sp.csr_matrix((np.array(VALS), (np.array(II, dtype='int32'), np.array(JJ, dtype='int32'))),
                        shape=(TV.dim(), V.dim()))

    mat = PETSc.Mat().createAIJ(comm=PETSc.COMM_WORLD,
                                size=csr.shape,
                                csr=(csr.indptr, csr.indices, csr.data))
    
    return mat

# --------------------------------------------------------------------

if __name__ == '__main__':
    from dolfin import *
    from xii import EmbeddedMesh
    from xii.assembler.average_shape import Circle

    
    def is_close(a, b=0): return abs(a - b) < 1E-10
    
    # ---
    
    mesh = UnitCubeMesh(10, 10, 10)

    f = MeshFunction('size_t', mesh, 1, 0)
    CompiledSubDomain('near(x[0], 0.5) && near(x[1], 0.5)').mark(f, 1)

    bmesh = EmbeddedMesh(f, 1)

    # Trace
    V = FunctionSpace(mesh, 'BDM', 1)
    Q = FunctionSpace(bmesh, 'DG', 1)
    
    radius = 0.01
    quad_degree = 10
    # PI
    shape = Circle(radius=radius, degree=quad_degree)

    # Case; co-normal
    f = interpolate(Expression(('x[0]-0.5', 'x[1]-0.5', '0'), degree=2), V)    

    Pi_f = Function(Q)
    
    Pi = flux_avg_mat(V, Q, bmesh, {'shape': shape, 'normalize': True})
    Pi.mult(f.vector(), Pi_f.vector())

    # True
    Pi_f0 = interpolate(Constant(radius), Q)
    Pi_f0.vector().axpy(-1, Pi_f.vector())
    assert is_close(Pi_f0.vector().norm('linf'))

    # Case; ortogonal
    f = interpolate(Expression(('x[1]-0.5', 'x[0]-0.5', '0'), degree=2), V)    

    Pi_f = Function(Q)
    
    Pi = flux_avg_mat(V, Q, bmesh, {'shape': shape, 'normalize': True})
    Pi.mult(f.vector(), Pi_f.vector())

    # True
    Pi_f0 = interpolate(Constant(0), Q)
    Pi_f0.vector().axpy(-1, Pi_f.vector())
    assert is_close(Pi_f0.vector().norm('linf'))

    # Case; ortogonal 2 
    f = interpolate(Expression(('0', '0', '1'), degree=2), V)    

    Pi_f = Function(Q)
    
    Pi = flux_avg_mat(V, Q, bmesh, {'shape': shape, 'normalize': True})
    Pi.mult(f.vector(), Pi_f.vector())

    # True
    Pi_f0 = interpolate(Constant(0), Q)
    Pi_f0.vector().axpy(-1, Pi_f.vector())
    assert is_close(Pi_f0.vector().norm('linf'))
    
    # Case; using v.n*ds = div(v)*dv = [2+3]*pi*r**2
    # /2*pi*r
    f = interpolate(Expression(('2*x[0]+x[1]', '3*x[1]+x[0]', '0'), degree=2), V)    

    Pi_f = Function(Q)
    
    Pi = flux_avg_mat(V, Q, bmesh, {'shape': shape, 'normalize': True})
    Pi.mult(f.vector(), Pi_f.vector())

    # True
    Pi_f0 = interpolate(Constant(5/2*radius), Q)
    Pi_f0.vector().axpy(-1, Pi_f.vector())
    assert is_close(Pi_f0.vector().norm('linf'))
    
    
