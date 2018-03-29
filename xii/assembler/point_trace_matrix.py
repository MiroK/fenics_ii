from xii.linalg.matrix_utils import petsc_serial_matrix

from dolfin import PETScMatrix, Point, Cell
from petsc4py import PETSc
import numpy as np


def point_trace_mat(V, TV, trace_mesh, data):
    '''
    Let u in V; u = ck phi_k then u(x0) \in TV = ck phi_k(x0). So this 
    is a 1 by dim(V) matrix where the column values are phi_k(x0).
    '''
    # The signature is for compatibility of API
    # Compatibility of spaces
    assert TV.ufl_element().family() == 'Real'
    assert V.mesh().id() == TV.mesh().id() == trace_mesh.id()

    x0 = data['point']
    assert len(x0) == V.mesh().geometry().dim()

    Tmat = point_trace_matrix(V, TV, x0)
    return PETScMatrix(Tmat)
                

def point_trace_matrix(V, TV, x0):
    '''
    Let u in V; u = ck phi_k then u(x0) \in TV = ck phi_k(x0). So this 
    is a 1 by dim(V) matrix where the column values are phi_k(x0).
    '''
    mesh = V.mesh()
    tree = mesh.bounding_box_tree()
    cell = tree.compute_first_entity_collision(Point(*x0))
    assert cell < mesh.num_cells()
    # Rows
    dofs = V.dofmap().cell_dofs(cell)
    Vel = V.element()
    value_size = V.ufl_element().value_size()
    assert value_size == 1
    basis_values = np.zeros(V.element().space_dimension()*value_size)

    Vcell = Cell(mesh, cell)
    vertex_coordinates = Vcell.get_vertex_coordinates()
    cell_orientation = Vcell.orientation()
    x0 = np.fromiter(x0, dtype=float)
    Vel.evaluate_basis_all(basis_values, x0, vertex_coordinates, cell_orientation)

    with petsc_serial_matrix(TV, V) as mat:
        for row in range(TV.dim()):
            mat.setValues(row, dofs, basis_values, PETSc.InsertMode.INSERT_VALUES)
    return mat

# --------------------------------------------------------------------

if __name__ == '__main__':
    from dolfin import *
    from xii import *
    
    mesh = UnitSquareMesh(32, 32)
    V = FunctionSpace(mesh, 'CG', 1)
    Q = FunctionSpace(mesh, 'R', 0)
    W = [V, Q]

    x0 = (0.33, 0.123)

    u, p = map(TrialFunction, W)
    v, q = map(TestFunction, W)
    # Point Constraints
    x0 = (0.33, 0.66)
    Du, Dv = PointTrace(u, x0), PointTrace(v, x0)

    a01 = inner(Dv, p)*dx
    a10 = inner(Du, q)*dx

    x = ii_assemble(a01)
    
    #f = Function(V)
    #f.vector().set_local(x)
    #f.vector().apply('insert')

    #print f(*x0)
