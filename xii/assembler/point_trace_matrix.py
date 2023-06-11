from scipy.sparse import csr_matrix
from dolfin import PETScMatrix, Point, Cell
from petsc4py import PETSc
import numpy as np


def memoize_point_trace(foo):
    '''Cached trace'''
    cache = {}
    def cached_trace_mat(V, TV, trace_mesh, data):
        key = ((V.ufl_element(), V.mesh().id()),
               (TV.ufl_element(), TV.mesh().id()), tuple(data['point']))
               
        if key not in cache:
            cache[key] = foo(V, TV, trace_mesh, data)
        return cache[key]

    return cached_trace_mat


@memoize_point_trace
def point_trace_mat(V, TV, trace_mesh, data):
    '''
    Let u in V; u = ck phi_k then u(x0) \in TV = ck phi_k(x0). So this 
    is a 1 by dim(V) matrix where the column values are phi_k(x0).
    '''
    # The signature is for compatibility of API
    # Compatibility of spaces
    assert TV.ufl_element().family() == 'Real'
    assert V.ufl_element().value_shape() == TV.ufl_element().value_shape()
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

    # Cell for restriction
    Vcell = Cell(mesh, cell)
    vertex_coordinates = Vcell.get_vertex_coordinates()
    cell_orientation = Vcell.orientation()
    x0 = np.fromiter(x0, dtype=float)

    # Columns - get all components at once
    all_dofs = V.dofmap().cell_dofs(cell).tolist()
    Vel = V.element()
    value_size = V.ufl_element().value_size()
    basis_values = np.zeros(V.element().space_dimension()*value_size)

    Vel.evaluate_basis_all(basis_values, x0, vertex_coordinates, cell_orientation)


    # Scalar gets all
    if value_size == 1:
        component_dofs = lambda component: V.dofmap().cell_dofs(cell)
    # Slices
    else:
        component_dofs = lambda component: V.sub(component).dofmap().cell_dofs(cell)
        
    rows, cols, values = [], [], []
    for row in map(int, TV.dofmap().cell_dofs(cell)):  # R^n components
        sub_dofs = component_dofs(row)
        sub_dofs_local = [all_dofs.index(dof) for dof in sub_dofs]

        rows.extend([row]*len(sub_dofs))
        cols.extend(sub_dofs)
        values.extend(basis_values[sub_dofs_local].flat)

    mat = csr_matrix((values, (rows, cols)), shape=(TV.dim(), V.dim()))        

    return PETSc.Mat().createAIJ(comm=PETSc.COMM_WORLD,
                                 size=mat.shape,
                                 csr=(mat.indptr, mat.indices, mat.data))

# --------------------------------------------------------------------

if __name__ == '__main__':
    from dolfin import *
    from xii import *
    
    mesh = UnitSquareMesh(32, 32)
    V = VectorFunctionSpace(mesh, 'CG', 1)
    Q = VectorFunctionSpace(mesh, 'R', 0)
    W = [V, Q]

    u, p = list(map(TrialFunction, W))
    v, q = list(map(TestFunction, W))
    # Point Constraints
    x0 = (0.33, 0.66)
    Du, Dv = PointTrace(u, x0), PointTrace(v, x0)

    a01 = inner(Dv, p)*dx
    a10 = inner(Du, q)*dx

    x = ii_convert(ii_assemble(a01))
    
