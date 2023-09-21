from scipy.sparse import csr_matrix
from dolfin import PETScMatrix, Point, Cell
import dolfin as df
from petsc4py import PETSc
import numpy as np


def memoize_point_trace(foo):
    '''Cached trace'''
    cache = {}
    def cached_trace_mat(V, TV, trace_mesh, data):
        key = ((V.ufl_element(), V.mesh().id()),
               (TV.ufl_element(), TV.mesh().id()))

        try:
            key = key + tuple(data['point'])
        except:
            key = key + (data['point'], )
               
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
    if V.ufl_element().family() == 'Lagrange':
        assert len(x0) == V.mesh().geometry().dim()        
        Tmat = point_trace_matrix_CG(V, TV, x0)
    else:
        assert V.ufl_element().family() == 'Discontinuous Lagrange'
        assert V.mesh().topology().dim() == 1
        # FIXME: for now assume scalar
        assert V.ufl_element().value_shape() == ()
        # FIXME: also assume that the bifurcation is given as an index 
        assert isinstance(x0, (np.int32, np.int64, int)), x0

        tangent = data['tangent']
        Tmat = point_trace_matrix_DG(V, TV, x0, tangent=tangent)
        
    return PETScMatrix(Tmat)

# ----

def point_trace_matrix_DG(V, TV, x0, tangent):
    '''
    Let u in V; u = ck phi_k then u(x0) \in TV = ck phi_k(x0). So this 
    is a 1 by dim(V) matrix where the column values are phi_k(x0).
    '''
    mesh = V.mesh()
    x = mesh.coordinates()
    
    tree = mesh.bounding_box_tree()
    cells = tree.compute_entity_collisions(Point(x[x0]))
    # Let's make sure we found all cells according to the bifurcation degree
    _, v2c = mesh.init(0, 1), mesh.topology()(0, 1)
    assert set(v2c(x0)) == set(cells)

    _, c2v = mesh.init(1, 0), mesh.topology()(1, 0)

    Vel = V.element()
    V_dofs_x = V.tabulate_dof_coordinates().reshape((-1, mesh.geometry().dim()))
    
    rows, cols, values = [], [], []
    for cell in cells:
        v0, v1 = x[c2v(cell)]
        mid = 0.5*(v0 + v1)
        # Get tangent of the cell
        cell_tangent = tangent(mid)
        
        # Cell for restriction
        Vcell = Cell(mesh, cell)
        vertex_coordinates = Vcell.get_vertex_coordinates()
        cell_orientation = Vcell.orientation()

        # Columns - get all components at once
        all_dofs = V.dofmap().cell_dofs(cell).tolist()

        value_size = V.ufl_element().value_size()
        # Take trace at point x0
        basis_values = Vel.evaluate_basis_all(x[x0], vertex_coordinates, cell_orientation)
        
        row, = TV.dofmap().cell_dofs(cell)
        
        rows.extend([row]*len(all_dofs))
        cols.extend(all_dofs)

        # Now we need sign of the dof
        sign_x = V_dofs_x[all_dofs[np.argmax(basis_values)]]
        cell_tangent_ = sign_x - mid
        sign = np.sign(np.dot(cell_tangent, cell_tangent_))
        
        values.extend(sign*basis_values)

    mat = csr_matrix((values, (rows, cols)), shape=(TV.dim(), V.dim()))        

    return PETSc.Mat().createAIJ(comm=PETSc.COMM_WORLD,
                                 size=mat.shape,
                                 csr=(mat.indptr, mat.indices, mat.data))


def point_trace_matrix_CG(V, TV, x0):
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
    
