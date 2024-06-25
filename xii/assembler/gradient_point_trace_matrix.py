from scipy.sparse import csr_matrix
from dolfin import PETScMatrix, Point, Cell
import dolfin as df
from petsc4py import PETSc
import numpy as np
import ufl_legacy

from xii.assembler.point_trace_matrix import memoize_point_trace


@memoize_point_trace
def gradient_point_trace_mat(V, TV, trace_mesh, data):
    '''
    Let u in V; u = ck phi_k then u(x0) \in TV = ck phi_k(x0). So this 
    is a 1 by dim(V) matrix where the column values are phi_k(x0).
    '''
    # The signature is for compatibility of API
    # Compatibility of spaces
    assert TV.ufl_element().family() == 'Real'
    assert V.ufl_element().value_shape() == TV.ufl_element().value_shape()
    assert V.mesh().id() == TV.mesh().id() == trace_mesh.id()

    x0, cell, tangent = data['point'], data['cell'], data['tangent']
    if V.ufl_element().family() == 'Lagrange':
        raise NotImplementedError
    else:
        assert V.ufl_element().family() == 'Discontinuous Lagrange'
        assert V.mesh().topology().dim() == 1
        # FIXME: for now assume scalar
        assert V.ufl_element().value_shape() == ()
        # FIXME: also assume that the bifurcation is given as an index 
        assert isinstance(x0, (np.int32, np.int64, np.uint32, np.uint64, int)), x0
        assert isinstance(cell, (np.int32, np.int64, np.uint32, np.uint64, int)), cell        
        
        Tmat = gradient_point_trace_matrix_DG(V, TV, x0, cell=cell, tangent=tangent)
        
    return PETScMatrix(Tmat)

# ----

def gradient_point_trace_matrix_DG(V, TV, x0, cell, tangent):
    '''
    Let u in V; u = ck phi_k then u(x0) \in TV = ck phi_k(x0). So this 
    is a 1 by dim(V) matrix where the column values are phi_k(x0).
    '''
    mesh = V.mesh()
    x = mesh.coordinates()

    gdim = mesh.geometry().dim()
    
    tree = mesh.bounding_box_tree()
    cells = tree.compute_entity_collisions(Point(x[x0]))
    assert cell in cells

    cells = (cell, )

    Vel = V.element()
    V_dofs_x = V.tabulate_dof_coordinates().reshape((-1, mesh.geometry().dim()))

    
    rows, cols, values = [], [], []
    for cell in cells:

        if isinstance(tangent, df.Constant):
            tau = tangent.values()
        elif isinstance(tangent, df.Function):
            tau = tangent(Cell(mesh, cell).midpoint().array()[:gdim])
        else:
            tau = tangent
        
        # Cell for restriction
        Vcell = Cell(mesh, cell)
        vertex_coordinates = Vcell.get_vertex_coordinates()

        try:
            cell_orientation = Vcell.orientation()
        except:
            cell_orientation = 0

        # Columns - get all components at once
        all_dofs = V.dofmap().cell_dofs(cell).tolist()

        # Take trace at point x0
        basis_values = Vel.evaluate_basis_derivatives_all(1, x[x0], vertex_coordinates, cell_orientation)
        basis_values = basis_values.reshape((len(all_dofs), gdim))

        assert np.linalg.norm(basis_values) > 0
        
        row, = TV.dofmap().cell_dofs(cell)
        
        rows.extend([row]*len(all_dofs))
        cols.extend(all_dofs)

        # Now we need sign of the dof
        values.extend(np.dot(basis_values, tau).flatten())

    mat = csr_matrix((values, (rows, cols)), shape=(TV.dim(), V.dim()))        

    return PETSc.Mat().createAIJ(comm=PETSc.COMM_WORLD,
                                 size=mat.shape,
                                 csr=(mat.indptr, mat.indices, mat.data))

# --------------------------------------------------------------------

if __name__ == '__main__':
    from dolfin import *
    from xii import *
    
    mesh = UnitSquareMesh(32, 32)
    mesh = BoundaryMesh(mesh, 'exterior')
    V = FunctionSpace(mesh, 'DG', 1)
    Q = FunctionSpace(mesh, 'R', 0)
    W = [V, Q]

    u, p = list(map(TrialFunction, W))
    v, q = list(map(TestFunction, W))

    cell = 0
    point0 = Cell(mesh, 0).entities(0)[0]

    QQ = VectorFunctionSpace(mesh, 'R', 0, 2)
    tau = interpolate(Constant((1, 1)), QQ)
    
    Du, Dv = GradientPointTrace(u, point0, cell, tau), GradientPointTrace(v, point0, cell, tau)
    hK = CellVolume(mesh)
    
    a01 = (1/hK)*inner(Dv, p)*dx
    a10 = (1/hK)*inner(Du, q)*dx

    x = ii_convert(ii_assemble(a01))
