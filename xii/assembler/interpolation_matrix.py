from xii.linalg.matrix_utils import petsc_serial_matrix
from xii.assembler.fem_eval import DegreeOfFreedom, FEBasisFunction
from petsc4py import PETSc
import dolfin as df
import numpy as np


def memoize_interp(f):
    '''Caching interpolation'''
    cache = {}
    def cached_interpolation_mat(V, Q):
        key = ((V.ufl_element(), V.mesh().id(), V.mesh().num_cells()),
               (Q.ufl_element(), Q.mesh().id(), Q.mesh().num_cells()))
               
        if key not in cache:
            cache[key] = f(V, Q)
        return cache[key]
    
    return cached_interpolation_mat


@memoize_interp
def interpolation_mat(V, Q):
    '''
    Matrix representation of interpolation operator from V to Q
    '''
    # Compatibility of spaces
    assert V.ufl_element().value_shape() == Q.ufl_element().value_shape()
    # We assume that the spaces are constructed on the same mesh
    assert V.mesh().id() == Q.mesh().id()

    # The idea is to evaluate Q's degrees of freedom at basis functions of V
    V_dm = V.dofmap()
    V_basis_f = FEBasisFunction(V)

    Q_dm = Q.dofmap()
    Q_dof = DegreeOfFreedom(Q)

    visited_rows = np.zeros(Q.dim(), dtype=bool)
    # Column values for row
    column_values = np.zeros(V_basis_f.elm.space_dimension(), dtype='double')
    with petsc_serial_matrix(Q, V) as mat:

        for cell in range(V.mesh().num_cells()):
            Q_dof.cell = cell
            V_basis_f.cell = cell

            cell_rows = Q_dm.cell_dofs(cell)
            column_indices = np.array(V_dm.cell_dofs(cell), dtype='int32')
            for local_row, row in enumerate(cell_rows):
                if visited_rows[row]: continue
                
                visited_rows[row] = True
                # Define dof
                Q_dof.dof = local_row

                # Eval at V basis functions
                for local_col, col in enumerate(column_indices):
                    # Set which basis foo
                    V_basis_f.dof = local_col
                    column_values[local_col] = Q_dof.eval(V_basis_f)
                # Can fill the matrix now
                mat.setValues([row], column_indices, column_values, PETSc.InsertMode.INSERT_VALUES)
    return mat

# --------------------------------------------------------------------

if __name__ == '__main__':
    from dolfin import *
    import block
    
    mesh = UnitSquareMesh(4, 4)

    V = FunctionSpace(mesh, 'BDM', 1)
    Q = VectorFunctionSpace(mesh, 'CG', 1)

    f = Expression(('x[0]+x[1]', 'x[1]'), degree=1)
    v = interpolate(f, V)
    
    I = PETScMatrix(interpolation_mat(V, Q))

    q = Function(Q, I*v.vector())

    print(sqrt(abs(assemble(inner(q - f, q - f)*dx))))
