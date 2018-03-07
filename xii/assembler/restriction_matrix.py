from xii.linalg.matrix_utils import petsc_serial_matrix
from xii.assembler.restriction_assembly import restriction_cell
from xii.assembler.fem_eval import DegreeOfFreedom, FEBasisFunction

from dolfin import Cell, PETScMatrix
from petsc4py import PETSc
import numpy as np


def restriction_mat(V, TV, rmesh, data):
    '''
    A mapping for computing restriction of function in V in TV. If f in V 
    then g in TV has coefficients equal to dofs_{TV}(trace V).

    The TV space is assumed to be setup on subdmoain of V mesh.
    '''
    # Compatibility of spaces
    assert V.ufl_element() == TV.ufl_element()
    # Check that rmesh came from OverlapMesh
    assert V.mesh().id() in rmesh.parent_entity_map

    return PETScMatrix(restriction_matrix(V, TV, rmesh))
                

def restriction_matrix(V, TV, rmesh):
    '''The first cell connected to the facet gets to set the values of TV'''
    assert TV.mesh().id() == rmesh.id()
    
    mesh = V.mesh()
    tdim = mesh.topology().dim()
    
    # Let's get the mapping or cell of TV mesh to V mesh cells
    mapping = rmesh.parent_entity_map[mesh.id()][tdim]  
    # The idea is to evaluate TV's degrees of freedom at basis functions
    # of V
    Tdmap = TV.dofmap()
    TV_dof = DegreeOfFreedom(TV)

    dmap = V.dofmap()
    V_basis_f = FEBasisFunction(V)

    # Rows
    visited_dofs = [False]*TV.dim()
    # Column values
    dof_values = np.zeros(V_basis_f.elm.space_dimension(), dtype='double')
    with petsc_serial_matrix(TV, V) as mat:

        for trace_cell in range(TV.mesh().num_cells()):
            TV_dof.cell = trace_cell
            trace_dofs = Tdmap.cell_dofs(trace_cell)
            # The corresponding cell in V mesh
            cell = mapping[trace_cell]
            V_basis_f.cell = cell
            
            dofs = dmap.cell_dofs(cell)
            for local_T, dof_T in enumerate(trace_dofs):

                if visited_dofs[dof_T]:
                    continue
                else:
                    visited_dofs[dof_T] = True

                # Define trace dof
                TV_dof.dof = local_T
                
                # Eval at V basis functions
                for local, dof in enumerate(dofs):
                    # Set which basis foo
                    V_basis_f.dof = local
                    
                    dof_values[local] = TV_dof.eval(V_basis_f)

                # Can fill the matrix now
                col_indices = np.array(dofs, dtype='int32')
                # Insert
                mat.setValues([dof_T], col_indices, dof_values, PETSc.InsertMode.INSERT_VALUES)
    return mat
