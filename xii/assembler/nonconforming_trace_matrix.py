from xii.linalg.matrix_utils import petsc_serial_matrix
from xii.assembler.trace_assembly import trace_cell
from xii.assembler.interpolation_matrix import interpolation_mat
from xii.assembler.fem_eval import DegreeOfFreedom, FEBasisFunction
from xii.linalg.matrix_utils import as_petsc
from xii.linalg.convert import convert

from petsc4py import PETSc
import dolfin as df
import numpy as np
import block


def nonconforming_trace_mat(V, T):
    '''
    Matrix taking function f from d dim space V to g in (d-1) space T. 
    T(f) ~ g should hold.
    '''
    # For this to work I only make sure that function values are the same
    assert V.dolfin_element().value_rank() == T.dolfin_element().value_rank()
    assert V.ufl_element().value_shape() == T.ufl_element().value_shape()

    # I want to evaluate T degrees of freedom at V basis functions, i.e.
    # L^T_k{phi^V_l}. The picture is
    #
    #   ------
    #   \    /\
    # ===\==/==\====T
    #     \/----\
    #
    # and thus is assume that each dof of T space (row) will involve
    # collisions with several cells/basis function of V space. However
    # here we only snap to the first one
    
    mesh = V.mesh()  # The (d-1)trace mesh
    tree = mesh.bounding_box_tree()
    limit = mesh.topology().size_global(mesh.topology().dim())

    Tdm = T.dofmap()
    elm_T = T.element()
    # Colliding cells with trace dofs
    collisions = []
    for Tdof_x in T.tabulate_dof_coordinates().reshape((T.dim(), -1)):
        c = tree.compute_first_entity_collision(df.Point(*Tdof_x))
        # Contained?
        c >= limit and df.warning('Some colliding cells not found')
            
        collisions.append(c)

    # So we fill rows by checking basis functions of on the isected cells
    Vdm = V.dofmap()
    elm_V = V.element()
    
    V_basis_function = FEBasisFunction(V)
    T_degree_of_freedom = DegreeOfFreedom(T)

    X_T = T.tabulate_dof_coordinates().reshape((T.dim(), -1))
    X_V = V.tabulate_dof_coordinates().reshape((V.dim(), -1))
    
    visited_dofs = np.zeros(T.dim(), dtype=bool)
    col_values = np.zeros(V_basis_function.elm.space_dimension(), dtype='double')
    with petsc_serial_matrix(T, V) as mat:

        for Tcell in range(T.mesh().num_cells()):
            # Set for this cell
            T_degree_of_freedom.cell = Tcell
            
            Tdofs = Tdm.cell_dofs(Tcell)
            for local_T, Tdof in enumerate(Tdofs):
                # Seen the row?
                if visited_dofs[Tdof]: continue

                visited_dofs[Tdof] = True

                # Set to current dof
                T_degree_of_freedom.dof = local_T

                # Now all the V cells and their basis functions
                c = collisions[Tdof]
                # If we have no reasonable cell leave the row empty
                if c >= limit: continue
                
                # Set the dof cell                
                V_basis_function.cell = c
                    
                Vdofs = np.array(Vdm.cell_dofs(c), dtype='int32')  # These are columns
                    
                # Fill column
                for local_V, Vdof in enumerate(Vdofs):
                    # Set as basis_function
                    V_basis_function.dof = local_V

                    # Evaluate trace dof at basis function
                    dof_value = T_degree_of_freedom.eval(V_basis_function)

                    col_values[local_V] = dof_value

                mat.setValues([Tdof], Vdofs, col_values, PETSc.InsertMode.INSERT_VALUES)
    return mat

# --------------------------------------------------------------------

if __name__ == '__main__':
    from dolfin import *

    
    mesh = UnitSquareMesh(32, 32)
    V = FunctionSpace(mesh, 'RT', 1)

    print(cg_element(V.ufl_element()))
