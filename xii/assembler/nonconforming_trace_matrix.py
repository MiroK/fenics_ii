from xii.linalg.matrix_utils import petsc_serial_matrix
from xii.assembler.trace_assembly import trace_cell
from xii.assembler.fem_eval import DegreeOfFreedom, FEBasisFunction
from dolfin import warning, cells, Point
from petsc4py import PETSc
import numpy as np


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
    # collisions with several cells/basis function of V space
    
    mesh = V.mesh()  # The (d-1)trace mesh
    tree = mesh.bounding_box_tree()
    limit = mesh.topology().size_global(mesh.topology().dim())

    Tdm = T.dofmap()
    elm_T = T.element()
    # Colliding cells with trace dofs
    collisions = []
    for Tdof_x in T.tabulate_dof_coordinates().reshape((T.dim(), -1)):
        cs = tree.compute_entity_collisions(Point(*Tdof_x))
        if any(c >= limit for c in cs):
            warning('Some colliding cells not found')
            cs = filter(lambda c: c < limit, cs)
        collisions.append(cs)

    # So we fill rows by checking basis functions of on the isected cells
    Vdm = V.dofmap()
    elm_V = V.element()
    
    V_basis_function = FEBasisFunction(V)
    T_degree_of_freedom = DegreeOfFreedom(T)

    visited_dofs = [False]*T.dim()
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

                col_indices, col_values = [], []
                # Now all the V cells and their basis functions
                for c in collisions[Tdof]:
                    # Set the dof cell
                    V_basis_function.cell = c
                    
                    Vdofs = Vdm.cell_dofs(c)
                    # Columns
                    for local_V, Vdof in enumerate(Vdofs):
                        if Vdof in col_indices:
                            continue
                        
                        # Set as basis_function
                        V_basis_function.dof = local_V

                        # Evaluate trace dof at basis function
                        dof_value = T_degree_of_freedom.eval(V_basis_function)
                        col_indices.append(Vdof)
                        col_values.append(dof_value)
                # Can fill the matrix row
                col_indices = np.array(col_indices, dtype='int32')
                col_values = np.array(col_values)

                mat.setValues([Tdof], col_indices, col_values, PETSc.InsertMode.INSERT_VALUES)
    return mat
