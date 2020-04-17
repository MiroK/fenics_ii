from xii.linalg.matrix_utils import petsc_serial_matrix
from xii.assembler.fem_eval import DegreeOfFreedom, FEBasisFunction
from xii import *

from dolfin import PETScMatrix
from petsc4py import PETSc
import numpy as np


# Restriction operators are potentially costly so we memoize the results.
# Let every operator deal with cache keys as it sees fit
def memoize_inj(inj_mat):
    '''Cached injection'''
    cache = {}
    def cached_inj_mat(Vc, Vf, fine_mesh, data):
        key = ((Vc.ufl_element(), Vc.mesh().id()),
               (Vf.ufl_element(), Vf.mesh().id()),
               fine_mesh.id())
               
        if key not in cache:
            cache[key] = inj_mat(Vc, Vf, fine_mesh, data)
        return cache[key]

    return cached_inj_mat


@memoize_inj
def injection_matrix(Vc, Vf, fine_mesh, data):
    '''Injection mapping from Vc to Vf'''
    mesh_c = Vc.mesh()
    assert fine_mesh.id() == Vf.mesh().id()
    mesh_f = Vf.mesh()

    assert mesh_f.has_parent() and mesh_c.has_child()
    assert mesh_f.parent().id() == mesh_c.id()

    tdim = mesh_f.topology().dim()
    # NOTE: if mesh_f came from adapt of coarse then we can access the
    # data through API
    if mesh_f.data().exists('parent_cell', tdim):
        fine_to_coarse = mesh_f.data().array('parent_cell', tdim)
    # FIXME: However, I can't do it when I refine myself to as a fall back
    else:
        keys, fine_to_coarse = zip(*fine_mesh.parent_entity_map[mesh_c.id()][tdim].items())
        fine_to_coarse = np.array(fine_to_coarse, dtype='uintp')
        fine_to_coarse[np.argsort(keys)] = fine_to_coarse
        
    # The idea is to evaluate Vf's degrees of freedom at basis functions of Vc
    fdmap = Vf.dofmap()
    Vf_dof = DegreeOfFreedom(Vf)

    cdmap = Vc.dofmap()
    Vc_basis_f = FEBasisFunction(Vc)

    # Column values
    visited_rows = [False]*Vf.dim()
    row_values = np.zeros(Vc_basis_f.elm.space_dimension(), dtype='double')

    with petsc_serial_matrix(Vf, Vc) as mat:
        
        for f_cell, c_cell in enumerate(fine_to_coarse):

            Vc_basis_f.cell = c_cell
            # These are the colums
            coarse_dofs = cdmap.cell_dofs(c_cell)

            Vf_dof.cell = f_cell
            
            fine_dofs = fdmap.cell_dofs(f_cell)

            for local, dof in enumerate(fine_dofs):
                if visited_rows[dof]:
                    continue
                else:
                    visited_rows[dof] = True
                            
                Vf_dof.dof = local
                # Evaluete coarse basis foos here
                for local_c, dof_c in enumerate(coarse_dofs):
                    Vc_basis_f.dof = local_c
                    row_values[local_c] = Vf_dof.eval(Vc_basis_f)
                # Insert
                mat.setValues([dof], coarse_dofs, row_values, PETSc.InsertMode.INSERT_VALUES)

    return PETScMatrix(mat)
