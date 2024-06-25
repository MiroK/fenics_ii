from xii.linalg.matrix_utils import petsc_serial_matrix, is_number
from xii.assembler.mean_form import mean_cell, mean_space

from numpy.polynomial.legendre import leggauss
from dolfin import PETScMatrix, cells, Point, Cell, Function
from collections import defaultdict
import scipy.sparse as sp
from petsc4py import PETSc
import dolfin as df
import numpy as np
import tqdm


def memoize_average(average_mat):
    '''Cached average'''
    cache = {}
    def cached_average_mat(V, TV, reduced_mesh, data):
        key = ((V.ufl_element(), V.mesh().id()),
               (TV.ufl_element(), TV.mesh().id()))

        if key not in cache:
            cache[key] = average_mat(V, TV, reduced_mesh, data)
        return cache[key]
    
    return cached_average_mat


@memoize_average
def mean_mat(V, TV, reduced_mesh, data):
    '''Do it'''
    # FIXME: only want scalars now
    assert V.ufl_element().value_shape() == TV.ufl_element().value_shape() == ()
    
    assert mean_cell(V) == TV.mesh().ufl_cell()
    assert V.mesh().geometry().dim() == TV.mesh().geometry().dim()
    
    omega = TV.mesh()
    v = df.TestFunction(V)
    chi = df.MeshFunction('size_t', omega, omega.topology().dim(), 0)
    chi_array = chi.array()
    # Wire up characteristic function
    dx = df.Measure('dx', domain=omega, subdomain_data=chi)
    # Now the averaging would be done as by integrating using indicator ...
    patch_volume0 = df.Constant(1)
    # ... function
    average_form = (1/patch_volume0)*v*dx(1)  
    # This gives us all the dofs but we are only after some
    Vdm = V.dofmap()
    v_dofs = df.Function(V).vector()
    
    patch_volumes = np.array([cell.volume() for cell in df.cells(omega)])

    patch_f = df.MeshFunction('size_t', omega, omega.topology().dim(), 1)
    patch_f = patch_f.array()
    color_ids = np.array([1])

    ii, jj, values = [], [], []
    for row in range(TV.dim()):  # dofs of TV define rows
        color = color_ids[row]

        patch_cells, = np.where(patch_f == color)
        patch_volume = np.sum(patch_volumes[patch_cells])
        # Update indicator
        chi_array[patch_cells] = 1
        patch_volume0.assign(df.Constant(patch_volume))
        # Integrate
        df.assemble(average_form, tensor=v_dofs)
        # Extract
        columns = np.unique(np.hstack([Vdm.cell_dofs(cell) for cell in patch_cells]))
        patch_values = v_dofs.get_local()[columns]
        # Reset for next round
        chi_array[:] = 0
        
        # Update matrix
        ii.extend(row*np.ones_like(columns))
        jj.extend(columns)
        values.extend(patch_values)
        
    matrix = sp.csr_matrix((values, (ii, jj)), shape=(TV.dim(), V.dim()))
    mat = PETSc.Mat().createAIJ(comm=PETSc.COMM_WORLD,
                                size=matrix.shape,
                                csr=(matrix.indptr, matrix.indices, matrix.data))

    return df.PETScMatrix(mat)
