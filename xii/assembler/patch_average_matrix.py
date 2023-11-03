from xii.linalg.matrix_utils import petsc_serial_matrix, is_number
from xii.assembler.patch_average_form import patch_average_cell, patch_average_space

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
               (TV.ufl_element(), TV.mesh().id()),
               data['vertex_f'], data['patch_f'])

        if key not in cache:
            cache[key] = average_mat(V, TV, reduced_mesh, data)
        return cache[key]
    
    return cached_average_mat


@memoize_average
def patch_avg_mat(V, TV, reduced_mesh, data):
    '''
    A mapping for computing the patch averages of function in V in the 
    space TV. 
    '''
    vertex_f, patch_f = data['vertex_f'], data['patch_f']

    if data['patch_coloring'] is not None:
        return patch_avg_mat_coloring(V, TV, reduced_mesh, data)
    
    Vgamma = vertex_f.function_space() 
    gamma = Vgamma.mesh()
    assert gamma.id() == reduced_mesh.id()
    
    assert TV.mesh().id() == reduced_mesh.id()
    
    # Compatibility of spaces
    # FIXME: only want scalars now
    assert V.ufl_element().value_shape() == TV.ufl_element().value_shape() == ()
    
    assert patch_average_cell(V) == TV.mesh().ufl_cell()
    assert V.mesh().geometry().dim() == TV.mesh().geometry().dim()
    assert V.mesh().id() == patch_f.mesh().id()
    
    omega = patch_f.mesh()
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
    
    patch_volumes = np.array([cell.volume() for cell in df.cells(patch_f.mesh())])
    patch_f = patch_f.array()
    color_ids = vertex_f.vector().get_local()

    ii, jj, values = [], [], []
    for row in tqdm.tqdm(range(TV.dim()), total=TV.dim()):  # dofs of TV define rows
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


@memoize_average
def patch_avg_mat_coloring(V, TV, reduced_mesh, data):
    '''
    Use connectivity of patches to assemble once over the disconnected once 
    of the same color
    '''
    vertex_f, patch_f, coloring = data['vertex_f'], data['patch_f'], data['patch_coloring']
    
    Vgamma = vertex_f.function_space() 
    gamma = Vgamma.mesh()
    assert gamma.id() == reduced_mesh.id()
    
    assert TV.mesh().id() == reduced_mesh.id()
    
    # Compatibility of spaces
    # FIXME: only want scalars now
    assert V.ufl_element().value_shape() == TV.ufl_element().value_shape() == ()
    
    assert patch_average_cell(V) == TV.mesh().ufl_cell()
    assert V.mesh().geometry().dim() == TV.mesh().geometry().dim()
    assert V.mesh().id() == patch_f.mesh().id()

    color_ids = vertex_f.vector().get_local()    
    # We will do the integration a group at a time
    groups = defaultdict(list)
    for color in np.asarray(color_ids, dtype='uintp'):
        group = coloring[color]
        groups[group].append(color)

    omega = patch_f.mesh()
    v = df.TestFunction(V)
    chi = df.MeshFunction('size_t', omega, omega.topology().dim(), 0)
    chi_array = chi.array()
    # Wire up characteristic function
    dx = df.Measure('dx', domain=omega, subdomain_data=chi)
    # Now the averaging would be done as by integrating using indicator function
    average_form = v*dx(1)  
    # This gives us all the dofs but we are only after some
    Vdm = V.dofmap()
    v_dofs = df.Function(V).vector()
    
    patch_volumes = np.array([cell.volume() for cell in df.cells(patch_f.mesh())])
    patch_f = patch_f.array()

    # The new thing here is to assemble once over all the patches that are not neighbors. 
    ii, jj, values = [], [], []
    for group in tqdm.tqdm(groups):
        group_colors = groups[group]
        # Collect info per color in the group
        volumes, columns, rows = {}, {}, {}
        for color in group_colors:
            patch_cells, = np.where(patch_f == color)
            patch_volume = np.sum(patch_volumes[patch_cells])
            # Add the indicator for this color
            chi_array[patch_cells] = 1

            patch_dofs = np.unique(np.hstack([Vdm.cell_dofs(cell) for cell in patch_cells]))

            volumes[color] = patch_volume
            columns[color] = patch_dofs

            d = np.abs(color_ids - color)
            index = np.argmin(d)
            assert d[index] < 1E-13
            
            rows[color] = index
        # Integrate over group!
        df.assemble(average_form, tensor=v_dofs)
        v_dofs_arr = v_dofs.get_local()
        # Extract
        for color in group_colors:
            patch_values = v_dofs_arr[columns[color]]/volumes[color]
            # Update matrix
            ii.extend(rows[color]*np.ones_like(columns[color]))
            jj.extend(columns[color])
            values.extend(patch_values)
            
        # Reset for next round
        chi_array[:] = 0
        
    matrix = sp.csr_matrix((values, (ii, jj)), shape=(TV.dim(), V.dim()))
    mat = PETSc.Mat().createAIJ(comm=PETSc.COMM_WORLD,
                                size=matrix.shape,
                                csr=(matrix.indptr, matrix.indices, matrix.data))

    return df.PETScMatrix(mat)
