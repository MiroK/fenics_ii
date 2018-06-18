from dolfin import (PETScMatrix, Matrix, IndexMap, PETScVector, Vector,
                    as_backend_type, mpi_comm_world)
from block import block_mat, block_vec
from scipy.sparse import csr_matrix
from contextlib import contextmanager
from petsc4py import PETSc
import numpy as np


def is_petsc_vec(v):
    '''Check Vector type'''
    return isinstance(v, (PETScVector, Vector))


def is_petsc_mat(A):
    '''Check Matrix type'''
    return isinstance(A, (PETScMatrix, Matrix))


def is_number(i):
    '''Check number type'''
    return isinstance(i, (float, int))


def as_petsc(A):
    '''Extract pointer to the underlying PETSc object'''
    if is_petsc_vec(A):
        return as_backend_type(A).vec()
    if is_petsc_mat(A):
        return as_backend_type(A).mat()

    raise ValueError('%r is not matrix/vector.' % type(A))


def transpose_matrix(A):
    '''Create a transpose of PETScMatrix/PETSc.Mat'''
    if isinstance(A, PETSc.Mat):
        At = PETSc.Mat()  # Alloc
        A.transpose(At)  # Transpose to At
        return At

    At = transpose_matrix(as_backend_type(A).mat())
    return PETScMatrix(At)


def diagonal_matrix(size, A=1):
    '''Dolfin A*I serial only'''
    d = PETSc.Vec().createWithArray(A*np.ones(size))
    I = PETSc.Mat().createAIJ(size=size, nnz=1)
    I.setDiagonal(d)
    I.assemble()

    return PETScMatrix(I)


def zero_matrix(nrows, ncols):
    '''Zero matrix'''
    mat = csr_matrix((np.zeros(nrows, dtype=float),  # Data
                      # Rows, cols = so first col in each row is 0
                      (np.arange(nrows), np.zeros(nrows, dtype=int))),  
                     shape=(nrows, ncols))

    A = PETSc.Mat().createAIJ(size=[[nrows, nrows], [ncols, ncols]],
                              csr=(mat.indptr, mat.indices, mat.data))
    A.assemble()

    return PETScMatrix(A)


@contextmanager
def petsc_serial_matrix(test_space, trial_space, nnz=None):
    '''
    PETsc.Mat from trial_space to test_space to be filled in the 
    with block. The spaces can be represented by intergers meaning 
    generic R^n.
    '''
    # Decide local to global map
    # For our custom case everything is serial
    if is_number(test_space) and is_number(trial_space):
        comm = mpi_comm_world().tompi4py()
        # Local same as global
        sizes = [[test_space, test_space], [trial_space, trial_space]]

        row_map = PETSc.IS().createStride(test_space, 0, 1, comm)
        col_map = PETSc.IS().createStride(trial_space, 0, 1, comm)
    # With function space this can be extracted
    else:
        mesh = test_space.mesh()
        comm = mesh.mpi_comm().tompi4py()
        
        row_map = test_space.dofmap()
        col_map = trial_space.dofmap()
    
        sizes = [[row_map.index_map().size(IndexMap.MapSize_OWNED),
                  row_map.index_map().size(IndexMap.MapSize_GLOBAL)],
                 [col_map.index_map().size(IndexMap.MapSize_OWNED),
                  col_map.index_map().size(IndexMap.MapSize_GLOBAL)]]

        row_map = map(int, row_map.tabulate_local_to_global_dofs())
        col_map = map(int, col_map.tabulate_local_to_global_dofs())
        
    assert comm.size == 1

    lgmap = lambda indices: (PETSc.LGMap().create(indices, comm=comm)
                             if isinstance(indices, list)
                             else
                             PETSc.LGMap().createIS(indices))
    
    row_lgmap, col_lgmap = map(lgmap, (row_map, col_map))


    # Alloc
    mat = PETSc.Mat().createAIJ(sizes, nnz=nnz, comm=comm)
    mat.setUp()
    
    mat.setLGMap(row_lgmap, col_lgmap)

    mat.assemblyBegin()
    # Fill
    yield mat
    # Tear down
    mat.assemblyEnd()
