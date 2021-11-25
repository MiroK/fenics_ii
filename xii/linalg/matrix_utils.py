from dolfin import (PETScMatrix, Matrix, IndexMap, PETScVector, Vector,
                    as_backend_type, FunctionSpace, MPI)
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


def diagonal_matrix(size, A):
    '''Dolfin A*I serial only'''
    if isinstance(A, (int, float)):
        d = PETSc.Vec().createWithArray(A*np.ones(size))
    else:
        d = as_backend_type(A).vec()
    I = PETSc.Mat().createAIJ(size=size, nnz=1)
    I.setDiagonal(d)
    I.assemble()

    return PETScMatrix(I)


def identity_matrix(V):
    '''u -> u for u in V'''
    if isinstance(V, FunctionSpace):
        return diagonal_matrix(V.dim(), 1)

    mat = block_mat([[0]*len(V) for _ in range(len(V))])
    for i in range(len(mat)):
        mat[i][i] = identity_matrix(V[i])
    return mat


def block_reshape(AA, offsets):
    '''Group rows/cols according to offsets'''
    nblocks = len(offsets)
    mat = block_mat([[0]*nblocks for _ in range(nblocks)])

    offsets = [0] + list(offsets)
    AA = AA.blocks
    for row, (ri, rj) in enumerate(zip(offsets[:-1], offsets[1:])):
        for col, (ci, cj) in enumerate(zip(offsets[:-1], offsets[1:])):
            if rj-ri == 1 and cj -ci == 1:
                mat[row][col] = AA[ri, ci]
            else:
                mat[row][col] = block_mat(AA[ri:rj, ci:cj])

    return mat


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


def row_matrix(rows):
    '''Short and fat matrix'''
    ncols, = set(row.size() for row in rows)
    nrows = len(rows)

    indptr = np.cumsum(np.array([0]+[ncols]*nrows))
    indices = np.tile(np.arange(ncols), nrows)
    data = np.hstack([row.get_local() for row in rows])

    mat = csr_matrix((data, indices, indptr), shape=(nrows, ncols))
    
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
        comm = MPI.comm_world
        # Local same as global
        sizes = [[test_space, test_space], [trial_space, trial_space]]

        row_map = PETSc.IS().createStride(test_space, 0, 1, comm)
        col_map = PETSc.IS().createStride(trial_space, 0, 1, comm)
    # With function space this can be extracted
    else:
        mesh = test_space.mesh()
        comm = mesh.mpi_comm()
        
        row_map = test_space.dofmap()
        col_map = trial_space.dofmap()
    
        sizes = [[row_map.index_map().size(IndexMap.MapSize.OWNED),
                  row_map.index_map().size(IndexMap.MapSize.GLOBAL)],
                 [col_map.index_map().size(IndexMap.MapSize.OWNED),
                  col_map.index_map().size(IndexMap.MapSize.GLOBAL)]]

        row_map = list(map(int, row_map.tabulate_local_to_global_dofs()))
        col_map = list(map(int, col_map.tabulate_local_to_global_dofs()))
        
    assert comm.size == 1

    lgmap = lambda indices: (PETSc.LGMap().create(indices, comm=comm)
                             if isinstance(indices, list)
                             else
                             PETSc.LGMap().createIS(indices))
    
    row_lgmap, col_lgmap = list(map(lgmap, (row_map, col_map)))


    # Alloc
    mat = PETSc.Mat().createAIJ(sizes, nnz=nnz, comm=comm)
    mat.setUp()
    
    mat.setLGMap(row_lgmap, col_lgmap)

    mat.assemblyBegin()
    # Fill
    yield mat
    # Tear down
    mat.assemblyEnd()
