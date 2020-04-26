from __future__ import absolute_import
from dolfin import PETScMatrix, Matrix, IndexMap, PETScVector, Vector, as_backend_type, FunctionSpace

# Compatibility with FEniCS 2017
try:
    from dolfin import mpi_comm_world

    def comm4py(comm):
        return comm.tompi4py()
except ImportError:
    from dolfin import MPI
    
    def mpi_comm_world():
        return MPI.comm_world

    def comm4py(comm):
        return comm


from block import block_mat, block_vec
from scipy.sparse import csr_matrix
from contextlib import contextmanager
from petsc4py import PETSc
import numpy as np
from six.moves import map
from six.moves import range


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


def identity_matrix(V):
    '''u -> u for u in V'''
    if isinstance(V, FunctionSpace):
        return diagonal_matrix(V.dim(), 1)

    mat = block_mat([[0]*len(V) for _ in range(len(V))])
    for i in range(len(mat)):
        mat[i][i] = identity_matrix(V[i])
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
        comm = comm4py(mpi_comm_world())
        # Local same as global
        sizes = [[test_space, test_space], [trial_space, trial_space]]

        row_map = PETSc.IS().createStride(test_space, 0, 1, comm)
        col_map = PETSc.IS().createStride(trial_space, 0, 1, comm)
    # With function space this can be extracted
    else:
        mesh = test_space.mesh()
        comm = comm4py(mesh.mpi_comm())
        
        row_map = test_space.dofmap()
        col_map = trial_space.dofmap()

        if hasattr(IndexMap, 'MapSize'):
            sizes = [[row_map.index_map().size(IndexMap.MapSize.OWNED),
                      row_map.index_map().size(IndexMap.MapSize.GLOBAL)],
                     [col_map.index_map().size(IndexMap.MapSize.OWNED),
                      col_map.index_map().size(IndexMap.MapSize.GLOBAL)]]
        else:
            sizes = [[row_map.index_map().size(IndexMap.MapSize_OWNED),
                      row_map.index_map().size(IndexMap.MapSize_GLOBAL)],
                     [col_map.index_map().size(IndexMap.MapSize_OWNED),
                      col_map.index_map().size(IndexMap.MapSize_GLOBAL)]]

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
