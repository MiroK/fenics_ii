from xii.linalg.matrix_utils import (is_petsc_vec, is_petsc_mat, sizes, diagonal_matrix,
                                     is_number, as_petsc, petsc_serial_matrix)

from block.block_compose import block_mul, block_add, block_sub, block_transpose
from block import block_mat, block_vec
from dolfin import PETScVector, PETScMatrix, mpi_comm_world
from scipy.sparse import bmat as numpy_block_mat
from scipy.sparse import csr_matrix
from petsc4py import PETSc
import numpy as np
import itertools
import operator


def convert(bmat, algorithm='numpy'):
    '''
    Attempt to convert bmat to a PETSc(Matrix/Vector) object.
    If succed this is at worst a number.
    '''
    # Block vec conversion
    if isinstance(bmat, block_vec):
        array = block_vec_to_numpy(bmat)
        vec = PETSc.Vec().createWithArray(array)
        return PETScVector(vec)
    
    # Conversion of bmat is bit more involved because of the possibility
    # that some of the blocks are numbers or composition of matrix operations
    if isinstance(bmat, block_mat):
        # Create collpsed bmat
        row_sizes, col_sizes = sizes(bmat)
        nrows, ncols = len(row_sizes), len(col_sizes)
        indices = itertools.product(range(nrows), range(ncols))
        
        blocks = np.zeros((nrows, ncols), dtype='object')
        for block, (i, j) in zip(bmat.blocks.flatten(), indices):
            # This might is guaranteed to be matrix or number
            A = collapse(block)
            # Only numbers on the diagonal block are interpresented as
            # scaled identity matrices
            if is_number(A):
                assert row_sizes[i] == col_sizes[j]
                A = diagonal_matrix(row_sizes[i], A)
            # The converted block
            blocks[i, j] = A
        # Now every block is a matrix/number and we can make a monolithic thing
        bmat = block_mat(blocks)

        assert all(is_petsc_mat(block) for block in bmat.blocks.flatten())
        # Do this via scipy sparse bmat
        if algorithm == 'numpy':
            # Convert to numpy
            array = block_mat_to_numpy(bmat)
            # Constuct from numpy
            return numpy_to_petsc(array)
        # Manual (maybe faster, let's see)
        else:
            return block_mat_to_petsc(bmat)

    # Try with a composite
    return collapse(bmat)


def collapse(bmat):
    '''Collapse what are blocks of bmat'''
    # Single block cases
    # Do nothing
    if is_petsc_mat(bmat) or is_number(bmat) or is_petsc_vec(bmat):
        return bmat

    # Multiplication
    if isinstance(bmat, block_mul):
        return collapse_mul(bmat)
    # +
    elif isinstance(bmat, block_add):
        return collapse_add(bmat)
    # -
    elif isinstance(bmat, block_sub):
        return collapse_sub(bmat)
    # T
    elif isinstance(bmat, block_transpose):
        return collapse_tr(bmat)
    # Some things in cbc.block know their matrix representation
    # E.g. InvLumpDiag...
    elif hasattr(bmat, 'A'):
        assert is_petsc_mat(bmat.A)
        return bmat.A

    raise ValueError('Do not know how to collapse %r' % type(bmat))


def collapse_tr(bmat):
    '''to Transpose'''
    # Base
    A = bmat.A
    if is_petsc_mat(A):
        A_ = as_petsc(A)
        C_ = PETSc.Mat()
        A_.transpose(C_)
        return PETScMatrix(C_)
    # Recurse
    return collapse_tr(collapse(bmat))

def collapse_add(bmat):
    '''A + B to single matrix'''
    A, B = bmat.A, bmat.B
    # Base case
    if is_petsc_mat(A) and is_petsc_mat(B):
        A_ = as_petsc(A)
        B_ = as_petsc(B)
        assert A_.size == B_.size
        C_ = A_.copy()
        # C = A + B
        C_.axpy(1., B_, PETSc.Mat.Structure.DIFFERENT)
        return PETScMatrix(C_)
    # Recurse
    return collapse_add(collapse(A) + collapse(B))


def collapse_sub(bmat):
    '''A - B to single matrix'''
    A, B = bmat.A, bmat.B
    # Base case
    if is_petsc_mat(A) and is_petsc_mat(B):
        A_ = as_petsc(A)
        B_ = as_petsc(B)
        assert A_.size == B_.size
        C_ = A_.copy()
        # C = A - B
        C_.axpy(-1., B_, PETSc.Mat.Structure.DIFFERENT)
        return PETScMatrix(C_)
    # Recurse
    return collapse_sub(collapse(A) - collapse(B))


def collapse_mul(bmat):
    '''A*B*C to single matrix'''
    # A0 * A1 * ...
    A, B = bmat.chain[0], bmat.chain[1:]

    if len(B) == 1:
        B = B[0]
        # Two matrices
        if is_petsc_mat(A) and is_petsc_mat(B):
            A_ = as_petsc(A)
            B_ = as_petsc(B)
            assert A_.size[1] == B_.size[0]
            C_ = PETSc.Mat()
            A_.matMult(B_, C_)

            return PETScMatrix(C_)
        # One of them is a number
        elif is_petsc_mat(A) and is_number(B):
            A_ = as_petsc(A)
            C_ = A_.copy()
            C_.scale(B)
            return PETScMatrix(C_)

        elif is_petsc_mat(B) and is_number(A):
            B_ = as_petsc(B)
            C_ = B_.copy()
            C_.scale(A)
            return PETScMatrix(C_)
        # Some compositions
        else:
            return collapse(collapse(A)*collapse(B))
    # Recurse
    else:
        return collapse_mul(collapse(A)*collapse(reduce(operator.mul, B)))                                    

# Conversion via numpy
def block_vec_to_numpy(bvec):
    '''Collapsing block bector to numpy array'''
    return np.hstack([v.get_local() for v in bvec])


def block_mat_to_numpy(bmat):
    '''Collapsing block mat of matrices to scipy's bmat'''
    # A single matrix
    if is_petsc_mat(bmat):
        bmat = as_petsc(bmat)
        return csr_matrix(bmat.getValuesCSR()[::-1], shape=bmat.size)
    # Recurse on blocks
    blocks = np.array(map(block_mat_to_numpy, bmat.blocks.flatten()))
    blocks = blocks.reshape(bmat.blocks.shape)
    # The bmat
    return numpy_block_mat(blocks).tocsr()


def numpy_to_petsc(mat):
    '''Build PETScMatrix with array structure'''
    # Dense array to matrix
    if isinstance(mat, np.ndarray):
        return numpy_to_petsc(csr_matrix(mat))
    # Sparse
    A = PETSc.Mat().createAIJ(size=mat.shape,
                              csr=(mat.indptr, mat.indices, mat.data)) 
    return PETScMatrix(A)


def block_mat_to_petsc(bmat):
    '''Block mat to PETScMatrix via assembly'''
    # This is beautiful but slow as hell :)
    def iter_rows(matrix):
        for i in range(matrix.size(0)):
            yield matrix.getrow(i)

    row_sizes, col_sizes = sizes(bmat)
    row_offsets = np.cumsum([0] + list(row_sizes))
    col_offsets = np.cumsum([0] + list(col_sizes))

    with petsc_serial_matrix(row_offsets[-1], col_offsets[-1]) as mat:
        row = 0
        for row_blocks in bmat.blocks:
            # Zip the row iterators of the matrices together
            for indices_values in itertools.izip(*map(iter_rows, row_blocks)):
                indices, values = zip(*indices_values)

                indices = [list(index+offset) for index, offset in zip(indices, col_offsets)]
                indices = sum(indices, [])
            
                row_values = np.hstack(values)

                mat.setValues([row], indices, row_values, PETSc.InsertMode.INSERT_VALUES)

                row += 1
    return PETScMatrix(mat)


# -------------------------------------------------------------------


if __name__ == '__main__':
    from dolfin import *

    mesh = UnitSquareMesh(32, 32)
    V = FunctionSpace(mesh, 'CG', 1)
    Q = FunctionSpace(mesh, 'DG', 0)
    W = [V, Q]
    
    u, p = map(TrialFunction, W)
    v, q = map(TestFunction, W)

    [[A00, A01],
     [A10, A11]] = [[assemble(inner(u, v)*dx), assemble(inner(v, p)*dx)],
                    [assemble(inner(u, q)*dx), assemble(inner(p, q)*dx)]]
    blocks = [[A00*A00, A01+A01],
              [2*A10 - A10, A11*A11*A11]]
    
    AA = block_mat(blocks)

    t = Timer('x'); t.start()
    X = convert(AA)
    print t.stop()

    t = Timer('x'); t.start()
    Y = convert(AA, 'foo')
    print t.stop()

    X_ = X.array()
    X_[:] -= Y.array()
    print np.linalg.norm(X_, np.inf)
