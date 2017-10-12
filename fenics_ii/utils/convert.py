from scipy.sparse import bmat, csr_matrix
from block import block_mat, block_vec
from dolfin import Matrix, Vector, as_backend_type, PETScMatrix, PETScVector
from petsc4py import PETSc
from itertools import chain
import numpy as np

null = PETSc.Mat()

def block_mat_to_matnest(block, comm=None):
    '''Conversion to PETScMatNest'''
    if isinstance(block, int):
        assert block == 0
        null = PETSc.Mat()
        return null

    if isinstance(block, (Matrix, PETScMatrix)):
        mat = as_backend_type(block).mat()
        return mat
    
    if isinstance(block, (np.ndarray, list)):
        mats = map(block_mat_to_matnest, block)
        return mats

    if isinstance(block, block_mat):
        mats = map(block_mat_to_matnest, block)
        comm = extract_comm(block)
        
        return PETSc.Mat().createNest(mats, comm=comm)


def block_to_dolfin(block):
    if isinstance(block, block_mat):
        A = block_to_scipy(block, blocks_only=False)
        A = A.tocsr()
        A = PETSc.Mat().createAIJ(size=A.shape,
                                  csr=(A.indptr, A.indices, A.data)) 
        return PETScMatrix(A)

    elif isinstance(block, block_vec):
        b = block_to_scipy(block, blocks_only=False)
        b = PETSc.Vec().createWithArray(b)
        return PETScVector(b)


def block_to_scipy(block, blocks_only=False):
    '''Converts cbc.block * to scipy'''
    if isinstance(block, block_mat): return block_mat_to_scipy(block, blocks_only)
    if isinstance(block, block_vec): return block_vec_to_scipy(block, blocks_only)
    assert False


def block_mat_to_scipy(mat, blocks_only):
    '''Converts cbc.block's block_mat to scipy's bmat'''
    if isinstance(mat, block_mat):
        blocks = [block_mat_to_scipy(mat_i, blocks_only) for mat_i in mat]
        if blocks_only:
            return blocks
        else:
            return bmat(blocks)
    elif isinstance(mat, (np.ndarray, list)):
        return [block_mat_to_scipy(mat_i, blocks_only) for mat_i in mat]
    elif isinstance(mat, (Matrix, PETScMatrix)):
        mat = as_backend_type(mat).mat()
        scipy_mat = csr_matrix(mat.getValuesCSR()[::-1], shape=mat.size)
        return scipy_mat
    else:
        assert isinstance(mat, int), type(mat)
        if mat == 0:
            return None
        else:
            return np.array([[mat]])


def block_vec_to_scipy(vec, blocks_only):
    '''Converts cbc.block's block_mat to scipy's bmat'''
    if isinstance(vec, block_vec):
        blocks = [block_vec_to_scipy(vec_i, blocks_only) for vec_i in vec]
        if blocks_only:
            return blocks
        else:
            return np.concatenate(blocks)
    else:
        assert isinstance(vec, (Vector, PETScVector)), type(vec)
        return vec.array()


def extract_comm(block):
    '''Extract mpi communicator from block_mat object'''
    if hasattr(block, '__iter__'): 
        return next(chain(map(extract_comm, block)))

    if isinstance(block, (PETScMatrix, Matrix)):
        return block.mpi_comm().tompi4py()
    else:
        return None


def set_lg_rc_map(block, spaces):
    '''Set local-global maps for matrices in block_mat'''
    assert isinstance(block, block_mat)
    assert len(block) == len(block[0]) == len(spaces)

    comm = extract_comm(block)
    for row, row_space in zip(block, spaces):
        for mat, col_space in zip(row, spaces):
            if isinstance(mat, PETScMatrix):
                row_lgmap = row_space.dofmap().tabulate_local_to_global_dofs()
                row_lgmap = PETSc.LGMap().create(map(int, row_lgmap), comm=comm)

                col_lgmap = col_space.dofmap().tabulate_local_to_global_dofs()
                col_lgmap = PETSc.LGMap().create(map(int, col_lgmap), comm=comm)

                as_backend_type(mat).mat().setLGMap(row_lgmap, col_lgmap)
    return block

# ----------------------------------------------------------------------------

if __name__ == '__main__':
    from dolfin import *

    mesh = UnitSquareMesh(32, 32)

    V = FunctionSpace(mesh, 'BDM', 1)  
    Q = FunctionSpace(mesh, 'DG', 0)   
    W = [V, Q]

    sigma, u = map(TrialFunction, W)
    tau, v = map(TestFunction, W)

    a00 = inner(sigma, tau)*dx
    a01 = inner(u, div(tau))*dx
    a10 = inner(div(sigma), v)*dx
    L0 = inner(Constant((2, 2)), tau)*dx
    L1 = inner(-Constant(1), v)*dx

    A00 = assemble(a00)
    A01 = assemble(a01)
    A10 = assemble(a10)
    b0 = assemble(L0)
    b1 = assemble(L1)

    AA = block_mat([[A00, A01], [A10, 0]])
    bb = block_vec([b0, b1])

    from direct import scipy_solve, dolfin_solve

    (u0, p0) = scipy_solve(AA, bb, W)
    (u1, p1) = dolfin_solve(AA, bb, 'mumps', W)

    u0.vector().axpy(-1, u1.vector()); print u0.vector().norm('linf')
    p0.vector().axpy(-1, p1.vector()); print p0.vector().norm('linf')
