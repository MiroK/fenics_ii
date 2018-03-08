from xii.linalg.convert import bmat_sizes
from xii.linalg.function import as_petsc_nest
from block import block_mat, block_vec
from dolfin import PETScVector, as_backend_type
from petsc4py import PETSc
import numpy as np


def block_diag_mat(diagonal):
    '''A block diagonal matrix'''
    blocks = np.zeros((len(diagonal), )*2, dtype='object')
    for i, A in enumerate(diagonal):
        blocks[i, i] = A
        
    return block_mat(blocks)


def ii_PETScOperator(bmat):
    '''Return an object with mult method which acts like bmat*'''
    assert isinstance(bmat, block_mat)

    row_sizes, col_sizes = bmat_sizes(bmat)
    
    class Foo(object):
        def __init__(self, A):
            self.A = A
    
        def mult(self, mat, x, y):
            '''y = A*x'''
            y *= 0
            # Now x shall be comming as a nested vector
            # Convert
            x_bvec = block_vec(map(PETScVector, x.getNestSubVecs()))
            # Apply
            y_bvec = self.A*x_bvec
            # Convert back
            y.axpy(1., as_petsc_nest(y_bvec))
    
    mat = PETSc.Mat().createPython([[sum(row_sizes), ]*2, [sum(col_sizes), ]*2])
    mat.setPythonContext(Foo(bmat))
    mat.setUp()

    return mat


def ii_PETScPreconditioner(bmat, ksp):
    '''Create from bmat a preconditioner for KSP'''
    assert isinstance(bmat, block_mat)

    class Foo(object):
        def __init__(self, A):
            self.A = A
    
        def apply(self, mat, x, y):
            '''y = A*x'''
            # Now x shall be comming as a nested vector
            y *= 0
            # Now x shall be comming as a nested vector
            # Convert
            x_bvec = block_vec(map(PETScVector, x.getNestSubVecs()))
            # Apply
            y_bvec = self.A*x_bvec
            # Convert back
            y.axpy(1., as_petsc_nest(y_bvec))

    pc = ksp.pc
    pc.setType(PETSc.PC.Type.PYTHON)
    pc.setPythonContext(Foo(bmat))
    pc.setUp()

    return pc
