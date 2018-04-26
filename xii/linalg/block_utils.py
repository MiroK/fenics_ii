from block.block_base import block_base
from block.object_pool import vec_pool
from xii.linalg.convert import bmat_sizes, get_dims
from xii.linalg.function import as_petsc_nest
from block import block_mat, block_vec
from dolfin import PETScVector, as_backend_type, Function
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
    # NOTE: we assume that this is a symmetric operator
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

        def multTranspose(self, mat, x, y):
            self.mult(mat, x, y)
    
    mat = PETSc.Mat().createPython([[sum(row_sizes), ]*2, [sum(col_sizes), ]*2])
    mat.setPythonContext(Foo(bmat))
    mat.setUp()

    return mat


def ii_PETScPreconditioner(bmat, ksp):
    '''Create from bmat a preconditioner for KSP'''
    assert isinstance(bmat, block_mat)
    # NOTE: we assume that this is a symmetric operator
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

        def applyTranspose(self, mat, x, y):
            self.apply(mat, x, y)

    pc = ksp.pc
    pc.setType(PETSc.PC.Type.PYTHON)
    pc.setPythonContext(Foo(bmat))
    pc.setUp()

    return pc


class VectorizedOperator(block_base):
    '''
    Suppose W is V x V x ... x V and bmat is an operator on V. Here we 
    represent an action of a diagonal operator where each diagonal block 
    is bmat.
    '''
    def __init__(self, bmat, W):
        # Really V X V ... nicely serialized
        self.bmat = bmat
        self.W = W
        
        if isinstance(W, list):
            # All V
            size = W[0].dim()
            assert all(size == Wj.dim() for Wj in W)

            # Matches A
            n, n = get_dims(bmat)
            assert n == m == size

            A = block_diag_mat([bmat]*len(W))

            self.__matvec__ = lambda self, b, A=A: A*b

            self.__create_vec__ = lambda dim, A=A: A.create_vec(dim)
        # Otherwise there is some more work with extracting
        else:
            # All V
            nsubs = W.num_sub_spaces()
            assert nsubs > 0

            elm = W.sub(0).ufl_element()
            assert all(elm == W.sub(j).ufl_element() for j in range(nsubs))
            
            # Matches
            n, m = get_dims(bmat)
            assert n == m == (W.dim()/nsubs)

            # Will need index sets for extracting components in Vj
            index_set = [PETSc.IS().createGeneral(np.fromiter(W.sub(j).dofmap().dofs(), dtype='int32'))
                         for j in range(nsubs)]
            # Prealoc one working array
            work = Function(W).vector().get_local()
            
            def __matvec__(self, b, A=bmat, indices=index_set, work=work):
                for index in indices:
                    # Exact apply assign
                    bj = PETScVector(as_backend_type(b).vec().getSubVector(index))
                    xj = A*bj
                    work[index] = xj.get_local()

                x = self.create_vec()
                x.set_local(work)
                return x
            self.__matvec__ = __matvec__
                
            self.__create_vec__ = lambda dim, W=W: Function(W).vector()
    
    def matvec(self, b):
        return self.__matvec__(self, b)

    @vec_pool
    def create_vec(self, dim):
        return self.__create_vec__(dim)
