from block.block_base import block_base
from block.object_pool import vec_pool
from block import block_transpose

from xii.linalg.convert import bmat_sizes, get_dims
from xii.linalg.function import as_petsc_nest
from xii.linalg.matrix_utils import as_petsc

from block import block_mat, block_vec
from dolfin import (PETScVector, as_backend_type, Function, Vector, GenericVector,
                    mpi_comm_world, Matrix, PETScMatrix)
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

    if isinstance(bmat, block_base):
        row_sizes, col_sizes = bmat_sizes(bmat)
        is_block = True
    else:
        row_sizes, col_sizes = (bmat.size(0), ), (bmat.size(1), )
        is_block = False

    class Foo(object):
        def __init__(self, A):
            self.A = A

        if is_block:
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
                '''y = A.T*x'''
                AT = block_transpose(self.A)
            
                y *= 0
                # Now x shall be comming as a nested vector
                # Convert
                x_bvec = block_vec(map(PETScVector, x.getNestSubVecs()))
                # Apply
                y_bvec = AT*x_bvec
                # Convert back
                y.axpy(1., as_petsc_nest(y_bvec))
        # No block
        else:
            def mult(self, mat, x, y):
                '''y = A*x'''
                y *= 0
                x_bvec = PETScVector(x)
                y_bvec = self.A*x_bvec
                y.axpy(1., as_petsc(y_bvec))

            def multTranspose(self, mat, x, y):
                '''y = A.T*x'''
                AT = block_transpose(self.A)
            
                y *= 0
                x_bvec = PETScVector(x)
                y_bvec = AT*x_bvec
                y.axpy(1., as_petsc(y_bvec))

    mat = PETSc.Mat().createPython([[sum(row_sizes), ]*2, [sum(col_sizes), ]*2])
    mat.setPythonContext(Foo(bmat))
    mat.setUp()

    return mat


def ii_PETScPreconditioner(bmat, ksp):
    '''Create from bmat a preconditioner for KSP'''
    # try:
    #     row_sizes, col_sizes = bmat_sizes(bmat)
    #     is_block = True
    # except ValueError:
    #     nrows, ncols = get_dims(bmat)
    #     row_sizes, col_sizes = (nrows, ), (ncols, )
    #     is_block = False
    is_block = not isinstance(bmat, (Matrix, PETScMatrix))
    # NOTE: we assume that this is a symmetric operator
    class Foo(object):
        def __init__(self, A):
            self.A = A

        if is_block:
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
                '''y = A.T*x'''
                AT = block_transpose(self.A)
                
                # Now x shall be comming as a nested vector
                y *= 0
                # Now x shall be comming as a nested vector
                # Convert
                x_bvec = block_vec(map(PETScVector, x.getNestSubVecs()))
                # Apply
                y_bvec = AT*x_bvec
                # Convert back
                y.axpy(1., as_petsc_nest(y_bvec))
        # no block; FIXME: would be so much simpler with type dispatch
        # Ie block_vec knows what to do + wrap x.getNestSubVecs
        else:
            def apply(self, mat, x, y):
                '''y = A*x'''
                y *= 0
                x_bvec = PETScVector(x)
                y_bvec = self.A*x_bvec
                y.axpy(1., as_petsc(y_bvec))

            def applyTranspose(self, mat, x, y):
                '''y = A.T*x'''
                AT = block_transpose(self.A)
                
                y *= 0
                x_bvec = PETScVector(x)
                y_bvec = AT*x_bvec
                y.axpy(1., as_petsc(y_bvec))

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
            size,  = set(Wi.dim() for Wi in W)

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

            elm,  = set(W.ufl_element().sub_elements())
            
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

    
def is_increasing(seq):
    if not seq:
        return False
    if len(seq) == 1:
        return True
    return seq[0] < seq[1] and is_increasing(seq[1:])


class ReductionOperator(block_base):
    '''
    This operator reduces block vector into a block vector with 
    at most the same number of blocks. The size of the blocks is specified
    by offsets. Eg (1, 2, 3, 4) is the idenity for vector with 4 block
    [with components [0:1], [1:2], [2:3], [3:4]], whole (2, 3, 4) would 
    produce a 3 vector from [0:2], [2:3], [3:4]
    '''
    def __init__(self, offsets, W):
        assert len(W) == offsets[-1]
        assert is_increasing(offsets)
        self.offsets = [0] + offsets

        self.index_sets = []
        for f, l in zip(self.offsets[:-1], self.offsets[1:]):
            if (l - f) == 1:
                self.index_sets.append([])
            else:
                index_set = []
                prev = 0
                for Wi in W[f:l]:
                    this = Wi.dofmap().dofs() + prev
                    index_set.append(PETSc.IS().createGeneral(this.tolist()))
                    prev = len(this)
                self.index_sets.append(index_set)
        # Handle get_dims
        self.__sizes__ = (sum(Wi.dim() for Wi in W), )*2

    def matvec(self, b):
        '''Reduce'''
        # print type(b), b.size(), self.offsets
        # assert b.size() == self.offsets[-1]

        reduced = []
        for f, l in zip(self.offsets[:-1], self.offsets[1:]):
            if (l - f) == 1:
                reduced.append(b[f])
            else:
                reduced.append(PETScVector(as_petsc_nest(block_vec(b.blocks[f:l]))))
        return block_vec(reduced) if len(reduced) > 1 else reduced[0]

    def transpmult(self, b):
        '''Unpack'''
        if isinstance(b, (Vector, GenericVector)):
            b = [b]
        else:
            b = b.blocks
        n = len(b)
        assert n == len(self.index_sets), self.index_sets

        unpacked = []
        for bi, iset in zip(b, self.index_sets):
            if len(iset) == 0:
                unpacked.append(bi)
            else:
                x_petsc = as_backend_type(bi).vec()

                subvecs = map(lambda indices, x=x_petsc: PETScVector(x.getSubVector(indices)),
                              iset)

                unpacked.extend(subvecs)
        return block_vec(unpacked)

    
class RegroupOperator(block_base):
    '''Block vec to Block vec of block_vec/vecs.'''
    def __init__(self, offsets, W):
        assert len(W) == offsets[-1]
        assert is_increasing(offsets)
        self.offsets = [0] + offsets

    def matvec(self, b):
        '''Reduce'''

        reduced = []
        for f, l in zip(self.offsets[:-1], self.offsets[1:]):
            if (l - f) == 1:
                reduced.append(b[f])
            else:
                reduced.append(block_vec(b.blocks[f:l]))
        return block_vec(reduced) if len(reduced) > 1 else reduced[0]

    def transpmult(self, b):
        '''Unpack'''
        b_block = []
        for bi in b:
            if isinstance(bi, (Vector, GenericVector)):
                b_block.append(bi)
            else:
                b_block.extend(bi.blocks)
                
        return block_vec(b_block)

    
class BlockPC(block_base):
    '''Wrap petsc preconditioner for cbc.block'''
    def __init__(self, pc):
        self.pc = pc
        self.A = pc.getOperators()[0]

    def matvec(self, x):
        y = self.create_vec(0)
        self.pc.apply(as_petsc(x), as_petsc(y))
        return y
        
    @vec_pool
    def create_vec(self, dim):
        if dim == 0:
            vec = self.A.createVecLeft()
        else:
            vec = self.A.createVecRight()
        return PETScVector(vec)

# --------------------------------------------------------------------

if __name__ == '__main__':
    from dolfin import *
    from block import block_transpose
    from xii import ii_convert, ii_assemble
    import numpy as np

    mesh = UnitSquareMesh(16, 16)
    
    V = VectorFunctionSpace(mesh, 'CG', 1)
    Vb = VectorFunctionSpace(mesh, 'Bubble', 3)

    u, v = TrialFunction(V), TestFunction(V)
    ub, vb = TrialFunction(Vb), TestFunction(Vb)

    b = [[0, 0], [0, 0]]
    b[0][0] = inner(grad(u), grad(v))*dx + inner(u, v)*dx
    b[0][1] = inner(grad(ub), grad(v))*dx + inner(ub, v)*dx
    b[1][0] = inner(grad(u), grad(vb))*dx + inner(u, vb)*dx
    b[1][1] = inner(grad(ub), grad(vb))*dx + inner(ub, vb)*dx

    BB = ii_assemble(b)
    
    x = Function(V).vector(); x.set_local(np.random.rand(x.local_size()))
    y = Function(Vb).vector(); y.set_local(np.random.rand(y.local_size()))
    bb = block_vec([x, y])

    z_block = BB*bb
    
    # Make into a monolithic matrix
    BB_m = ii_convert(BB)
    
    R = ReductionOperator([2], W=[V, Vb])

    z = (R.T)*BB_m*(R*bb)

    print (z - z_block).norm()

    y  = BB_m*(R*bb)
    print np.linalg.norm(np.hstack([bi.get_local() for bi in z_block])-y.get_local())
