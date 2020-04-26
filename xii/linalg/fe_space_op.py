from __future__ import absolute_import
from __future__ import print_function
from block.block_base import block_base
from block.object_pool import vec_pool
import scipy.sparse as sparse
from petsc4py import PETSc
import dolfin as df
from six.moves import range


class FESpaceOperator(block_base):
    '''Operator mapping V1 -> V0'''
    def __init__(self, V1, V0=None):
        self.V1 = V1
        self.V0 = V1 if V0 is None else V0

    @vec_pool
    def create_vec(self, dim=1):
        '''Create vector in space'''
        V = [self.V0, self.V1][dim]
        # FIXME: for now ignore [df.FunctionSpace]
        assert isinstance(V, df.FunctionSpace)
        return df.Vector(V.mesh().mpi_comm(), V.dim())

    def collapse(self):
        '''Return a matrix representation'''
        # Check cache
        if hasattr(self, 'matrix_repr'): return self.matrix_repr

        # Otherwise compute it, but only once
        x = self.create_vec(dim=1)
        x_values = x.get_local()

        columns = []
        df.info('Collapsing to %d x %d matrix' % (self.V0.dim(), self.V1.dim()))
        timer = df.Timer('Collapse')
        for i in range(self.V1.dim()):
            # Basis of row space
            x_values[i] = 1.
            x.set_local(x_values)

            column = sparse.csc_matrix((self*x).get_local())
            columns.append(column)
            # Reset
            x_values[i] = 0.
        # Alltogether
        mat = (sparse.vstack(columns).T).tocsr()
        df.info('\tDone in %g' % timer.stop())
        
        # As PETSc.Mat
        A = PETSc.Mat().createAIJ(comm=PETSc.COMM_WORLD,
                                  size=mat.shape,
                                  csr=(mat.indptr, mat.indices, mat.data))
        # Finally for dolfin
        self.matrix_repr = df.PETScMatrix(A)

        return self.matrix_repr

    def matvec(self, b):
        '''Action on vector from V1'''
        raise NotImplementedError

    def transpmult(self, b):
        '''Action on vector from V0'''
        raise NotImplementedError

# --------------------------------------------------------------------    

if __name__ == '__main__':
    import numpy as np
    
    mesh = df.UnitSquareMesh(32, 32)
    V = df.FunctionSpace(mesh, 'CG', 1)
    u, v = df.TrialFunction(V), df.TestFunction(V)

    a = df.inner(u, df.grad(v)[0])*df.dx
    A = df.assemble(a)

    class Foo(FESpaceOperator):
        def __init__(self, V=V, A=A):
            self.A = A
            
            FESpaceOperator.__init__(self, V1=V)

        def matvec(self, b):
            return self.A*b

    foo = Foo()

    x = df.interpolate(df.Constant(0), V).vector()
    x.set_local(np.r_[1, np.zeros(V.dim()-1)])
    
    A_ = foo.collapse()

    print(np.linalg.norm(A.array() - A_.array()))

