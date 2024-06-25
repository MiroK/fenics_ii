from petsc4py import PETSc
from dolfin import PETScMatrix
import numpy as np
from block.block_base import block_base


class OuterProduct(block_base):
    '''Low rank operator'''
    def __init__(self, u, v=None):
        if v is None:
            v = u
        self._u = u
        self._v = v
        self._matrix_ = None

    def create_vec(self, dim):
        return (self._u if dim == 0 else self._v).copy()

    def matvec(self, x):
        alpha = self._v.inner(x)
        return alpha*self._u

    def collapse(self):
        if self._matrix_ is None:
            Mean = np.outer(self._u.get_local(), self._v.get_local())
            MeanMat = PETSc.Mat().createDense(size=(self._u.local_size(), self._v.local_size()),
                                              array=Mean)
            self._matrix_ = PETScMatrix(MeanMat)
        return self._matrix_
