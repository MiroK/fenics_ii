from xii.linalg.matrix_utils import petsc_serial_matrix, is_number
from xii.assembler.mean_form import mean_cell, mean_space

from numpy.polynomial.legendre import leggauss
from dolfin import PETScMatrix, cells, Point, Cell, Function
from collections import defaultdict
import scipy.sparse as sp
from petsc4py import PETSc
import dolfin as df
import numpy as np
import tqdm


def memoize_average(average_mat):
    '''Cached average'''
    cache = {}
    def cached_average_mat(V, TV, reduced_mesh, data):
        key = ((V.ufl_element(), V.mesh().id()),
               (TV.ufl_element(), TV.mesh().id()),
               data['weight'], data['measure'])

        if key not in cache:
            cache[key] = average_mat(V, TV, reduced_mesh, data)
        return cache[key]
    
    return cached_average_mat


@memoize_average
def mean_mat(V, TV, reduced_mesh, data):
    '''
    A mapping for computing the patch averages of function in V in the 
    space TV. 
    '''
    v = df.TestFunction(V)

    measure, weight = data['measure'], data['weight']
    if measure is None:
        measure = df.dx
    if weight is None:
        weight = df.Constant(1)
    
    values = df.assemble(v*weight*measure)
    jj = np.arange(V.dim())
    ii = np.zeros_like(jj)
        
    matrix = sp.csr_matrix((values, (ii, jj)), shape=(TV.dim(), V.dim()))
    mat = PETSc.Mat().createAIJ(comm=PETSc.COMM_WORLD,
                                size=matrix.shape,
                                csr=(matrix.indptr, matrix.indices, matrix.data))
    return df.PETScMatrix(mat)
