from __future__ import print_function
from xii.linalg.convert import bmat_sizes, convert
from dolfin import PETScVector, mpi_comm_world
import block.iterative as cbc
import scipy.sparse.linalg as sp
from block import block_vec
import numpy as np
import pyamg


class ScipyLinOp(sp.LinearOperator):
    def __init__(self, bmat):
        self.bmat = bmat
        self.dtype = np.dtype(float)

        row_sizes, col_sizes = bmat_sizes(bmat)            
        
        self.shape = (sum(row_sizes), sum(col_sizes))
        self.x = block_vec([PETScVector(mpi_comm_world(), n) for n in row_sizes])
            
    def _matvec(self, xarr):
        # Fill
        first = 0
        for xi in self.x:
            size = xi.local_size()
            xi.set_local(xarr[first:first + size])
            first = first + size
        # Apply and wrap
        y = convert(self.bmat*self.x)
            
        return y.get_local()


def pyamg_solve(method, AA, bb, tol, M, x0, maxiter=None, **kwargs):
    '''Solve AA*x = bb (block-system) using M-Krylov method until tolerance'''
    # Asuming block mat
    AA_ = ScipyLinOp(AA)

    if M is not None:
        M_ = ScipyLinOp(M)
    else:
        M_ = None
        
    # Asuming block vec
    bb_ = convert(bb).get_local()
    x0_ = convert(x0).get_local()

    krylov_solve = getattr(pyamg.krylov, method)

    residuals = []
    callback = lambda x, res=residuals: print('{} residual norm {}'.format(len(res), res[-1]))
    xarr, status = krylov_solve(A=AA_, b=bb_, M=M_, x0=x0_, tol=tol,
                                maxiter=maxiter, callback=callback, residuals=residuals,
                                **kwargs)

    assert status == 0, status

    # Fill back
    first = 0
    for xi in x0:
        size = xi.local_size()
        xi.set_local(xarr[first:first+size])
        first = first + size

    return x0, residuals


def scipy_solve(method, AA, bb, tol, M, x0, maxiter=None, **kwargs):
    '''Solve AA*x = bb (block-system) using M-Krylov method until tolerance'''
    # Asuming block mat
    AA_ = ScipyLinOp(AA)

    if M is not None:
        M_ = ScipyLinOp(M)
    else:
        M_ = None
        
    # Asuming block vec
    bb_ = convert(bb).get_local()
    x0_ = convert(x0).get_local()

    krylov_solve = getattr(sp, method)

    residuals = []
    callback = lambda x, res=residuals: (residuals.append(np.linalg.norm(x)),
                                         print('{} residual norm {}'.format(len(res), res[-1])))
    
    xarr, status = krylov_solve(A=AA_, b=bb_, M=M_, x0=x0_, tol=tol,
                                maxiter=maxiter, callback=callback,
                                **kwargs)

    assert status == 0, status

    # Fill back
    first = 0
    for xi in x0:
        size = xi.local_size()
        xi.set_local(xarr[first:first+size])
        first = first + size

    return x0, residuals


def cbc_solve(method, AA, bb, tol, M, x0, maxiter=None, **kwargs):
    '''Solve AA*x = bb (block-system) using M-Krylov method until tolerance'''
    if M is None:
        M=1

    krylov_solve = {'lgmres': cbc.LGMRES,
                    'cg': cbc.ConjGrad,
                    'bicg': cbc.BiCGStab,
                    'cgn': cbc.CGN,
                    'symmlq': cbc.SymmLQ,
                    'tfqmr': cbc.TFQMR,
                    'minres': cbc.SubMinRes}[method]

    AAinv = krylov_solve(A=AA,
                         precond=M,
                         tolerance=tol,
                         initial_guess=x0,
                         maxiter=maxiter,
                         show=3,
                         relativeconv=True)
    residuals = []

    x0 = AAinv*bb

    return x0, AAinv.residuals
