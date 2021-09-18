import sys, petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc

from xii.linalg.matrix_utils import as_petsc
from xii.linalg.convert import convert
from block.algebraic.petsc import LU, SUPERLU_LU
from block.algebraic.petsc.precond import precond
from block import block_mat, block_vec
import numpy as np


def pc_nest(block_pc, pc, Amat):
    '''Setup field split and its operators'''
    isets, _ = Amat.getNestISs()
    # We target block matrices
    assert len(isets) > 1
    pc.setFieldSplitIS(*[(str(i), iset) for i, iset in enumerate(isets)])
    
    # The preconditioner will always be fieldsplit
    assert pc.getType() == 'fieldsplit'
    
    pc_ksps = pc.getFieldSplitSubKSP()
    
    nblocks, = set(block_pc.blocks.shape)

    # For now we can only do pure block_mats
    assert isinstance(block_pc, block_mat)
    # Consistency        
    assert len(isets) == nblocks
    # ... and the have to be diagonal
    assert all(i == j or block_pc[i][j] == 0 for i in range(nblocks) for j in range(nblocks))
    
    Bmat = block_mat([[0 for i in range(nblocks)] for j in range(nblocks)])
    for i, pc_ksp in enumerate(pc_ksps):
        pc_ksp.setType('preonly')
        pci = pc_ksp.getPC()

        block = block_pc[i][i]

        if isinstance(block, LU):
            pci.setType('lu')
            pci.setFactorPivot(1E-16)
        elif isinstance(block, SUPERLU_LU):
            pci.setType('lu')
            pci.setFactorPivot(1E-16)            
            pci.setFactorSolverType('superlu')
        else:
            assert False, type(block)

        Bmat[i][i] = block.A
    # Return out for setOperators
    return nest(Bmat)


def nest(tensor, elim_zeros_tol=1E-15, W=None):
    '''Block mat/vec -> PETSc.Nest mat/vec'''
    # Convert block-vector
    if isinstance(tensor, block_vec):
        vecs = [as_petsc(bi) for bi in tensor]
        # FIXME: handle numbers            
        vec = PETSc.Vec().createNest(vecs)
        vec.assemble()
        return vec

    # Convert block-matrix
    if isinstance(tensor, block_mat):
        # Maybe we have preconditioner
        if any(isinstance(block, precond) for block in tensor.blocks.flatten()):
            return 

        nrows, ncols = tensor.blocks.shape
        A = [[None for j in range(ncols)] for i in range(nrows)]
        for i in range(nrows):
            for j in range(ncols):

                if isinstance(tensor[i][j], (int, float)):
                    assert i != j
                    block = None
                else:
                    # Optimize for zeros and insert None                       
                    block = as_petsc(convert(tensor[i][j]))
                    if block.norm(2) < elim_zeros_tol:
                        block = None
                A[i][j] = block

        mat = PETSc.Mat().createNest(A)
        mat.assemble()
        return mat
