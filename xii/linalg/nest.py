import sys, petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc

from xii.linalg.matrix_utils import as_petsc
from xii.linalg.convert import convert
from block.algebraic.petsc import LU, SUPERLU_LU, AMG, Elasticity
from block.algebraic.petsc.precond import precond
from block import block_mat, block_vec, block_mul
import numpy as np


def pc_nest(pc, block_pc, Amat):
    '''Setup field split and its operators'''
    isets, jsets = Amat.getNestISs()
    # Sanity
    assert len(isets) == len(jsets)
    assert all(i.equal(j) for i, j in zip(isets, jsets)), [(i.array[[0, -1]], j.array[[0, -1]]) for i, j in zip(isets, jsets)]
    
    # We target block matrices
    assert len(isets) > 1

    grouped_isets = []
    # It may be that block_pc requires different grouping if indinces.
    # This is signaled by presence of RestrictionOperator
    if isinstance(block_pc, block_mul):
        Rt, block_pc, R = block_pc.chain
        assert Rt.A == R

        offsets = R.offsets
        assert len(offsets) == len(isets)

        for first, last in zip(R.offsets[:-1], R.offsets[1:]):
            if last - first == 1:
                grouped_isets.append(isets[first])
            else:
                group = isets[first]
                for i in range(first+1, last):
                    group = group.union(isets[i])
                grouped_isets.append(group)
    else:
        grouped_isets = isets

    assert not isinstance(block_pc, block_mat) or set(block_pc.blocks.shape) == set((len(grouped_isets), ))

    pc.setFieldSplitIS(*[(str(i), iset) for i, iset in enumerate(grouped_isets)])
    
    # The preconditioner will always be fieldsplit
    assert pc.getType() == 'fieldsplit'
    
    pc_ksps = pc.getFieldSplitSubKSP()
    
    nblocks, = set(block_pc.blocks.shape)

    # For now we can only do pure block_mats
    assert isinstance(block_pc, block_mat)
    # Consistency        
    assert len(grouped_isets) == nblocks
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

        elif isinstance(block, AMG):
            pci.setType('hypre')

        elif isinstance(block, Elasticity):
            pci.setType('gamg')
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
