import sys, petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc

from xii.linalg.matrix_utils import as_petsc, diagonal_matrix
from xii.linalg.convert import convert
from block.algebraic.petsc import LU, SUPERLU_LU, AMG, Elasticity
from block.algebraic.petsc.precond import precond
from block import block_mat, block_vec, block_mul, block_add
import numpy as np


def pc_mat(pc, block):
    '''Set pc for (non-block) operator'''
    if isinstance(block, LU):
        pc.setType('lu')
        pc.setFactorPivot(1E-18)
        return block.A
            
    if isinstance(block, SUPERLU_LU):
        pc.setType('lu')
        pc.setFactorPivot(1E-18)            
        pc.setFactorSolverType('superlu')
        return block.A

    if isinstance(block, AMG):
        pc.setType('hypre')
        return block.A

    if isinstance(block, Elasticity):
        pc.setType('gamg')
        return block.A

    # FIXME: Very add hoc support for sum, this should recursive
    if isinstance(block, block_add):
        this, that = block.A, block.B
        assert isinstance(this, precond) and isinstance(that, precond)
            
        pc.setType('composite')
        pc.setCompositeType(PETSc.PC.CompositeType.ADDITIVE)

        for sub, op in enumerate((this, that)):
            # Fake it 
            pc_ = PETSc.PC().create()
            A = pc_mat(pc_, op)

            pc.addCompositePC(pc_.getType())

            pc_sub = pc.getCompositePC(sub)            
            # Make it
            pc_mat(pc_sub, op)
            pc_sub.setOperators(as_petsc(A))

        mat = diagonal_matrix(op.A.size(0), 1)

        return mat

    assert False, type(block)


def pc_nest(pc, block_pc, Amat):
    '''Setup field split and its operators'''
    isets, jsets = Amat.getNestISs()
    # Sanity
    assert len(isets) == len(jsets)
    assert all(i.equal(j) for i, j in zip(isets, jsets)), [(i.array[[0, -1]], j.array[[0, -1]]) for i, j in zip(isets, jsets)]
    
    # We target block matrices
    if len(isets) == 1:
        A = pc_mat(pc, block_pc[0][0])
        return as_petsc(A)

    grouped_isets = []
    # It may be that block_pc requires different grouping if indinces.
    # This is signaled by presence of RestrictionOperator
    if isinstance(block_pc, block_mul):
        Rt, block_pc, R = block_pc.chain
        assert Rt.A == R

        offsets = R.offsets

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
        # Set individual preconds for blocks
        block = block_pc[i][i]
        mat = pc_mat(pci, block)

        Bmat[i][i] = mat
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
