from block import block_mat
import numpy as np


def block_diag_mat(diagonal):
    '''A block diagonal matrix'''
    blocks = np.zeros((len(diagonal), )*2, dtype='object')
    for i, A in enumerate(diagonal):
        blocks[i, i] = A
        
    return block_mat(blocks)
