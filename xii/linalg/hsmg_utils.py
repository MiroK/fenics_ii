from hsmg.hseig import InterpolationMatrix
from block.block_compose import block_mul, block_add, block_sub

from xii.linalg.matrix_utils import is_petsc_mat, is_number
from xii.linalg.convert import numpy_to_petsc

from scipy.linalg import eigh
import numpy as np


def inverse(bmat):
    '''
    Inverse of a linear combination of Hs norms. We very strictly 
    enforce the form of sum_j alpha_j H^{s_j}
    '''
    if isinstance(bmat, InterpolationMatrix):
        return bmat**-1
    # Does it satisfy the definittion
    assert is_well_defined(bmat)
    # Try to see if sombody computes the eigenvalues
    lmbda, U = extract_attributes(bmat, ('lmbda', 'U'))
    # Do it your self
    if U is None or lmbda is None:
        A, M = extract_attributes(bmat, ('A', 'M'))
    lmbda, U = eigh(A.array(), M.array())

    diagonal = np.zeros_like(lmbda)
    for alpha, s in collect(bmat):
        diagonal[:] += alpha*(lmbda**s)
    # Invert
    diagonal[:] = 1./diagonal
    
    array = U.dot(np.diag(diagonal).dot(U.T))

    return numpy_to_petsc(array)


def crawl(bmat):
    '''Return terminals in cbc.block expression that are relevant for inverse'''
    if isinstance(bmat, block_mul):
        for block in bmat.chain:
            for item in crawl(block):
                yield item

    if isinstance(bmat, (block_add, block_sub)):
        for item in crawl(bmat.A):
            yield item

        for item in crawl(bmat.B):
            yield item

    if is_number(bmat) or isinstance(bmat, InterpolationMatrix):
        yield bmat

        
def is_well_defined(bmat):
    '''Is expression of the form that inverse accepts'''
    A, M = None, None
    for mat in crawl(bmat):
        if is_number(mat): continue
        
        if A is None:
            A, M = mat.A, mat.M
        else:
            if not ((A.size(0) == mat.A.size(0)) and (A.size(1) == mat.A.size(1))): 
                return False
            if not ((M.size(0) == mat.M.size(0)) and (M.size(1) == mat.M.size(1))): 
                return False
    return True


def collect(bmat):
    '''Extract aplha_j, s_j pairs'''
    if isinstance(bmat, InterpolationMatrix):
        return [[1, bmat.s]]

    if isinstance(bmat, block_add):
        return collect(bmat.A) + collect(bmat.B)

    if isinstance(bmat, block_sub):
        b = collect(bmat.B)
        return collect(bmat.A) + [[-bi[0], bi[1]] for bi in b]

    if isinstance(bmat, block_mul):
        assert len(bmat.chain) == 2
        A, B = bmat.chain
        
        if isinstance(A, InterpolationMatrix):
            assert is_number(B)
            return [[B, A.s]]
        else:
            assert is_number(A) and isinstance(B, InterpolationMatrix), (A, B)
            return [[A, B.s]]

        
def extract_attributes(bmat, attrs):
    '''Crawl the expression for attributes'''
    if isinstance(attrs, str):

        for item in crawl(bmat):
            found = getattr(item, attrs, None)
            if found is not None: return [found]
            
        return [None]
    else:
        return sum((extract_attributes(bmat, attr) for attr in attrs), [])
    
# --------------------------------------------------------------------

if __name__ == '__main__':
    from dolfin import *

    mesh = UnitSquareMesh(10, 10)
    V = FunctionSpace(mesh, 'CG', 1)

    u, v = TrialFunction(V), TestFunction(V)
    a = inner(grad(u), grad(v))*dx
    m = inner(u, v)*dx

    A, M = assemble(a+m), assemble(m)
    I = InterpolationMatrix(A, M, 0.5)
    J = InterpolationMatrix(A, M, 0.25)

    print (inverse(0.1234*I)).array() - ((1./0.1234)*((I**-1).array()))
