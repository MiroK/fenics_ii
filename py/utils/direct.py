from block import block_mat, block_vec
from scipy.sparse.linalg import splu
from convert import block_to_scipy, block_to_dolfin
from dolfin import Function, PETScVector, PETScMatrix, LUSolver, as_backend_type
from petsc4py import PETSc
import numpy as np


def scipy_solve(AA, bb, spaces=None):
    '''Solve AA*x=bb by scipy's direct solver.'''
    # NOTE: serial
    assert isinstance(AA, block_mat)
    assert isinstance(bb, block_vec)
    if spaces is not None: 
        assert len(spaces) == len(bb)
        for b, sub in zip(bb, spaces): assert b.size() == sub.dim()
    else:
        spaces = []

    # Convert
    A = block_to_scipy(AA, blocks_only=False)
    b = block_to_scipy(bb, blocks_only=False)
    # Solve
    Ainv = splu(A)
    x = Ainv.solve(b)
    # Break into pieces
    blocks = np.cumsum([0] + [bi.size() for bi in bb])
    blocks = [x[blocks[i]:blocks[i+1]] for i in range(len(bb))]

    if spaces:
        functions = []  
        for subspace, values in zip(spaces, blocks):
            f = Function(subspace)
            f.vector().set_local(values)
            f.vector().apply('insert')
            functions.append(f)
        return functions
    else:
        return blocks


def dolfin_solve(AA, bb, method='default', spaces=None):
    '''Solve AA*x=bb by dolfin solver.'''
    # NOTE: serial
    assert isinstance(AA, block_mat)
    assert isinstance(bb, block_vec)
    if spaces is not None: 
        assert len(spaces) == len(bb)
        for b, sub in zip(bb, spaces): assert b.size() == sub.dim()
    else:
        spaces = []


    A, b = map(block_to_dolfin, (AA, bb))
    x = b.copy()

    solver = LUSolver(A, method)
    solver.solve(x, b)   # Ax=b

    x = x.array()
    # Break into pieces - this is not very optimal; too much numpy
    blocks = np.cumsum([0] + [bi.size() for bi in bb])
    blocks = [x[blocks[i]:blocks[i+1]] for i in range(len(bb))]

    if spaces:
        functions = []  
        for subspace, values in zip(spaces, blocks):
            f = Function(subspace)
            f.vector().set_local(values)
            f.vector().apply('insert')
            functions.append(f)
        return functions
    else:
        return blocks

# ----------------------------------------------------------------------------

if __name__ == '__main__':
    from dolfin import *

    n = 20
    
    f = Constant(100)
    mesh = UnitSquareMesh(n, n)

    # -------------------------------------------------------

    V = FunctionSpace(mesh, 'BDM', 1)  # sigma
    Q = FunctionSpace(mesh, 'DG', 0)   # u
    W = [V, Q]

    sigma, u = map(TrialFunction, W)
    tau, v = map(TestFunction, W)

    # System
    a00 = inner(sigma, tau)*dx
    a01 = inner(u, div(tau))*dx
    a10 = inner(div(sigma), v)*dx
    L0 = inner(Constant((0, 0)), tau)*dx
    L1 = inner(-f, v)*dx

    A00 = assemble(a00)
    A01 = assemble(a01)
    A10 = assemble(a10)
    b0 = assemble(L0)
    b1 = assemble(L1)

    AA = block_mat([[A00, A01], [A10, 0]])
    bb = block_vec([b0, b1])

    sigma0, u0 = dolfin_solve(AA, bb, method='default', spaces=W)

    # -------------------------------------------------------

    V = FiniteElement('BDM', mesh.ufl_cell(), 1)
    Q = FiniteElement('DG', mesh.ufl_cell(), 0)
    W = MixedElement(V, Q)
    W = FunctionSpace(mesh, W)

    sigma, u = TrialFunctions(W)
    tau, v = TestFunctions(W)

    # System
    a = inner(sigma, tau)*dx+inner(u, div(tau))*dx+inner(div(sigma), v)*dx
    L = inner(Constant((0, 0)), tau)*dx + inner(-f, v)*dx
    
    wh = Function(W)
    solve(a == L, wh)
    sigma1, u1 = wh.split(deepcopy=True)

    print sqrt(abs(assemble(inner(sigma0-sigma1, sigma0-sigma1)*dx)))
    print sqrt(abs(assemble(inner(u0-u1, u0-u1)*dx)))
