# What is the fast way to apply bcs
from block import block_bc, block_assemble
from dolfin import *

from xii.linalg.convert import convert 
from xii.linalg.convert import numpy_to_petsc
from scipy.sparse import csr_matrix
from block import block_mat, block_vec
from block.block_bc import block_rhs_bc
from petsc4py import PETSc
import numpy as np


def apply_bc(A, b, bcs, diag_val=1.):
    '''
    Apply block boundary conditions to block system A, b    
    '''
    if not any(bcs): return A, b
    
    # Allow for A, b be simple matrices. To proceed we wrap them as
    # block objects
    has_wrapped_A = False
    if not isinstance(A, block_mat):
        A = block_mat([[A]])
        has_wrapped_A = True
    
    if not isinstance(b, block_vec):
        assert has_wrapped_A
        b = block_vec([b])
        bcs = [bcs]

    # block boundary conditions is a list with bcs for each block of A, b
    assert len(A) == len(b) == len(bcs)

    bcs_ = []
    for bc in bcs:
        assert isinstance(bc, (DirichletBC, list))
        if isinstance(bc, DirichletBC):
            bcs_.append([bc])
        else:
            bcs_.append(bc)
    bcs = bcs_

    if not bcs or not any(bcs): return A, b

    # Obtain a monolithic matrix
    AA, bb = map(convert, (A, b))
    # PETSc guys
    AA, bb = as_backend_type(AA).mat(), as_backend_type(bb).vec()

    # We want to make a monotlithic matrix corresponding to function sp ace
    # where the spaces are serialized
    
    # Break apart for block
    offsets = [0]
    for bi in b:
        offsets.append(offsets[-1] + bi.size())

    rows = []
    x_values = []
    # Each bc in the group has dofs numbered only wrt to its space.
    # We're after global numbering
    for shift, bcs_sub in zip(offsets, bcs): 
        for bc in bcs_sub:
            # NOTE: bcs can be a dict or DirichletBC in which case we extract
            # the dict
            if isinstance(bc, DirichletBC): bc = bc.get_boundary_values()
            # Dofs and values for rhs
            rows.extend(shift + np.array(bc.keys(), dtype='int32'))
            x_values.extend(bc.values())
            
    rows = np.hstack(rows)
    x_values = np.array(x_values)

    x = bb.copy()
    x.zeroEntries()
    x.setValues(rows, x_values)

    # Apply to monolithic
    len(rows) and AA.zeroRowsColumns(rows, diag=diag_val, x=x, b=bb)

    blocks = []
    for first, last in zip(offsets[:-1], offsets[1:]):
        blocks.append(PETSc.IS().createStride(last-first, first, 1))

    # Reasamble
    comm = mpi_comm_world()
    b = [PETSc.Vec().createWithArray(bb.getValues(block), comm=comm) for block in blocks]
    for bi in b:
        bi.assemblyBegin()
        bi.assemblyEnd()
    b = block_vec(map(PETScVector, b))

    # PETSc 3.7.x
    try:
        AA.getSubMatrix
        A = [[PETScMatrix(AA.getSubMatrix(block_row, block_col)) for block_col in blocks]
             for block_row in blocks]
    # NOTE: 3.8+ does not ahve getSubMatrix, there is getLocalSubMatrix
    # but cannot get that to work - petsc error 73. So for now everything
    # is done with scipy    
    except AttributeError:
        AA = csr_matrix(AA.getValuesCSR()[::-1], shape=AA.size)

        A = [[numpy_to_petsc(AA[block_row.array, :][:, block_col.array]) for block_col in blocks]
             for block_row in blocks]
    # Block mat
    A = block_mat(A)

    if has_wrapped_A: return A[0][0], b[0]
    
    return A, b


def identity(ncells):
    mesh = UnitSquareMesh(ncells, ncells)

    V = VectorFunctionSpace(mesh, 'CG', 2)
    Q = FunctionSpace(mesh, 'CG', 1)
    W = [V, Q]

    u, p = map(TrialFunction, W)
    v, q = map(TestFunction, W)

    a = [[0]*len(W) for _ in range(len(W))]
    a[0][0] = inner(grad(u), grad(v))*dx
    a[0][1] = inner(p, div(v))*dx
    a[1][0] = inner(q, div(u))*dx

    x, y = SpatialCoordinate(mesh)
    L = [inner(as_vector((y*x**2, sin(pi*(x+y)))), v)*dx,
         inner(x+y, q)*dx]

    A0 = block_assemble(a)
    b0 = block_assemble(L)

    # First methog
    bcs = [[], []]

    A, b = apply_bc(A0, b0, bcs)

    b0_ = np.hstack([bi.get_local() for bi in b0])
    b_ = np.hstack([bi.get_local() for bi in b])

    eb = (b - b0).norm()

    for i in range(len(W)):
        for j in range(len(W)):
            Aij = A[i][j]  # Always matrix
            
            x = Vector(mpi_comm_world(), Aij.size(1))
            x.set_local(np.random.rand(x.local_size()))

            y = Aij*x
            y0 = A0[i][j]*x

            print i, j, '>>>', (y - y0).norm('linf')
    return eb

    

def speed(ncells):
    mesh = UnitSquareMesh(ncells, ncells)

    V = VectorFunctionSpace(mesh, 'CG', 2)
    Q = FunctionSpace(mesh, 'CG', 1)
    W = [V, Q]

    u, p = map(TrialFunction, W)
    v, q = map(TestFunction, W)

    a = [[0]*len(W) for _ in range(len(W))]
    a[0][0] = inner(grad(u), grad(v))*dx
    a[0][1] = inner(p, div(v))*dx
    a[1][0] = inner(q, div(u))*dx

    x, y = SpatialCoordinate(mesh)
    L = [inner(as_vector((y*x**2, sin(pi*(x+y)))), v)*dx,
         inner(x+y, q)*dx]

    A0 = block_assemble(a)
    b0 = block_assemble(L)

    # First method
    bc = DirichletBC(V, Constant((0, 0)), 'on_boundary')
    bcs = [[bc], []]
    t = Timer('first')
    block_bc(bcs, True).apply(A0).apply(b0)
    dt0 = t.stop()

    dimW = sum(Wi.dim() for Wi in W)

    A, b = map(block_assemble, (a, L))
    
    t = Timer('second')
    A, b = apply_bc(A, b, bcs)
    dt1 = t.stop()

    print '>>>', (b - b0).norm()

    # First method
    A, c = map(block_assemble, (a, L))    
    block_rhs_bc(bcs, A).apply(c)

    print '>>>', (b - c).norm()

    return dimW, dt0, dt1, dt0/dt1

# --------------------------------------------------------------------

if __name__ == '__main__':

    # Check that we are extracting block right
    # identity(ncells)

    # Now speed
    msg = 'dim(W) = %d, block = %g s, petsc = %g s, speedup %.2f'
    for ncells in [4, 8, 16, 32, 64, 128]:
        print msg % speed(ncells) 




