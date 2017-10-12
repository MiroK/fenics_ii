from convert import block_to_scipy, block_mat_to_matnest, block_to_dolfin
from dolfin import info, mpi_comm_world, PETScMatrix, SLEPcEigenSolver, Matrix
from dolfin import has_slepc4py, Timer
from scipy.linalg import eigvalsh, eigh
from block import block_mat
from petsc4py import PETSc
import numpy as np


def identity_matrix(size, comm=None):
    '''Dolfin I serial only'''
    comm = mpi_comm_world().tompi4py() if comm is None else comm
    if isinstance(size, int):
        d = PETSc.Vec().createWithArray(np.ones(size))
        I = PETSc.Mat().createAIJ(size=size, nnz=1, comm=comm)
        I.setDiagonal(d)

        return PETScMatrix(I)
    # Blocks for cbc.block
    diag_blocks = map(identity_matrix, size)
    n = len(size)
    I = [[0]*n for _ in range(n)]
    for i in range(n):
        I[i][i] = diag_blocks[i]

    return I


def eig_zero(A_blocks, B_blocks=None, tol=1E-13):
    '''Chase 0 eigenvalues'''
    # We assume here that the non-Matrix block is int
    sizes = []
    for row in A_blocks:
        for col in row:
            if isinstance(col, (Matrix, PETScMatrix)):
                nrows = col.size(0)
                sizes.append(nrows)
                break
    if B_blocks is None: 
        B_blocks = identity_matrix(sizes)
        B_blocks = block_mat(B_blocks)

    size = sum(sizes)
    assert size < 10000
    AA = block_to_scipy(A_blocks)
    BB = block_to_scipy(B_blocks)

    info('Computing eigenvalues A(%s), B(%s)' % (AA.shape, BB.shape))
    t = Timer('gevp')
    eigenvalues, eigenvectors = eigh(AA.toarray(), BB.toarray())

    zeros = np.where(np.abs(eigenvalues) < tol)[0]
    eigenvalues = eigenvalues[zeros]
    eigenvectors = eigenvectors[:, zeros].T
    # Lets chop each eignvector
    splits = np.cumsum([0] + sizes)
    pairs = []
    for p, vec in zip(eigenvalues, eigenvectors):
        vecs = [vec[first:last] for first, last in zip(splits[:-1], splits[1:])]
        pairs.append((p, vecs))

    dt = t.stop()
    info('Done in %g' % dt)

    return pairs


def eig(A_blocks, B_blocks=None, backend='dolfin', **kwargs):
    '''
    Given block structured matrices A, B solve the generalized eigenvalue problem.
    '''
    # We assume here that the non-Matrix block is int
    sizes = []
    for row in A_blocks:
        for col in row:
            if isinstance(col, (Matrix, PETScMatrix)):
                nrows = col.size(0)
                sizes.append(nrows)
                break
    if B_blocks is None: 
        B_blocks = identity_matrix(sizes)
        B_blocks = block_mat(B_blocks)

    size = sum(sizes)
    # Small systems are handled directly
    if size < 10000:
        AA = block_to_scipy(A_blocks)
        BB = block_to_scipy(B_blocks)

        info('Computing eigenvalues A(%s), B(%s)' % (AA.shape, BB.shape))
        t = Timer('gevp')
        eigenvalues = eigvalsh(AA.toarray(), BB.toarray())

        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.loglog(np.sort(np.abs(eigenvalues)), '*', linestyle='none')
        # plt.show()

        dt = t.stop()
        info('Done in %g' % dt)

        return size, eigenvalues
    # Iterative guys
    if backend == 'slepc':
        AA = block_mat_to_matnest(A_blocks)
        BB = block_mat_to_matnest(B_blocks)
        return eig_slepc(AA, BB, **kwargs)

    if backend == 'dolfin':
        AA = block_to_dolfin(A_blocks)
        BB = block_to_dolfin(B_blocks)
        return eig_dolfin(AA, BB, **kwargs)


# FIXME
def eig_slepc(AA, BB, **kwargs):
    '''Slepc solver'''
    assert has_slepc4py()
    from slepc4py import SLEPc

    nev = 3  # Number of eigenvalues
    eigenvalues = np.array([])
    size = AA.size[0]
    info('System size: %i' % size)
    eps_type = kwargs.get('eps_type', SLEPc.EPS.Type.GD)
    for which in (SLEPc.EPS.Which.SMALLEST_MAGNITUDE, SLEPc.EPS.Which.LARGEST_MAGNITUDE):
        # Setup the eigensolver
        E = SLEPc.EPS().create()
        E.setOperators(AA ,BB)
        E.setType(E.Type.GD)
        E.setDimensions(nev, PETSc.DECIDE)
        E.setTolerances(1E-6, 8000)

        E.setWhichEigenpairs(which)
        E.setProblemType(SLEPc.EPS.ProblemType.GHEP)
        E.setFromOptions()

        # Solve the eigensystem
        E.solve()

        its = E.getIterationNumber()
        info('Number of iterations of the method: %i' % its)
        nconv = E.getConverged()
        info('Number of converged eigenpairs: %d' % nconv)
        assert nconv > 0
        eigenvalues = np.r_[eigenvalues, 
                            np.array([E.getEigenvalue(i).real
                                      for i in range(nconv)])]
    return size, eigenvalues


def eig_dolfin(AA, BB, **kwargs):
    '''DOLFIN solver'''
    nev = 3  # Number of eigenvalues
    eigenvalues = np.array([])
    size = AA.size(0)
    info('System size: %i' % size)
    parameters = kwargs.get('parameters', {})
    for spec in ('smallest magnitude', 'largest magnitude'):
        eigensolver = SLEPcEigenSolver(AA, BB)
        eigensolver.parameters['spectrum'] = spec
        eigensolver.parameters['tolerance'] = 1E-8
        eigensolver.parameters['problem_type'] = 'gen_hermitian'
        # eigensolver.parameters['spectral_transform'] = 'shift-and-invert'
        # eigensolver.parameters['spectral_shift'] = 5924.   # Fine grid lambda_min

        eigensolver.solve(nev)
        nconv = eigensolver.get_number_converged() 
        info('Number of converged eigenpairs: %d' % nconv)
        assert nconv > 0
        if has_slepc4py():
            its = eigensolver.eps().getIterationNumber()
            info('Number of iterations of the method: %i' % its)

        eigenvalues = np.r_[eigenvalues, 
                            np.array([eigensolver.get_eigenpair(i)[0]
                                      for i in range(nconv)])]
    return size, eigenvalues

# ----------------------------------------------------------------------------

if __name__ == '__main__':
    from aux_systems import mp_system, t_system

    # system, select = mp_system, [4, 8, 16, 32, 64]
    system, select = t_system, [1, 2, 4, 8, 16, 32]
    # ------------------------------------------------------------------------
    # FIXME: need preconditioner, ST?
    #        the other system - setup with DGT, the solution?
    #        work on EMI system and 2 preconditioner, schur - find, Hdiv - how?

    template = '%d %g %g %g'
    # Test with mixed Poisson
    preconds = ['one']
    # eps_type = SLEPc.EPS.Type.KRYLOVSCHUR
    for precond in preconds:
        rows = []
        out = open('mp_eigs_%s' % precond, 'w')
        for ncells in select:
            AA, BB = system(ncells, precond=precond)

            size, eigs = eig(AA, BB, backend='dolfin', parameters={'solver': 'krylov-schur'})

            print min(eigs), max(eigs)
            lmin, lmax = np.sort(np.abs(eigs))[[0, -1]]
            cond = lmax/lmin
            print lmin, lmax, cond
            row = [size, lmin, lmax, cond]
            rows.append(row)

            out.write(template % tuple(row) + '\n')
        out.close()
        
        print
        print precond
        for row in rows:
            print template % tuple(row)
