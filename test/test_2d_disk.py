from dolfin import *
from xii import *

from hsmg.hseig import HsEig as HsEigNorm
from block.algebraic.petsc import LU, KSP
import numpy as np

ncells = 128

mesh = UnitSquareMesh(ncells, ncells)

facet_f = MeshFunction('size_t', mesh, 1, 0)
segment = CompiledSubDomain('near(x[1], 0.5) && (x[0] > 0.25-tol) && (x[0] < 0.75+tol)', tol=1E-10)
segment.mark(facet_f, 1)

line_mesh = EmbeddedMesh(facet_f, 1)
dx_ = Measure('dx', domain=line_mesh)

V = FunctionSpace(mesh, 'CG', 1)
Q = FunctionSpace(line_mesh, 'CG', 1)
W = [V, Q]

u, p = map(TrialFunction, W)
v, q = map(TestFunction, W)

radius = 0.2
disk = Disk(radius=radius, degree=20, quad_scheme='simple')
Pi_u, Pi_v = (Average(arg, line_mesh, disk) for arg in (u, v))

a = block_form(W, 2)
a[0][0] = inner(grad(u), grad(v))*dx
a[0][1] = inner(Pi_v, p)*dx_
a[1][0] = inner(Pi_u, q)*dx_

L = block_form(W, 1)
L[1] = inner(Constant(1), q)*dx_

Vbcs = [DirichletBC(V, Constant(0), 'on_boundary')]
Qbcs = []  #DirichletBC(Q, Constant(0), 'on_boundary')]
Wbcs = [Vbcs, Qbcs]

A, b = map(ii_assemble, (a, L))
A, b = apply_bc(A, b, bcs=Wbcs)

facet_f = MeshFunction('size_t', line_mesh, line_mesh.topology().dim()-1, 0)
DomainBoundary().mark(facet_f, 1)
Hs = HsEigNorm(Q, s=-1.0, bcs=None).collapse()  #[(facet_f, 1)]).collapse()

B = block_diag_mat([LU(A[0][0]), LU(Hs)])

Ainv = KSP(A, precond=B, 
            # PETScOptions
            ksp_type='minres',
            ksp_rtol=1E-12,
            ksp_view=None,
            ksp_max_it=1_000,
            ksp_monitor_true_residual=None,
            ksp_initial_guess_nonzero=1,
            ksp_converged_reason=None)

xx = Ainv*b

wh = ii_Function(W)
for i, xxi in enumerate(xx):
    wh[i].vector().axpy(1, xxi)
niters = len(Ainv.residuals)

eigw = np.sort(np.abs(Ainv.eigenvalue_estimates()))
lmin, lmax = eigw[[0, -1]]
print(niters, lmin, lmax, lmax/lmin)


uh, ph = wh

File('uh.pvd') << uh
File('ph.pvd') << ph
