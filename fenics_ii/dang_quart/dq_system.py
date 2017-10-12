import sys
sys.path.append('..')
from dolfin import *
from block import block_mat, block_vec, block_bc
from trace_tools.embedded_mesh import EmbeddedMesh
from trace_tools.trace_matrices import Lagrange_trace_matrix
from trace_tools.pipe_surf_avg import pipe_surface_average_operator
from utils.direct import dolfin_solve
from utils.convert import set_lg_rc_map
from random_path import random_path
from petsc4py import PETSc
import numpy as np

# Start with mesh
n = 5
omega = UnitCubeMesh(2*n, 2*n, 8*n)

gamma_edges = EdgeFunction('size_t', omega, 0)
CompiledSubDomain('near(x[0], 0.5) && near(x[1], 0.5)').mark(gamma_edges, 1)
gamma_edges = random_path(omega)

gamma = EmbeddedMesh(omega, gamma_edges, 1)

# Onto blocks of the system
V = FunctionSpace(omega, 'CG', 1)
Q = FunctionSpace(gamma.mesh, 'CG', 1)
W = [V, Q]

u, p = map(TrialFunction, W)
v, q = map(TestFunction, W)

# A + M
C0, C1 = Constant(0.2), Constant(0.4)
AM = C0*inner(grad(u), grad(v))*dx + C1*inner(u, v)*dx
AM = assemble(AM)

# a + m
c0, c1 = Constant(0.1), Constant(0.3)
am = c0*inner(grad(p), grad(q))*dx + c1*inner(p, q)*dx
am = assemble(am)

# Nontrivial part, first we make the pieces
beta = 0.7
m = beta*inner(p, q)*dx
m = assemble(m)

T = Lagrange_trace_matrix(space=V, trace_space=Q)
Pi = pipe_surface_average_operator(V, Q, R=0.1, deg=5)

# Convert everthing to petsc.mat
pieces = AM, am, m, Pi
AM, am, m, Pi = map(lambda matrix: as_backend_type(matrix).mat(), pieces)

# -(beta*m)*Pi
B10 = PETSc.Mat()
m.matMult(Pi, B10)
assert B10.size == (Q.dim(), V.dim())

# T'((beta*m)*Pi
B00 = PETSc.Mat()
T.transposeMatMult(B10, B00)
assert B00.size == (V.dim(), V.dim())

# Now we can flip the sign back
B10 *= -1

B01 = PETSc.Mat()
T.transposeMatMult(m, B01)
B01 *= -1
assert B01.size == (V.dim(), Q.dim())

# Combining with AM, am blocks
#[(C0*A + C1*M + beta*T'*m*Pi), -beta*T'm;  [u;       [f;
#                                                   = 
# - beta*m*Pi,      (c0*a + c1*m + beta*m)]  p]        g].
AM.axpy(1., B00)
am.axpy(1., m)

# Wrap everything for dolfin
AM, B01, B10, am = map(lambda mat: PETScMatrix(mat), (AM, B01, B10, am))
# Add parts so that we have u and p block separared
AA = block_mat([[AM, B01],
                [B10, am]])
# The rhs
F, f = Expression('sin(pi*(x[0]+x[1]+x[2]))', degree=3), Constant(1)
bV = assemble(inner(F, v)*dx)
bQ = assemble(inner(f, q)*dx)
bb = block_vec([bV, bQ])

# NOTE: For setting bcs the hand built matrices need row/col maps to be set
AA = set_lg_rc_map(AA, W)
# Sample bcs
# FIXME: Agree on what goes here
# bcs = [[DirichletBC(V, Constant(0), 'near(x[0], %g)' % Xmin[0])],
#        [DirichletBC(Q, Constant(1), 'on_boundary')]]
# block_bc(bcs, symmetric=False).apply(AA).apply(bb)

# Finally solve (direct!)
timer = Timer('LU')
timer.start()
info('About to LU solve the system of size %d.' % (V.dim()+Q.dim()))
uh, ph = dolfin_solve(AA, bb, spaces=W)
dt = timer.stop()
info('Done in %g s.' % dt)

File('uh.pvd') << uh
File('ph.pvd') << ph
