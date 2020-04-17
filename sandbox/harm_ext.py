from dolfin import *
from xii import (EmbeddedMesh, ii_assemble, Extension, Trace, ii_assemble,
                 StraightLineMesh, apply_bc)
import numpy as np


# This is how the "harmonic extension could work"
# We extend from x = 0.5. As a result of geometry computation there
# will be submesh made of cells of background intersected by the curve
# and a facet function on it which corresponds to domain of extension
nx = 31
ny = (nx+1)/2
A, B = (nx-1)/2./nx, (nx+1)/2./nx

mesh = UnitSquareMesh(nx, ny)

cell_f = MeshFunction('size_t', mesh, 2, 0)
CompiledSubDomain('x[0] > A - tol && x[0] < B + tol', A=A, B=B, tol=DOLFIN_EPS).mark(cell_f, 1)

left = CompiledSubDomain('A-tol < x[0] && x[0] < A+tol', A=A, tol=DOLFIN_EPS)
right = CompiledSubDomain('A-tol < x[0] && x[0] < A+tol', A=B, tol=DOLFIN_EPS)

# We would be extending to
facet_f = MeshFunction('size_t', mesh, 1, 0)
left.mark(facet_f, 1)
right.mark(facet_f, 1)

ext_mesh = EmbeddedMesh(facet_f, 1)
Q = FunctionSpace(ext_mesh, 'CG', 1)

# The auxiliary problem would be speced at
aux_mesh = SubMesh(mesh, cell_f, 1)
facet_f = MeshFunction('size_t', aux_mesh, 1, 0)
DomainBoundary().mark(facet_f, 1)
left.mark(facet_f, 2)
right.mark(facet_f, 2)

# Extending from
gamma_mesh = StraightLineMesh(np.array([0.5, 0]), np.array([0.5, 1]), 3*ny)
V1 = FunctionSpace(gamma_mesh, 'CG', 1)
f = interpolate(Constant(1), V1)

# The extension problem would be
V2 = FunctionSpace(aux_mesh, 'CG', 1)
u2, v2 = TrialFunction(V2), TestFunction(V2)

a = inner(grad(u2), grad(v2))*dx + inner(u2, v2)*dx
L = inner(f, Trace(v2, gamma_mesh))*dx(domain=gamma_mesh)

A, b = list(map(ii_assemble, (a, L)))

# We have boundary conditions to apply
# bc = DirichletBC(V2, Constant(0), facet_f, 1)
# A, b = apply_bc(A, b, bc)

u2h = Function(V2)
solve(A, u2h.vector(), b)

# We get to extend domain by
p, q = TrialFunction(Q), TestFunction(Q)
f = Trace(u2h, ext_mesh)

a = inner(p, q)*dx
L = inner(f, q)*dx(domain=ext_mesh)

A, b = list(map(ii_assemble, (a, L)))

# We have boundary conditions to apply
# bc = DirichletBC(Q, u2h, 'on_boundary')
# A, b = apply_bc(A, b, bc)

qh = Function(Q)
solve(A, qh.vector(), b)

File('u2h.pvd') << u2h
File('qh.pvd') << qh
