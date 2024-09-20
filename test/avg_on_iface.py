from dolfin import *
from xii import *
from xii.meshing.make_mesh_cpp import make_mesh
from xii.assembler.average_matrix import average_matrix
from xii.assembler.average_shape import Circle
from xii import EmbeddedMesh
import numpy as np
import pytest


n = 8
radius = 1E-3
mesh = BoxMesh(Point(-1, -1, -1), Point(1, 1, 1), n, n, n)

cell_f = MeshFunction('size_t', mesh, mesh.topology().dim(), 0)
CompiledSubDomain('x[0] < DOLFIN_EPS').mark(cell_f, 1)
dx = Measure('dx', domain=mesh, subdomain_data=cell_f)

edge_f = MeshFunction('size_t', mesh, 1, 0)
CompiledSubDomain('near(x[0], 0.0) && near(x[1], 0.0)').mark(edge_f, 1)
line_mesh = EmbeddedMesh(edge_f, 1)


V = FunctionSpace(mesh, 'DG', 1)
TV = FunctionSpace(line_mesh, 'DG', 1)

# Make up different data for LHS and RHS
fs = {0: Expression('x[2]', degree=1),
      1: Expression('10+x[2]', degree=1)}

u, v = TrialFunction(V), TestFunction(V)
a = inner(u, v)*dx
L = sum(inner(fi, v)*dx(i) for (i, fi) in fs.items())

fh = Function(V)
solve(a == L, fh)

cylinder = Circle(radius, degree=20)

# Which one to pick on the interface ...
which = 1
resolve_interfaces = InterfaceResolution(subdomains=cell_f,
                                         resolve_conflicts={(0, 1): which})
# resolve_interfaces = None
Pif = Average(fh, line_mesh, cylinder, normalize=True, resolve_interfaces=resolve_interfaces)

Q = FunctionSpace(line_mesh, 'DG', 1)
p, q = TrialFunction(Q), TestFunction(Q)
dx_ = Measure('dx', domain=line_mesh)

a = inner(p, q)*dx_
L = inner(Pif, q)*dx_
A, b = (ii_convert(ii_assemble(form)) for form in (a, L))

Pifh = Function(Q)
solve(A, Pifh.vector(), b)

# ... compare against that
print(errornorm(fs[which], Pifh))
