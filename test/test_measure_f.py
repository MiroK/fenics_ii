from __future__ import absolute_import
from xii.assembler.average_form import Average
from xii.assembler.average_matrix import MeasureFunction
from xii.assembler.average_shape import Circle, Square, SquareRim, Disk
from xii.meshing.embedded_mesh import EmbeddedMesh
import dolfin as df
import numpy as np


def is_close(a, b=0): return abs(a-b) < 1E-12


mesh = df.UnitCubeMesh(4, 4, 8)
u = df.Function(df.FunctionSpace(mesh, 'CG', 2))

# Make 1d
f = df.MeshFunction('size_t', mesh, 1, 0)
df.CompiledSubDomain('near(x[0], 0.5) && near(x[1], 0.5)').mark(f, 1)
line_mesh = EmbeddedMesh(f, 1)

# Grow thicker with z
ci = Circle(radius=lambda x0: x0[2]/2., degree=8)
reduced = Average(u, line_mesh, ci)
f = MeasureFunction(reduced)
# Circumnference of the circle
true = df.Expression('2*pi*x[2]/2', degree=1)
L = df.inner(f - true, f - true)*df.dx
assert is_close(df.sqrt(abs(df.assemble(L))))

di = Disk(radius=lambda x0: x0[2]/2., degree=8)
reduced = Average(u, line_mesh, di)
f = MeasureFunction(reduced)
# Circumnference of the circle
true = df.Expression('pi*(x[2]/2)*(x[2]/2)', degree=2)
L = df.inner(f - true, f - true)*df.dx
assert is_close(df.sqrt(abs(df.assemble(L))))

# --------------------------------------------------------------------

# Grow thicker with z
sq = SquareRim(P=lambda x0: x0 - np.array([x0[2]/2., x0[2]/2., 0]), degree=8)
reduced = Average(u, line_mesh, sq)
f = MeasureFunction(reduced)
# Circumnference of the square
true = df.Expression('4*(2*x[2]/2)', degree=1)
L = df.inner(f - true, f - true)*df.dx
assert is_close(df.sqrt(abs(df.assemble(L))))


sq = Square(P=lambda x0: x0 - np.array([x0[2]/2., x0[2]/2., 0]), degree=8)
reduced = Average(u, line_mesh, sq)
f = MeasureFunction(reduced)
# Circumnference of the circle
true = df.Expression('(2*x[2]/2)*(2*x[2]/2)', degree=2)
L = df.inner(f - true, f - true)*df.dx
assert is_close(df.sqrt(abs(df.assemble(L))))

# --------------------------------------------------------------------

mesh = df.UnitCubeMesh(16, 16, 16)
u = df.Function(df.FunctionSpace(mesh, 'CG', 2))

# Make 1d
f = df.MeshFunction('size_t', mesh, 1, 0)
df.CompiledSubDomain('near(x[0], x[1]) && near(x[1], x[2])').mark(f, 1)
line_mesh = EmbeddedMesh(f, 1)

rad = 0.1

ci = Circle(radius=lambda x0: rad, degree=8)
reduced = Average(u, line_mesh, ci)
f = MeasureFunction(reduced)
# Circumnference of the circle
true = df.Expression('2*pi*r', degree=1, r=rad)
L = df.inner(f - true, f - true)*df.dx
assert is_close(df.sqrt(abs(df.assemble(L))))

di = Disk(radius=lambda x0: rad, degree=8)
reduced = Average(u, line_mesh, di)
f = MeasureFunction(reduced)
# Circumnference of the circle
true = df.Expression('pi*r*r', degree=2, r=rad)
L = df.inner(f - true, f - true)*df.dx
assert is_close(df.sqrt(abs(df.assemble(L))))
