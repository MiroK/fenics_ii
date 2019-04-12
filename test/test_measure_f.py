from xii.assembler.average_form import Average
from xii.assembler.caverage_form import CrossAverage
from xii.assembler.average_matrix import MeasureFunction
from xii.assembler.average_shape import Circle, Square
from xii.meshing.embedded_mesh import EmbeddedMesh
import dolfin as df
import numpy as np


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
print df.sqrt(abs(df.assemble(L)))

reduced = CrossAverage(u, line_mesh, ci)
f = MeasureFunction(reduced)
# Circumnference of the circle
true = df.Expression('pi*(x[2]/2)*(x[2]/2)', degree=2)
L = df.inner(f - true, f - true)*df.dx
print df.sqrt(abs(df.assemble(L)))

# --------------------------------------------------------------------

# Grow thicker with z
ci = Square(P=lambda x0: x0 - np.array([x0[2]/2., x0[2]/2., 0]), degree=8)

reduced = Average(u, line_mesh, ci)
f = MeasureFunction(reduced)
# Circumnference of the square
true = df.Expression('4*(2*x[2]/2)', degree=1)
L = df.inner(f - true, f - true)*df.dx
print df.sqrt(abs(df.assemble(L)))

reduced = CrossAverage(u, line_mesh, ci)
f = MeasureFunction(reduced)
# Circumnference of the circle
true = df.Expression('(2*x[2]/2)*(2*x[2]/2)', degree=2)
L = df.inner(f - true, f - true)*df.dx
print df.sqrt(abs(df.assemble(L)))

