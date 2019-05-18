from dolfin import *
from xii import EmbeddedMesh, ii_assemble, Extension, Trace, ii_assemble
import numpy as np

n = 4

mesh = UnitSquareMesh(n, n)

# We will have the 1d guy live in the middle and the multiplier slighly
# off
middle = CompiledSubDomain('near(x[0], 0.5)')
off_middle = CompiledSubDomain('near(x[0], A)', A=(n/2+1.)/n)

facet_f = MeshFunction('size_t', mesh, 1, 0)
middle.mark(facet_f, 1)
off_middle.mark(facet_f, 2)

# Mesh to extend from, (so the d1)
mesh_1d = EmbeddedMesh(facet_f, 1)
# And thhe multiplier one
mesh_lm = EmbeddedMesh(facet_f, 2)

V_2d = FunctionSpace(mesh, 'CG', 1)
V_1d = FunctionSpace(mesh_1d, 'CG', 1)
Q = FunctionSpace(mesh_lm, 'CG', 1)

# We are after (Trace(V_2d) - Ext(V_1d), Q)
q = TestFunction(Q)
u2d, u1d = TrialFunction(V_2d), TrialFunction(V_1d)
Tu2d = Trace(u2d, mesh_lm)
Eu1d = Extension(u1d, mesh_lm, type='uniform')

dxLM = Measure('dx', domain=mesh_lm)

T = ii_assemble(inner(Tu2d, q)*dxLM)
E = ii_assemble(inner(Eu1d, q)*dxLM)

f2d = Expression('1+x[0]+2*x[1]', degree=1)
# The extended function will practically be shifted so I want invariance
# in that case
f1d = Expression('2-x[1]', degree=1)
# LM is free
fLM = Expression('4-2*x[1]+x[0]', degree=1)

# What we are after
true = assemble(inner(f2d - f1d, fLM)*dxLM)

# How do we get it?
f2d_ = interpolate(f2d, V_2d)
f1d_ = interpolate(f1d, V_1d)
fLM_ = interpolate(fLM, Q)

mine = fLM_.vector().inner(T*f2d_.vector() - E*f1d_.vector())

print abs(true - mine)
