from dolfin import *
from xii import OuterNormal, EmbeddedMesh
import numpy as np


mesh = UnitSquareMesh(10, 10)
f = MeshFunction('size_t', mesh, 1, 0)
CompiledSubDomain('near(x[1], 0)').mark(f, 1)
CompiledSubDomain('near(x[1], 1)').mark(f, 1)

bmesh = EmbeddedMesh(f, 1)
n = OuterNormal(bmesh, [0.5, 0.5])

# Top?
for x in bmesh.coordinates():
    if near(x[1], 1.0):
        assert np.linalg.norm(n(x) - np.array([0, 1.])) < 1E-13

# Bottom
for x in bmesh.coordinates():
    if near(x[1], 0.0):
        assert np.linalg.norm(n(x) - np.array([0, -1.])) < 1E-13
