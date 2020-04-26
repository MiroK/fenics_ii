from __future__ import absolute_import
from __future__ import print_function
from dolfin import *
from xii import inverse, VectorizedOperator
from hsmg import HsNorm
import numpy as np


mesh = UnitSquareMesh(10, 10)
VV = FunctionSpace(mesh, MixedElement([FiniteElement('Lagrange', triangle, 1)]*3))
u, v = TrialFunction(VV), TestFunction(VV)
M = assemble(inner(u, v)*dx)

x = M.create_vec()
x.set_local(np.random.rand(x.local_size()))

y = M*x

V = FunctionSpace(mesh, 'CG', 1)
u, v = TrialFunction(V), TestFunction(V)
Mj = assemble(inner(u, v)*dx)
# Should match M
X = VectorizedOperator(Mj, VV)
y0 = X*x

print(x.norm('l2'), (y - y0).norm('linf'))

I = HsNorm(V, s=0.5)
II = VectorizedOperator(I, VV)

foo = II*x
x0 = inverse(II)*foo

print((x - x0).norm('linf'))
