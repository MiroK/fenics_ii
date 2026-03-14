from dolfin import *

from xii import *
from xii.linalg.convert import numpy_to_petsc
from block import block_mat
import numpy as np

from block.algebraic.petsc import LU
from block.algebraic.petsc import KSP
from block.iterative import ConjGrad

mesh = UnitSquareMesh(16, 16)

V = FunctionSpace(mesh, 'CG', 1)
print(V.dim())

u, v = TrialFunction(V), TestFunction(V)

M = assemble(inner(grad(u), grad(v))*dx + inner(u, v)*dx)
K = assemble(inner(u, v)*dx)
C = assemble(2*inner(u, v)*dx)

x, y = SpatialCoordinate(mesh)

f = x**2 + 2*y
L = inner(f, v)*dx
b = assemble(L)

Minv_arr = np.linalg.inv(M.array())
Kinv_arr = np.linalg.inv(K.array())

iMpiS_inv = np.linalg.inv(Minv_arr + Kinv_arr)

AA = numpy_to_petsc(iMpiS_inv + C.array())

BB = block_mat([[M + C, C],
                [C, K + C]])

B = monolithic(BB)
invB = LU(B)

S = StackOperator(2, V)  # V - >
W = [V, V]
R = ReductionOperator([2], W)

precond = S.T*R.T*invB*R*S
invA = ConjGrad(AA, precond=precond, tolerance=1E-14)

x = invA*b
eigv = invA.eigenvalue_estimates()
lmin, lmax = np.sort(eigv)[[0, -1]]
print(x.norm('l2'), lmin, lmax, (AA*x - b).norm('l2'))


# V -> (VxV) -> R ...
# 
