from dolfin import *
import numpy as np
from xii import *


mesh = UnitSquareMesh(32, 32)

V = FunctionSpace(mesh, 'CG', 1)
f = Function(V)
f.vector().set_local(np.random.rand(V.dim()))

smesh = BoundaryMesh(mesh, 'exterior')

true = assemble(f*ds)#/assemble(Constant(1)*ds(domain=mesh))


fhat = SurfaceMean(f, smesh)
dx_ = Measure('dx', domain=smesh)
form = fhat*dx_

value = ii_assemble(form)

print(true, value)

Q = FunctionSpace(smesh, 'R', 0)
q = TestFunction(Q)
u = TrialFunction(V)

form = inner(q, SurfaceMean(u, smesh))*dx_
B = ii_assemble(form)

fvec = f.vector()
value = (B*fvec).get_local()[0]

print(true, value)
