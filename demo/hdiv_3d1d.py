# FIXME: scaling wrt to radius of the different inner products!

from dolfin import *
from xii import *
import numpy as np

n = 2**3

mesh = UnitCubeMesh(n, n, 2*n)
radius = 0.01           # Averaging radius for cyl. surface
quadrature_degree = 10  # Quadraure degree for that integration

K = Constant(1E-4)
Kb = Constant(1E0)

gamma = MeshFunction('size_t', mesh, 1, 0)
CompiledSubDomain('near(x[0], 0.5) && near(x[1], 0.5)').mark(gamma, 1)
bmesh = EmbeddedMesh(gamma, 1)

tau = TangentCurve(bmesh)
Div = lambda arg, t=tau: dot(grad(arg), tau)
    
V = FunctionSpace(mesh, 'RT', 1)
Vb = FunctionSpace(bmesh, 'CG', 1)

Q = FunctionSpace(mesh, 'DG', 0)
Qb = FunctionSpace(bmesh, 'DG', 0)

W = (V, Vb, Q, Qb)

u, ub, p, pb = map(TrialFunction, W)
v, vb, q, qb = map(TestFunction, W)

# Averaging surface
cylinder = Circle(radius=radius, degree=quadrature_degree)

Pi_u, Pi_v = (FluxAverage(arg, bmesh, cylinder) for arg in (u, v))

dx_ = Measure('dx', domain=bmesh)

# Parts without the coupling
a = block_form(W, 2)
a[0][0] = (1/K)*inner(u, v)*dx
a[0][2] = -inner(p, div(v))*dx
a[1][1] = (1/Kb)*inner(ub, vb)*dx
a[1][3] = -inner(pb, Div(vb))*dx_
a[2][0] = -inner(q, div(u))*dx
a[3][1] = -inner(qb, Div(ub))*dx_

n = FacetNormal(mesh)
p0 = Expression('x[0]', degree=1)

L = block_form(W, 1)
L[0] = -inner(p0, dot(v, n))*ds
L[3] = inner(Constant(1), qb)*dx

A, b = map(ii_assemble, (a, L))

gamma = Constant(1E2)

# Add cooupling
A[0][0] += ii_assemble(gamma*inner(Pi_u, Pi_v)*dx_)
A[0][1] += ii_assemble(-gamma*inner(ub, Pi_v)*dx_)
A[1][0] += ii_assemble(-gamma*inner(vb, Pi_u)*dx_)
A[1][1] += ii_assemble(gamma*inner(ub, vb)*dx_)

wh = ii_Function(W)
solve(monolithic(A), wh.vector(), monolithic(b))

uh, ubh, ph, pbh = wh
wh.rename([('uh', ''), ('ubh', ''), ('ph', ''), ('pbh', '')])

for whi in wh:
    print(whi.vector().norm('l2'))
    with XDMFFile(f'{whi.name()}.xdmf') as out:
        out.write(whi)

