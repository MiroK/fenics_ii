from __future__ import absolute_import
from __future__ import print_function
from dolfin import *
from xii.assembler.trace_matrix import trace_mat
from xii import EmbeddedMesh


f = Expression('x[0]*x[0] -2*x[1]*x[1] - x[2]*x[1] - x[0]*x[1]', degree=2)
g = Expression('x[2]*x[2] -3*x[1]*x[1] - 4*x[2]*x[1] - x[0]*x[1]', degree=2)

true = lambda mesh, f=f, g=g: assemble(inner(f, g)*ds(domain=mesh))

# Exact --------------------------------------------------------------
mesh = UnitCubeMesh(10, 10, 10)
bmesh = BoundaryMesh(mesh, 'exterior')

V = FunctionSpace(mesh, 'CG', 2)
TV = FunctionSpace(bmesh, 'CG', 2)

Trace = trace_mat(V, TV, bmesh, {'restriction': '', 'normal': None})

f_ = interpolate(f, V)
Tf = Function(TV)
Trace.mult(f_.vector(), Tf.vector())

g_ = interpolate(g, TV)
ans = assemble(inner(Tf, g_)*dx(domain=bmesh))
true = true(mesh)
print((ans - true), ans)

# Approximation
e0, h0 = None, None
for n in (4, 8, 16, 32):
    mesh = UnitCubeMesh(*(n, )*3)
    bmesh = BoundaryMesh(mesh, 'exterior')

    V = FunctionSpace(mesh, 'CG', 1)
    TV = FunctionSpace(bmesh, 'CG', 1)

    Trace = trace_mat(V, TV, bmesh, {'restriction': '', 'normal': None})

    f_ = interpolate(f, V)
    Tf = Function(TV)
    Trace.mult(f_.vector(), Tf.vector())

    g_ = interpolate(g, TV)
    ans = assemble(inner(Tf, g_)*dx(domain=bmesh))
    h = bmesh.hmin()
    
    if e0 is not None:
        rate = ln(abs(ans-true)/e0)/ln(h/h0)
    else:
        rate = None
    e0, h0 = ans-true, h
    
    print(n, '->', abs(ans - true), ans, rate)
