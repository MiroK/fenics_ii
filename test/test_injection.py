from xii.meshing.refinement import centroid_refine
from dolfin import *
import numpy as np
from xii import *
import sympy as sp


n = 10
mesh_c = UnitIntervalMesh(n)
mesh = centroid_refine(mesh_c)    

U = VectorFunctionSpace(mesh, 'CG', 1, 2)
Lc = VectorFunctionSpace(mesh_c, 'CG', 1, 2)

W = [U, Lc]
u, lmc = list(map(TrialFunction, W))
v, gmc = list(map(TestFunction, W))

lm, gm = Injection(lmc, mesh), Injection(gmc, mesh)
dx = Measure('dx', domain=mesh)

C_arr = np.array([[1, 2], [3, 4]])
C = Constant(C_arr)

a = inner(u.dx(0) + dot(C, u), gm)*dx
A = ii_assemble(a)

x = sp.Symbol('x[0]')
u = sp.Matrix([2*x - 1, 3*x+2])

g = sp.Matrix([3*x + 1, x])

left = u.diff(x, 1) + sp.Matrix(sp.Matrix(C_arr).dot(u))
left_expr = Expression(list(map(sp.printing.ccode, left)), degree=1)

u_expr = Expression(list(map(sp.printing.ccode, u)), degree=1)
g_expr = Expression(list(map(sp.printing.ccode, g)), degree=1)

reference = assemble(inner(left_expr, g_expr)*dx(domain=mesh))

u_ = interpolate(u_expr, U).vector()
g_ = interpolate(g_expr, Lc).vector()

mine = g_.inner(A*u_)

print(abs(mine - reference))

# Let's try with convergence
x = sp.Symbol('x[0]')
u = sp.Matrix([2*x**2 - 1, 3*x**2+2])
g = sp.Matrix([3*x**2 + 1, x**2])

u_expr = Expression(list(map(sp.printing.ccode, u)), degree=1)
g_expr = Expression(list(map(sp.printing.ccode, g)), degree=1)

reference = sp.integrate((u.diff(x, 1) + sp.Matrix(sp.Matrix(C_arr).dot(u))).dot(g), (x, 0, 1))

for n in (8, 16, 32, 64, 128):
    mesh_c = UnitIntervalMesh(n)
    mesh = centroid_refine(mesh_c)    

    U = VectorFunctionSpace(mesh, 'CG', 1, 2)
    Lc = VectorFunctionSpace(mesh_c, 'CG', 1, 2)

    W = [U, Lc]
    u, lmc = list(map(TrialFunction, W))
    v, gmc = list(map(TestFunction, W))

    lm, gm = Injection(lmc, mesh), Injection(gmc, mesh)
    dx = Measure('dx', domain=mesh)

    C_arr = np.array([[1, 2], [3, 4]])
    C = Constant(C_arr)

    a = inner(u.dx(0) + dot(C, u), gm)*dx
    A = ii_assemble(a)

    u_ = interpolate(u_expr, U).vector()
    g_ = interpolate(g_expr, Lc).vector()

    mine = g_.inner(A*u_)

    print(abs(mine - reference))
