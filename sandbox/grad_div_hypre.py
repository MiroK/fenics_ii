from dolfin import *
from petsc4py import PETSc

n = 32

mesh = UnitSquareMesh(n, n)
V = FunctionSpace(mesh, 'RT', 1)

u, v = TrialFunction(V), TestFunction(V)

f = Expression(('sin(pi*(x[0]+x[1])', 'sin(pi*(x[0]+x[1])')), degree=3)

a = inner(div(u), div(v))*dx + inner(u, v)*dx
L = inner(f, v)

A, b = assemble_system(a, L, bc)

# Solve
# The gradient matrix
Q = FunctionSpace(mesh, 'CG', 1)
p = TrialFunction(Q)

G = assemble(inner(v, grad(p))*dx)

# CG Solver
ksp = PETSc.KSP().create(PETSc.COMM_WORLD)
ksp.setType('cg')
ksp.setTolerances(rtol=1E-8, atol=1E-12, divtol=1.0E10, max_it=300)

# AMS preconditioner
pc = ksp.getPC()
pc.setType('hypre')
pc.setHYPREType('ams')

vec = lambda x: as_backend_type(x).vec()
mat = lambda A: as_backend_type(A).mat()

# Attach gradient
pc.setHYPREDiscreteGradient(mat(G))

# Constant nullspace (in case not mass and bcs)
constants = [vec(interpolate(c, V).vector())
             for c in (Constant((1, 0)), Constant((0, 1)))]

pc.setHYPRESetEdgeConstantVectors(*constants)

# pc.setHYPRESetBetaPoissonMatrix(None)

# Set operator for the linear solver
ksp.setOperators(mat(A))

uh = Function(V)

ksp.solve(vec(b), vec(uh))

# Show linear solver details
ksp.view()
