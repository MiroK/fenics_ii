# The system from d'Angelo & Quarteroni paper on tissue perfusion
# With Omega a 3d domain and Gamma a 1d domain inside it we want
#
# A1(grad(u), grad(v))_3 + A0(u, v)_3 + (Pi u, Tv)_3 - beta(p, Tv)_1 = (f, Tv)_1
# -beta(q, Pi u)_1      + a1(grad(p), grad(q))_1 + (a0+beta)(p, q)_1 = (f, q)_1
#

from dolfin import *
from xii import *


# I setup the constants arbitraily
Alpha1, Alpha0 = Constant(0.02), Constant(0.01)
alpha1, alpha0 = Constant(2), Constant(0.01)
beta = Constant(10)


n = 16
mesh3d = UnitCubeMesh(n, n, 2*n)
radius = 0.01           # Averaging radius for cyl. surface
quadrature_degree = 10  # Quadraure degree for that integration

# Not strictly necessary but here the 1d mesh is made of edges of 3d
edge_f = MeshFunction('size_t', mesh3d, 1, 0)
CompiledSubDomain('near(x[0], 0.5) && near(x[1], 0.5)').mark(edge_f, 1)
mesh1d = EmbeddedMesh(edge_f, 1)

V = FunctionSpace(mesh3d, 'CG', 1)
Q = FunctionSpace(mesh1d, 'CG', 1)
W = [V, Q]

u, p = list(map(TrialFunction, W))
v, q = list(map(TestFunction, W))

# Averaging surface is cylinder obtained by sweeping shape along 1d
shape = Circle(radius=radius, degree=quadrature_degree)

Pi_u = Average(u, mesh1d, shape=shape)  # Reduce 3d to 1d by surface integral
T_v = Average(v, mesh1d, shape=None) # This is 3d-1d trace

dxGamma = Measure('dx', domain=mesh1d)

a = block_form(W, 2)
# NOTE: block need to be defined on single space
a[0][0] = Alpha1*inner(grad(u), grad(v))*dx + Alpha0*inner(u, v)*dx + beta*inner(Pi_u, T_v)*dxGamma
a[0][1] = -beta*inner(p, T_v)*dxGamma
a[1][0] = -beta*inner(Pi_u, q)*dxGamma
a[1][1] = alpha1*inner(grad(p), grad(q))*dxGamma + (alpha0+beta)*inner(p, q)*dxGamma


f = Expression('sin(pi*(x[0]+x[1]+x[2]))', degree=4)

L = block_form(W, 1)  # The second argument is arity
L[0] = inner(f, T_v)*dxGamma
L[1] = inner(f, q)*dxGamma

# We assemble now into block_mat objects (not necessarily matrices)
A, b = list(map(ii_assemble, (a, L)))

# After the assembler pass; the block of A that involve coupling are not
# represented as matrices. Instead they are matrix expressions. For example
# the a[1][0] block is a product of M * Pi where Pi is the matrix representation
# of the averaging operator from V to PiV space and M is the mass matrix
# between PiV and Q
_, PiMat = A[1][0].chain
# Check that Pi really goes from V
assert PiMat.size(1) == V.dim()
# The PiV space is inferred by the assembler. If you are not happy with
# it you can always get the matrix representation your self
from xii.assembler.average_matrix import average_matrix
# Here we get it map to DG0 on the line
MyPiMat = average_matrix(V,
                         FunctionSpace(mesh1d, 'DG', 0),
                         shape=shape)
# Whily before DG1 was used
print('inferred', PiMat.size(0), 'manual', MyPiMat.size(0))


# Suppose now that there are also boundary conditions; we specify them
# as a list for every subspace
V_bcs = [DirichletBC(V, Constant(0), 'near(x[2], 0)')]
Q_bcs = []
W_bcs = [V_bcs, Q_bcs]
# Apply
A, b = apply_bc(A, b, W_bcs)

# Just checking if the off-diagonal block sane
print('|A01| and |A10|', A[0][1].norm('linf'), A[1][0].norm('linf'))
print('# Unknowns', sum(Wi.dim() for Wi in W))

wh = ii_Function(W)
# Here A is still a block operator

solve = 'gmres'
# If the system is to be solved directly there are some extra steps
if solve == 'lu':
    # Make the system monolithic
    A, b = list(map(ii_convert, (A, b)))

    LUSolver(A).solve(wh.vector(), b)
# Iterative
else:
    # Here for simplicity cbc.block
    from block.iterative import LGMRES
    from block.algebraic.petsc import AMG
    # A mock up preconditioner is based on inverse of the diagonal block
    # using AMG to approx inv
    B = block_diag_mat([AMG(A[0][0]), AMG(A[1][1])])

    Ainv = LGMRES(A, precond=B, tolerance=1E-10, show=1)
    x = Ainv*b

    for i in range(len(x)):
        wh[i].vector()[:] = x[i]
    
# Hope for no NaN
print(wh[0].vector().norm('l2'))
print(wh[1].vector().norm('l2'))

# Output
File('uh3d.pvd') << wh[0]
File('uh1d.pvd') << wh[1]
