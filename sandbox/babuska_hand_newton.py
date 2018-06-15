# Nonlinear Babuska
# -div(nl(u)*grad(u)) = f in [0, 1]^2
#               T(u) = h on boundary
#

from dolfin import *
from xii import *

mesh = UnitSquareMesh(32, 32)
bmesh = BoundaryMesh(mesh, 'exterior')

V = FunctionSpace(mesh, 'CG', 1)
Q = FunctionSpace(bmesh, 'CG', 1)
W = (V, Q)

up = ii_Function(W)
u, p = up  # Split

v, q = map(TestFunction, W)
Tu, Tv = (Trace(x, bmesh) for x in (u, v))

dxGamma = Measure('dx', domain=bmesh)

# Forcing
f = interpolate(Expression('sin(pi*(x[0]+x[1]))', degree=1), V)
h = interpolate(Expression('x[0]*x[0]+x[1]*x[1]', degree=1), Q)

# Nonlinearity
nl = lambda u: (1+u)**2

# The nonlinear functional
F = [inner((1 + u)**2*grad(u), grad(v))*dx + inner(p, Tv)*dxGamma - inner(f, v)*dx,
     inner(Tu, q)*dxGamma - inner(h, q)*dxGamma]

dF = block_jacobian(F, up)

# Newton
eps = 1.0
tol = 1.0E-10
niter = 0
maxiter = 25

dup = ii_Function(W)
while eps > tol and niter < maxiter:
    niter += 1
    
    A, b = map(ii_assemble, (dF, F))

    A, b = map(ii_convert, (A, b))
        
    solve(A, dup.vector(), b)
        
    eps = sqrt(sum(x.norm('l2')**2 for x in dup.vectors()))
        
    print '%d |du| = %.6E |A|= %.6E |b| = %.6E' % (niter, eps, A.norm('linf'), b.norm('l2'))

    # FIXME: Update
    for i in range(len(W)):
        up[i].vector().axpy(-1, dup[i].vector())

for i, x in enumerate(up):
    File('./nl_results/%s_%d.pvd' % ('test_nlin', i)) << x

