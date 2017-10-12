from dolfin import *
from scipy.linalg import eigh
import numpy as np

gamma = ['near((x[0]-0.25)*(x[0]-0.75), 0) && (0.25-DOLFIN_EPS < x[1]) && (x[1] < 0.75+DOLFIN_EPS)',
         'near((x[1]-0.25)*(x[1]-0.75), 0) && (0.25-DOLFIN_EPS < x[0]) && (x[0] < 0.75+DOLFIN_EPS)']
gamma = map(lambda x: '('+x+')', gamma)
gamma = ' || '.join(gamma)
gamma = CompiledSubDomain(gamma)

n = 2
n *= 4
mesh = UnitSquareMesh(n, n)
facet_f = FacetFunction('size_t', mesh, 0)
gamma.mark(facet_f, 1)

V = FunctionSpace(mesh, 'CG', 1)
u = TrialFunction(V)
v = TestFunction(V)

not_gamma = lambda x, on_boundary: not gamma.inside(x, on_boundary)
bc = DirichletBC(V, Constant(0), not_gamma, 'pointwise')
n = V.dim()
_TOL = 1E-10*np.log(n)

a_form = inner(grad(u), grad(v))*dx
m_form = inner(u, v)*dx
L_form = inner(Constant(0), v)*dx

# Full EVP problem
A, _ = assemble_system(a_form, L_form, bc)
A = A.array()

info('Solving EVP of size %d' % len(A))
timer = Timer('EVP')
eigw0, eigv0 = eigh(A)
t0 = timer.stop()
info('Done in %g' % t0)

# Now we claim that to each bdry dof the corresponding column of identity
# matrix is an eigenvector with the eigenvalue being the sole nonzero entry
# of the vector
I = bc.get_boundary_values().keys()
eigw_I = []
eigv_I = []
m = len(I)
for i in I:
    ei = np.zeros(n)
    ei[i] = 1.

    lmbda_i = A.dot(ei).sum()
    assert np.linalg.norm(A.dot(ei)-lmbda_i*ei) < _TOL

    eigw_I.append(lmbda_i)
    eigv_I.append(ei)
E = np.array(eigv_I).T

# Build the reduced problem
I_perp = set(map(long, range(0, n))).difference(set(I))
k = len(I_perp)
assert m + k == n

E_perp = np.zeros((n, k))
for row, col in enumerate(I_perp): E_perp[col, row] = 1. 

assert np.linalg.norm(E_perp.T.dot(E)) < _TOL 

A_hat = E_perp.T.dot(A.dot(E_perp))
info('Solving reduced EVP of size %d' % len(A_hat))
timer = Timer('EVP')
eigw_Ip, eigv_Ip = eigh(A_hat)
t = timer.stop()
info('Done in %g' % t)

# The full eigenvalues from the reduced problem
eigw = np.sort(eigw_I + eigw_Ip.tolist())
# Let's see about the difference in eigenvalues
info('Eigenvalue error %g' % np.linalg.norm(eigw-eigw0))

# Build the full eigenvectors
# Make sure they really are eigenvectors
X = eigv_Ip.T
Y = E.T
for lmbda, c in zip(eigw_Ip, X):
    u = E_perp.dot(c)
    assert np.linalg.norm(A.dot(u) - lmbda*u) < _TOL
    # They new eigs are in the nullspace of E.T
    assert np.linalg.norm(Y.dot(u)) < _TOL

# And finally are othonormal
for i, ui in enumerate(X):
    assert abs((ui*ui).sum() - 1) < _TOL
    for uj in X[i+1:]:
        assert abs((ui*uj).sum()) < _TOL
