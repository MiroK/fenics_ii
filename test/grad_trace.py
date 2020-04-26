from __future__ import absolute_import
from __future__ import print_function
from dolfin import *
from xii import *
from xii.assembler.trace_matrix import trace_mat_no_restrict
import numpy as np
import block


def check(ncells, Qelm):
    mesh = UnitSquareMesh(*(ncells, )*2)
    bmesh = BoundaryMesh(mesh, 'exterior')

    V = FunctionSpace(mesh, 'CG', 1)
    Q = FunctionSpace(bmesh, Qelm(bmesh.ufl_cell()))

    u = TrialFunction(V)
    q = TestFunction(Q)

    n = FacetNormal(mesh)
    # Want grad(u).n * q
    v = TestFunction(V)

    a = inner(dot(grad(u), n), v)*ds
    A = assemble(a)
    # Now take it from testV to test Q

    T = PETScMatrix(trace_mat_no_restrict(V, Q, bmesh))
    
    B = ii_convert(T*A)

    # Now we should be able to integrate linear function exactly
    x, y = SpatialCoordinate(mesh)
    
    f = 2*x + y
    g = x - 3*y

    truth = assemble(inner(dot(grad(f), n), g)*ds)

    fV = interpolate(Expression('x[1] + 2*x[0]', degree=1), V)
    gQ = interpolate(Expression('x[0] - 3*x[1]', degree=1), Q)

    me = gQ.vector().inner(B*fV.vector())

    return abs(truth - me)

# Exeact as we linears are used for testing
assert check(ncells=8, Qelm=lambda c: FiniteElement('Lagrange', c, 1)) < 1E-14
assert check(ncells=16, Qelm=lambda c: FiniteElement('Lagrange', c, 1)) < 1E-14
assert check(ncells=32, Qelm=lambda c: FiniteElement('Lagrange', c, 1)) < 1E-14

# Does it converge with P1-P0
errors = []
for n in (8, 16, 32, 64, 128):
    errors.append(check(ncells=n,
                        Qelm=lambda c: FiniteElement('Discontinuous Lagrange', c, 0)))

errors = np.array(errors)

assert np.all(np.log(errors[1:]/errors[:-1])/np.log(1./2) > 1.5)
print(errors)
