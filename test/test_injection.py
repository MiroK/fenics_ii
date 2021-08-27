from xii.meshing.refinement import centroid_refine
from dolfin import *
import numpy as np
from xii import *
import pytest

@pytest.mark.parametrize('refine_method', (refine, centroid_refine))
@pytest.mark.parametrize('inject_method', (None, 'interpolate'))
def test_P2_P1(refine_method, inject_method):
    mesh = UnitSquareMesh(16, 16)
    
    rmesh = refine_method(mesh)

    V = FunctionSpace(rmesh, 'CG', 1)
    Q = FunctionSpace(mesh, 'CG', 2)

    u, v = TrialFunction(V), TestFunction(V)
    p, q = TrialFunction(Q), TestFunction(Q)
    
    m = inject_method
    Ip, Iq = Injection(p, rmesh, not_nested_method=m), Injection(q, rmesh, not_nested_method=m)

    M = ii_assemble(inner(Ip, v)*dx(domain=rmesh))

    f = Expression('x[0]*x[0] + 2*x[1]*x[1]', degree=2)
    g = Expression('x[0] + x[1]', degree=1)

    value0 = assemble(inner(f, g)*dx(domain=mesh))
    value = (interpolate(g, V).vector()).inner(M*(interpolate(f, Q).vector()))

    assert abs(value - value0) < 1E-13


@pytest.mark.parametrize('refine_method', (refine, centroid_refine))
@pytest.mark.parametrize('inject_method', (None, 'interpolate'))
def test_P1_P1(refine_method, inject_method):
    mesh = UnitSquareMesh(16, 16)
    
    rmesh = refine_method(mesh)

    V = FunctionSpace(rmesh, 'CG', 1)
    Q = FunctionSpace(mesh, 'CG', 1)

    u, v = TrialFunction(V), TestFunction(V)
    p, q = TrialFunction(Q), TestFunction(Q)
    
    m = inject_method
    Ip, Iq = Injection(p, rmesh, not_nested_method=m), Injection(q, rmesh, not_nested_method=m)

    M = ii_assemble(inner(Ip, v)*dx(domain=rmesh))

    f = Expression('x[0] + 2*x[1]', degree=2)
    g = Expression('x[0] + x[1]', degree=1)

    value0 = assemble(inner(f, g)*dx(domain=mesh))
    value = (interpolate(g, V).vector()).inner(M*(interpolate(f, Q).vector()))

    assert abs(value - value0) < 1E-13


@pytest.mark.parametrize('refine_method', (refine, centroid_refine))
@pytest.mark.parametrize('inject_method', (None, 'interpolate'))
def test_P1_P0(refine_method, inject_method):
    mesh = UnitSquareMesh(32, 32)
    
    rmesh = refine_method(mesh)

    V = FunctionSpace(rmesh, 'DG', 0)
    Q = FunctionSpace(mesh, 'CG', 1)

    u, v = TrialFunction(V), TestFunction(V)
    p, q = TrialFunction(Q), TestFunction(Q)
    
    m = inject_method
    Ip, Iq = Injection(p, rmesh, not_nested_method=m), Injection(q, rmesh, not_nested_method=m)

    M = ii_assemble(inner(Ip, v)*dx(domain=rmesh))

    f = Expression('x[0] + x[1]', degree=1)
    
    for g, tol in ((Constant(2), 1E-10), (Expression('x[0] + 2*x[1]', degree=1), 1E-2)):
        g = interpolate(g, V)

        value0 = assemble(inner(f, g)*dx(domain=mesh))
        value = (interpolate(g, V).vector()).inner(M*(interpolate(f, Q).vector()))

        assert abs(value - value0) < tol


@pytest.mark.parametrize('refine_method', (refine, centroid_refine))
def test_P2_P1_nn(refine_method):
    mesh = UnitSquareMesh(16, 16)
    rmesh = UnitSquareMesh(17, 19)

    V = FunctionSpace(rmesh, 'CG', 1)
    Q = FunctionSpace(mesh, 'CG', 2)

    u, v = TrialFunction(V), TestFunction(V)
    p, q = TrialFunction(Q), TestFunction(Q)
    
    Ip, Iq = Injection(p, rmesh), Injection(q, rmesh)

    M = ii_assemble(inner(Ip, v)*dx(domain=rmesh))

    f = Expression('x[0]*x[0] + 2*x[1]*x[1]', degree=2)
    g = Expression('x[0] + x[1]', degree=1)

    value0 = assemble(inner(f, g)*dx(domain=mesh))
    value = (interpolate(g, V).vector()).inner(M*(interpolate(f, Q).vector()))

    assert abs(value - value0) < 1E-13
