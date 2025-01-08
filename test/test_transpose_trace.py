from dolfin import *
from xii import *
import numpy as np


def get_solution_trace(boundaries, u0, f0):
    '''Map u to facet'''
    
    mesh = boundaries.mesh()
    bmesh = EmbeddedMesh(boundaries, 1)

    V = FunctionSpace(mesh, 'CG', 1)
    Q = FunctionSpace(bmesh, 'CG', 1)
    W = [V, Q]

    u, p = map(TrialFunction, W)
    v, q = map(TestFunction, W)

    Tu, Tv = Trace(u, bmesh), Trace(v, bmesh)
    dx_ = Measure('dx', domain=bmesh)
    
    a = block_form(W, 2)
    a[0][0] = inner(grad(u), grad(v))*dx
    a[0][1] = inner(Tv, p)*dx_
    a[1][0] = inner(Tu, q)*dx_

    L = block_form(W, 1)
    L[0] = inner(f0, v)*dx
    L[1] = inner(u0, q)*dx_

    A, b = map(ii_assemble, (a, L))

    wh = ii_Function(W)
    solve(monolithic(A), wh.vector(), monolithic(b))
    uh, ph = wh

    return uh


def get_solution_transpose_trace(boundaries, u0, f0):
    '''Map u to facet'''
    
    mesh = boundaries.mesh()
    bmesh = EmbeddedMesh(boundaries, 1)

    V = FunctionSpace(mesh, 'CG', 1)
    Q = FunctionSpace(bmesh, 'CG', 1)
    W = [V, Q]

    u, p = map(TrialFunction, W)
    v, q = map(TestFunction, W)

    Ep, Eq = TransposeTrace(p, bmesh), TransposeTrace(q, bmesh)
    ds = Measure('ds', domain=mesh, subdomain_data=boundaries)
    dx_ = Measure('dx', domain=bmesh)
    
    a = [[inner(grad(u), grad(v))*dx, inner(v, Ep)*ds],
         [inner(u, Eq)*ds,                          0]]
    
    L = [inner(f0, v)*dx,
         inner(u0, q)*dx_]

    A, b = map(ii_assemble, (a, L))

    wh = ii_Function(W)
    solve(monolithic(A), wh.vector(), monolithic(b))
    uh, ph = wh

    return uh


def get_solution_HCG(boundaries, u0, f0):
    '''Map u to facet'''
    mesh = boundaries.mesh()
    bmesh = EmbeddedMesh(boundaries, 1)

    V = FunctionSpace(mesh, 'CG', 1)
    Q = FunctionSpace(bmesh, 'CG', 1)
    W = [V, Q]

    u, p = map(TrialFunction, W)
    v, q = map(TestFunction, W)

    Ep, Eq = TransposeTrace(p, bmesh), TransposeTrace(q, bmesh)
    ds = Measure('ds', domain=mesh, subdomain_data=boundaries)
    dx_ = Measure('dx', domain=bmesh)

    gamma = Constant(20)
    h = Constant(mesh.hmin())
    n = FacetNormal(mesh)
    
    a00 = (inner(grad(u), grad(v))*dx - inner(dot(grad(v), n), u)*ds + (gamma/h)*inner(u, v)*ds
           - inner(dot(grad(u), n), v)*ds)

    a01 = inner(dot(grad(v), n), Ep)*ds - (gamma/h)*inner(v, Ep)*ds
    a10 = inner(dot(grad(u), n), Eq)*ds - (gamma/h)*inner(u, Eq)*ds
    a11 = (gamma/h)*inner(p, q)*dx_
    
    a = [[a00, a01],
         [a10, a11]]
    
    L = [inner(f0, v)*dx,
         inner(Constant(0), q)*dx_]

    A, b = map(ii_assemble, (a, L))

    Qbcs = [dict(enumerate(interpolate(u0, Q).vector().get_local()))]
    Wbcs = [[], Qbcs]
    A, b = apply_bc(A, b, bcs=Wbcs)
    
    wh = ii_Function(W)
    solve(monolithic(A), wh.vector(), monolithic(b))
    uh, ph = wh

    return uh

# --------------------------------------------------------------------

if __name__ == '__main__':
    import tabulate
    import ulfy
    
    x, y = SpatialCoordinate(UnitSquareMesh(2, 2))
    u = sin(pi*(x-y))
    f = -div(grad(u)) + u

    u0, f0 = (ulfy.Expression(arg, degree=4) for arg in (u, f))

    # get_solution = get_solution_trace
    # get_solution = get_solution_transpose_trace
    get_solution = get_solution_HCG

    history, h0, e0 = [], None, None
    for k in range(1, 6):
        ncells = 2**k
        
        mesh = UnitSquareMesh(ncells, ncells)

        facet_f = MeshFunction('size_t', mesh, mesh.topology().dim()-1, 0)
        DomainBoundary().mark(facet_f, 1)

        uh = get_solution(facet_f, u0, f0)
        h = mesh.hmin()
        ndofs = uh.vector().local_size()
        eh = errornorm(u0, uh, 'H1')
        
        if e0 is None:
            rate = -1
        else:
            rate = ln(eh/e0)/ln(h/h0)
        history.append((h, ndofs, eh, rate))

        e0, h0 = eh, h
        
        print(tabulate.tabulate(history))

    uh.vector().axpy(-1, interpolate(u0, uh.function_space()).vector())
    File('uh.pvd') << uh
