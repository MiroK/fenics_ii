from xii import *
from dolfin import *

i = 2

def main(i):
    n = 4*2**i
    mesh = UnitSquareMesh(*(n, )*2)
    bmesh = BoundaryMesh(mesh, 'exterior')

    V = FunctionSpace(mesh, 'CG', 1)
    Q = FunctionSpace(bmesh, 'DG', 0)
    W = [V, Q]

    u, p = map(TrialFunction, W)
    v, q = map(TestFunction, W)
    Tu = Trace(u, bmesh)
    Tv = Trace(v, bmesh)

    # The line integral
    dx_ = Measure('dx', domain=bmesh)

    a00 = inner(grad(u), grad(v))*dx + inner(u, v)*dx
    a01 = inner(Tv, p)*dx_
    a10 = inner(Tu, q)*dx_

    f = Expression('sin(pi*(x[0]+x[1]))', degree=3)
    
    L0 = inner(f, v)*dx
    L1 = inner(f, q)*dx_
        
    a = [[a00, a01], [a10, 0]]
    L = [L0, L1]

    wh = ii_Function(W)

    AA, bb = map(ii_convert, map(ii_assemble, (a, L)))
    LUSolver('umfpack').solve(AA, wh.vector(), bb)

    # Verify a01 for computed solution
    x = ii_assemble(inner(Trace(wh[0], bmesh), q)*dx_)
    y = ii_assemble(L1)

    assert (x - y).norm('linf') < 1E-13

# --------------------------------------------------------------------

if __name__ == '__main__':
    map(main, (3, 4, 5, 6))
