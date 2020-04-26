from __future__ import absolute_import
from dolfin import *
from xii import *


def setup_domain(n):
    '''
    Inner is [0.25, 0.75]^2, inner is [0, 1]^2 \ [0.25, 0.75]^2 and 
    \partial [0.25, 0.75]^2 is the interface
    '''
    # Avoiding mortar meshes here because of speed 
    interior = CompiledSubDomain('std::max(fabs(x[0] - 0.5), fabs(x[1] - 0.5)) < 0.25')
    outer_mesh = UnitSquareMesh(n, n)
    
    subdomains = MeshFunction('size_t', outer_mesh, outer_mesh.topology().dim(), 0)
    # Awkward marking
    for cell in cells(outer_mesh):
        x = cell.midpoint().array()            
        subdomains[cell] = int(interior.inside(x, False))
    assert sum(1 for _ in SubsetIterator(subdomains, 1)) > 0

    stokes_domain = EmbeddedMesh(subdomains, 0)
    darcy_domain = EmbeddedMesh(subdomains, 1)

    # Interior boundary
    surfaces = MeshFunction('size_t', darcy_domain, darcy_domain.topology().dim()-1, 0)
    DomainBoundary().mark(surfaces, 1)
    iface_domain = EmbeddedMesh(surfaces, 1)

    # Mark the outiside for Stokes
    facet_f = MeshFunction('size_t', stokes_domain, stokes_domain.topology().dim()-1, 0)
    CompiledSubDomain('near(x[0]*(1-x[0]), 0) || near(x[1]*(1-x[1]), 0)').mark(facet_f, 1)
    stokes_domain.subdomains = facet_f

    return stokes_domain, darcy_domain, iface_domain


def term(n):
    stokes_domain, darcy_domain, iface_domain = setup_domain(n)

    V1 = VectorFunctionSpace(stokes_domain, 'CG', 2)
    Q = FunctionSpace(darcy_domain, 'CG', 1)
    Q1 = FunctionSpace(stokes_domain, 'CG', 1)

    v1 = TestFunction(V1)
    T_v1 = Trace(v1, iface_domain)

    p = TrialFunction(Q)
    T_p = Trace(p, iface_domain)

    n = OuterNormal(iface_domain, [0.5, 0.5])  # Outer of Darcy
    n1 = -n                                    # Outer of Stokes

    dxGamma = Measure('dx', domain=iface_domain)
    a = inner(dot(T_v1, n1), T_p)*dxGamma

    A = ii_convert(ii_assemble(a))

    uh_expr = Expression(('x[0]*x[0]+2*x[0]*x[1]', 'x[1]*x[1]-x[0]*x[1]'), degree=2)
    ph_expr = Expression('2*x[0]-x[1]', degree=1)
    
    uh = interpolate(uh_expr, V1)
    ph = interpolate(ph_expr, Q)

    x = Function(V1).vector()
    A.mult(ph.vector(), x)
    ans = x.inner(uh.vector())

    ans0 = assemble(inner(dot(uh_expr, n1), ph_expr)*dxGamma)
    
    return abs(ans-ans0) < 1E-14

# --------------------------------------------------------------------

if __name__ == '__main__':
    assert all(term(n) for n in (4, 8, 16, 32))
