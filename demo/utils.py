from collections import namedtuple
from xii import EmbeddedMesh
import dolfin as df

# Convenient way to get tangents
rotate = lambda v: df.dot(df.Constant(((0, 1), (-1, 0))), v)

# For piecewise straight domains expressions depending e.g. on normal
# are easiest to define piecewise
PiecewiseExpression = namedtuple('PiecewiseExpr', ('subdomains', 'expressions'))


def immersed_geometry(i, which):
    '''
    We discretize the geometrical setup where inner domain [0.25, 0.75]^2
    is enclosed by the outer domain such that their union fills a unit 
    square. Return mesh and (coupling relevant marking function)
    '''
    # The surfaces are tagged as follows
    # Outer                      Inner
    #       ^
    #       8
    #       v                       ^
    #       4                       4
    # <5 1>   <2  6>            <1     2>   
    #       3                       3 
    #       ^                       v
    #       7
    #       v
    n = 4*2**i
    mesh = df.UnitSquareMesh(n, n)

    inner = df.CompiledSubDomain(' && '.join(['(x[0] < 0.75+DOLFIN_EPS)',
                                              '(0.25-DOLFIN_EPS < x[0])',
                                              '(x[1] < 0.75+DOLFIN_EPS)',
                                              '(0.25-DOLFIN_EPS < x[1])']))

    cell_f = df.MeshFunction('size_t', mesh, 2, 1)
    inner.mark(cell_f, 2)

    outer_domain = EmbeddedMesh(cell_f, 1)
    outer_ff = df.MeshFunction('size_t', outer_domain, 1, 0)
    outer_boundaries = {
        1: df.CompiledSubDomain('near(x[0], 0.25) && ((0.25-DOLFIN_EPS < x[1]) && (x[1] < 0.75+DOLFIN_EPS))'),
        2: df.CompiledSubDomain('near(x[0], 0.75) && ((0.25-DOLFIN_EPS < x[1]) && (x[1] < 0.75+DOLFIN_EPS))'),
        3: df.CompiledSubDomain('near(x[1], 0.25) && ((0.25-DOLFIN_EPS < x[0]) && (x[0] < 0.75+DOLFIN_EPS))'),
        4: df.CompiledSubDomain('near(x[1], 0.75) && ((0.25-DOLFIN_EPS < x[0]) && (x[0] < 0.75+DOLFIN_EPS))'),
        5: df.CompiledSubDomain('near(x[0], 0.0)'),
        6: df.CompiledSubDomain('near(x[0], 1.0)'),
        7: df.CompiledSubDomain('near(x[1], 0.0)'),
        8: df.CompiledSubDomain('near(x[1], 1.0)')}
    [bdry.mark(outer_ff, tag) for tag, bdry in outer_boundaries.items()]

    inner_domain = EmbeddedMesh(cell_f, 2)
    inner_ff = df.MeshFunction('size_t', inner_domain, 1, 0)
    inner_boundaries = {
        1: df.CompiledSubDomain('near(x[0], 0.25) && ((0.25-DOLFIN_EPS < x[1]) && (x[1] < 0.75+DOLFIN_EPS))'),
        2: df.CompiledSubDomain('near(x[0], 0.75) && ((0.25-DOLFIN_EPS < x[1]) && (x[1] < 0.75+DOLFIN_EPS))'),
        3: df.CompiledSubDomain('near(x[1], 0.25) && ((0.25-DOLFIN_EPS < x[0]) && (x[0] < 0.75+DOLFIN_EPS))'),
        4: df.CompiledSubDomain('near(x[1], 0.75) && ((0.25-DOLFIN_EPS < x[0]) && (x[0] < 0.75+DOLFIN_EPS))')
    }
    [bdry.mark(inner_ff, tag) for tag, bdry in inner_boundaries.items()]
    
    if which == 'outer':
        return (outer_domain, outer_ff)

    if which == 'inner':
        return (inner_domain, inner_ff)

    assert which == 'interface'
    
    imesh = EmbeddedMesh(inner_ff, (1, 2, 3, 4))
    return (imesh, imesh.marking_function)
