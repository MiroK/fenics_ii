from xii.assembler.trace_matrix import trace_mat_no_restrict
from dolfin import *
from xii import *


def test_DLT():
    '''Injection of DLT dofs'''
    for n in (4, 8, 16, 32, 64):
        mesh = UnitSquareMesh(n, n)

        V = FunctionSpace(mesh, 'Discontinuous Lagrange Trace', 0)
        v = TestFunction(V)

        f = Expression('x[0]+2*x[1]', degree=1)

        fV = Function(V)
        assemble((1/FacetArea(mesh))*inner(f, v)*ds, tensor=fV.vector())

        facet_f = MeshFunction('size_t', mesh, 1, 0)
        DomainBoundary().mark(facet_f, 1)

        trace_mesh = EmbeddedMesh(facet_f, 1)
        TV = FunctionSpace(trace_mesh, 'DG', 0)

        T = PETScMatrix(trace_mat_no_restrict(V, TV, trace_mesh))

        gTV = Function(TV)
        gTV.vector()[:] = T*fV.vector()

        assert sqrt(abs(assemble(inner(gTV-f, gTV-f)*ds))) < 1E-13

test_DLT()
