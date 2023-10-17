from dolfin import *
from xii import *


def test():
    mesh = UnitSquareMesh(32, 32, 'crossed')

    facet_f = MeshFunction('size_t', mesh, 1, 0)

    subdomains = ('near(x[0], x[1]) && x[0] < 0.5 + tol',
                  'near(x[0], 1-x[1]) && x[0] < 0.5 + tol',
                  'near(x[0], x[1]) && x[0] > 0.5 - tol', 
                  'near(x[0], 1-x[1]) && x[0] > 0.5 - tol',
                  'near(x[0], 0.5) && x[1] < 0.5 + tol',
                  'near(x[0], 0.5) && x[1] > 0.5 - tol',
                  'near(x[1], 0.5) && x[0] < 0.5 + tol',
                  'near(x[1], 0.5) && x[0] > 0.5 - tol')

    colors = []
    for color, subd in enumerate(subdomains, 1):
        CompiledSubDomain(subd, tol=1E-10).mark(facet_f, color)
        colors.append(color)

    mesh = EmbeddedMesh(facet_f, colors)
    cell_f = mesh.marking_function


    dx = Measure('dx', domain=mesh, subdomain_data=cell_f)
    hK = CellVolume(mesh)
    # Make up a function
    V = FunctionSpace(mesh, 'DG', 0)
    v = TestFunction(V)

    fh = Function(V)
    assemble(sum((1/hK)*inner(Constant(tag), v)*dx(tag) for tag in colors),
             fh.vector())

    _, v2c = mesh.init(0, 1), mesh.topology()(0, 1)

    bif, = (v for v in range(mesh.num_vertices()) if len(v2c(v)) > 2)

    R = FunctionSpace(mesh, 'Real', 0)
    u = TrialFunction(V)
    dr = TestFunction(R)

    scale = Constant(assemble(Constant(1)*dS(domain=mesh)))
    for cell in v2c(bif):
        A = ii_assemble(inner(1/scale*avg(PointTrace(u, point=bif, cell=cell)), avg(dr))*dS)
        x = A*fh.vector()

        val, = x.get_local()
        val0 = cell_f[cell]
        assert abs(val - val0) < 1E-10, (val, val0)
