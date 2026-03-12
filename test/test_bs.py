from dolfin import *
from xii import *


if False:
    mesh = BoxMesh(Point(-0.5, -0.5, -0.5), Point(0.5, 0.5, 0.5), 8, 8, 8)


    segment = CompiledSubDomain('near(x[0], 0.0) && near(x[1], 0.0) && x[2] < 0.0+tol',
                                tol=1E-10)
    point = CompiledSubDomain('near(x[2], 0.0)')
    
    f = Expression('x[0]*x[0] + x[1]*x[1] + x[2]*x[2]', degree=2)
else:
    mesh = BoxMesh(Point(0, 0, 0), Point(1, 1, 1), 8, 8, 8)

    segment = CompiledSubDomain('near(x[0], 0.5) && near(x[1], 0.5) && x[2] < 0.5+tol',
                                tol=1E-10)
    point = CompiledSubDomain('near(x[2], 0.5)')
    
    f = Expression('pow(x[0]-0.5, 2) + pow(x[1]-0.5, 2) + pow(x[2]-0.5, 2)', degree=2)    
    
edge_f = MeshFunction('size_t', mesh, 1, 0)

segment.mark(edge_f, 1)


line_mesh = EmbeddedMesh(edge_f, 1)
vertex_f = MeshFunction('size_t', line_mesh, 0, 0)
point.mark(vertex_f, 1)
ds_ = Measure('ds', domain=line_mesh, subdomain_data=vertex_f)

assert assemble(Constant(1)*ds_(1)) > 0

radius = 0.1
shape = Ball(radius, degree=20)
print(shape)
Q = FunctionSpace(line_mesh, 'R', 0)
q = TestFunction(Q)

V = FunctionSpace(mesh, 'CG', 2)
fh = interpolate(f, V)

Pi_f = Average(fh, line_mesh, shape, normalize=False) #False)
L = inner(Pi_f, q)*ds_(1)
print(ii_assemble(L).get_local()[0], (2*2*pi*(radius**5)/5))

Pi_f = Average(fh, line_mesh, shape, normalize=True)
L = inner(Pi_f, q)*ds_(1)
print(ii_assemble(L).get_local()[0], (2*2*pi*(radius**5)/5)/(4/3*pi*radius**3))

