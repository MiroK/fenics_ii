from dolfin import *
import numpy as np
from xii import *

mesh = RectangleMesh(Point(0, 0), Point(1, 2), 8, 6)

cell_f = MeshFunction('size_t', mesh, 2, 0)
CompiledSubDomain('x[0] > 0.5-DOLFIN_EPS').mark(cell_f, 1)
dG = Measure('dx', domain=mesh, subdomain_data=cell_f)

Q = FunctionSpace(mesh, 'DG', 0)
p_foo, q_foo = Function(Q), Function(Q)
p_foo.vector().set_local(np.random.rand(Q.dim()))
q_foo.vector().set_local(np.random.rand(Q.dim()))

mu = Function(Q)
mu.vector().set_local(np.random.rand(Q.dim())**2)

mean_p = Constant(assemble(Constant(4)*p_foo*dG(1)))
mean_q = Constant(assemble(Constant(4)*q_foo*dG(1)))
# print(mean_p(0), mean_q(0))
target = assemble(inner((1/mu)*mean_p, mean_q)*dx(domain=mesh))


p, q = TrialFunction(Q), TestFunction(Q)

pi_p, pi_q = (Mean(arg, measure=dG(1), weight=Constant(4)) for arg in (p, q))
B = ii_assemble((1/mu*inner(pi_p, pi_q)*dx))

mine = q_foo.vector().inner(B*p_foo.vector())
print(target, mine)    
