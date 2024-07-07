from dolfin import *
import numpy as np
from xii import *

mesh = RectangleMesh(Point(0, 0), Point(1, 2), 5, 6)

Q = FunctionSpace(mesh, 'DG', 0)
p_foo, q_foo = Function(Q), Function(Q)
p_foo.vector().set_local(np.random.rand(Q.dim()))
q_foo.vector().set_local(np.random.rand(Q.dim()))

mu = Function(Q)
mu.vector().set_local(np.random.rand(Q.dim())**2)

mean_p = Constant(assemble(p_foo*dx))
mean_q = Constant(assemble(q_foo*dx))
# print(mean_p(0), mean_q(0))
target = assemble(inner((1/mu)*mean_p, mean_q)*dx(domain=mesh))


p, q = TrialFunction(Q), TestFunction(Q)

pi_p, pi_q = Mean(p), Mean(q)
B = ii_assemble((1/mu*inner(pi_p, pi_q)*dx))

mine = q_foo.vector().inner(B*p_foo.vector())
print(target, mine)    
