# I check here assembly of coupling integrals that are needed in Federica's
# LM formulation
from __future__ import absolute_import
from __future__ import print_function
from dolfin import *
from weak_bcs.burman.generation import StraightLineMesh
import numpy as np
from xii import *
from six.moves import map


H, n = 4, 10
# Vasculature
mesh_1d = StraightLineMesh(np.array([0.5, 0.5, 0]),
                           np.array([0.5, 0.5, H]),
                           3*n)  # Just to make it finer

# This is a model background mesh where all the cells would be intersected
# by a line. FIXME: how to compute this
mesh = BoxMesh(Point(0, 0, 0), Point(1, 1, H), 1, 1, n)

# Now we have a bulk unknown, the 1d unknown and the multiplier
V3 = FunctionSpace(mesh, 'CG', 1)          # This
V1 = FunctionSpace(mesh_1d, 'CG', 1)
Q = FunctionSpace(mesh, 'DG', 0)           # and this would be different

u3, u1 = list(map(TrialFunction, (V3, V1)))
q = TestFunction(Q)

# 3d->1d
avg_shape = Circle(lambda x: 0.05, degree=10)
Pi_u = Average(u3, mesh_1d, avg_shape)
# To aveluate the multiplier on the line we use trace (Average of None shape)
Tq = Average(q, mesh_1d, None)

# The coupling is still on the line
dx_ = Measure('dx', domain=mesh_1d)
# The coupling term 
a20 = inner(Pi_u, Tq)*dx_
a21 = - inner(u1, Tq)*dx_

forms = (a20, a21)
for a in forms:
    A = ii_convert(ii_assemble(a)).array()
    print(np.linalg.norm(A, 2))
