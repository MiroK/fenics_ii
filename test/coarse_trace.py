from xii import *
from dolfin import *


# n = 16
# mesh = UnitSquareMesh(*(n, )*2)
# mesh_fine = UnitSquareMesh(*(2*n, )*2)
# bmesh = BoundaryMesh(mesh_fine, 'exterior')

# V = FunctionSpace(mesh, 'CG', 1)
# Q = FunctionSpace(bmesh, 'CG', 1)

# v = TestFunction(V)
# p = TrialFunction(Q)
# Tv = Trace(v, bmesh)

# # The line integral
# dx_ = Measure('dx', domain=bmesh)
# a = inner(Tv, p)*dx_
# A = ii_assemble(a)

# foo = Expression('1+x[0]-2*x[1]', degree=1)
# bar = Expression('1-3*x[0]-2*x[1]', degree=1)

# true = assemble(foo*bar*ds(domain=mesh_fine))

# num = interpolate(bar, V).vector().inner(A*interpolate(foo, Q).vector())

# print abs(true - num)


from xii.assembler.nonconforming_trace_matrix import nonconforming_trace_mat
from block import block_assemble

for n in (8, ):
    mesh = UnitSquareMesh(*(n, )*2)

    mesh_fine = UnitSquareMesh(*(3*n, )*2)
    cell_f = MeshFunction('size_t', mesh_fine, 1, 0)
    CompiledSubDomain('near(x[0], 0)').mark(cell_f, 1)
    bmesh = EmbeddedMesh(cell_f, 1)

    V = FunctionSpace(mesh, 'BDM', 1)
    bar = Expression(('x[0]', 'x[1]'), degree=1)

    v = interpolate(bar, V)
    V = VectorFunctionSpace(mesh, 'CG', 1)
    # V = VectorFunctionSpace(mesh, 'DG', 1)
    v = interpolate(v, V)

    L = inner(bar - v, bar - v)*dx
    print '>>>', sqrt(abs(assemble(L)))


    Q = VectorFunctionSpace(bmesh, 'DG', 1)

    T = PETScMatrix(nonconforming_trace_mat(V, Q))

    v = interpolate(bar, V)
    q = Function(Q, T*v.vector())

    L = inner(bar - q, bar - q)*dx
    print sqrt(abs(assemble(L)))
    # foo 
    # num = interpolate(bar, V).vector().inner(A*interpolate(foo, Q).vector())
    # print abs(true - num)


# for n in (8, 16, 32):
#     mesh = UnitSquareMesh(*(n, )*2)

#     mesh_fine = UnitSquareMesh(*(3*n, )*2)
#     cell_f = MeshFunction('size_t', mesh_fine, 1, 0)
#     CompiledSubDomain('near(x[0], 0)').mark(cell_f, 1)
#     bmesh = EmbeddedMesh(cell_f, 1)

#     V = FunctionSpace(mesh, 'BDM', 1)
#     # V = VectorFunctionSpace(mesh, 'CG', 1)
#     Q = FunctionSpace(bmesh, 'CG', 1)

#     v = TestFunction(V)
#     p = TrialFunction(Q)
#     Tv = Trace(v, bmesh)

#     n = Constant((1, 0))
#     # The line integral
#     dx_ = Measure('dx', domain=bmesh)
#     a = inner(dot(Tv, n), p)*dx_
#     A = ii_assemble(a)

#     bar = Expression(('1+x[0]-2*x[1]', '0'), degree=1)
#     foo = Expression('1-3*x[0]-2*x[1]', degree=1)

#     bar_ = Expression('1+x[0]-2*x[1]', degree=1)   
#     true = assemble(foo*bar_*ds(domain=mesh_fine, subdomain_data=cell_f)(1))

#     num = interpolate(bar, V).vector().inner(A*interpolate(foo, Q).vector())
    
#     print abs(true - num)
