from __future__ import absolute_import
from xii.assembler.average_shape import Circle, Square, SquareRim, Disk
from xii.assembler.average_form import Average, average_space
from xii.meshing.embedded_mesh import EmbeddedMesh
from xii.assembler.average_matrix import avg_mat

import dolfin as df
import numpy as np


def is_close(a, b=0): return abs(a-b) < 1E-12


L_curve = df.CompiledSubDomain('(near(x[0], 0.5) && near(x[1], 0.5) && x[2] < 0.5 + tol) || (near(x[2], 0.5) && near(x[1], 0.5) && x[0] > 0.5 - tol)',
                               tol=1E-10)


I_curve = df.CompiledSubDomain('(near(x[0], 0.5) && near(x[1], 0.5))',
                               tol=1E-10)


# --------------------------------------------------------------------


def sanity_test(n, subd, shape):
    '''Constant is preserved'''
    mesh = df.UnitCubeMesh(n, n, n)
    V = df.FunctionSpace(mesh, 'CG', 2)
    v = df.interpolate(df.Constant(1), V)

    f = df.MeshFunction('size_t', mesh, 1, 0)
    subd.mark(f, 1)
    
    line_mesh = EmbeddedMesh(f, 1)
    Q = average_space(V, line_mesh)
    q = df.Function(Q)
    
    Pi = avg_mat(V, Q, line_mesh, {'shape': shape})
    Pi.mult(v.vector(), q.vector())

    q0 = df.Constant(1)
    # Error
    L = df.inner(q0 - q, q0 - q)*df.dx

    e = q.vector().copy()
    e.axpy(-1, df.interpolate(q0, Q).vector())
    
    return df.sqrt(abs(df.assemble(L)))


if True:
    for shape in [Circle(radius=0.1, degree=10), Disk(radius=0.1, degree=10)]:
        for n in (4, 8, 16, 32):
            assert is_close(sanity_test(n, L_curve, shape=shape))

        
    for shape in [Circle(radius=0.1, degree=10),
                  Disk(radius=0.1, degree=10),
                  Square(P=lambda x0: x0-np.array([0.1, 0.1, 0]), degree=10),
                  SquareRim(P=lambda x0: x0-np.array([0.1, 0.1, 0]), degree=10)]:
        for n in (4, 8, 16, 32):
            assert is_close(sanity_test(n, I_curve, shape=shape))

            
# --------------------------------------------------------------------


def identity_test(n, shape, subd=I_curve):
    '''Averaging over indep coords of f'''
    true = df.Expression('x[2]*x[2]', degree=2)
    
    mesh = df.UnitCubeMesh(n, n, n)
    V = df.FunctionSpace(mesh, 'CG', 2)
    v = df.interpolate(true, V)

    f = df.MeshFunction('size_t', mesh, 1, 0)
    subd.mark(f, 1)
    
    line_mesh = EmbeddedMesh(f, 1)
    Q = average_space(V, line_mesh)
    q = df.Function(Q)
    
    Pi = avg_mat(V, Q, line_mesh, {'shape': shape})
    Pi.mult(v.vector(), q.vector())

    q0 = true
    # Error
    L = df.inner(q0 - q, q0 - q)*df.dx

    e = q.vector().copy()
    e.axpy(-1, df.interpolate(q0, Q).vector())
    
    return df.sqrt(abs(df.assemble(L)))

if True:
    for shape in [Circle(radius=0.1, degree=10),
                  Disk(radius=0.1, degree=10),
                  Square(P=lambda x0: x0-np.array([0.1, 0.1, 0]), degree=10),
                  SquareRim(P=lambda x0: x0-np.array([0.1, 0.1, 0]), degree=10)]:
        for n in (4, 8, 16, 32):
            assert is_close(identity_test(n, shape=shape))

# --------------------------------------------------------------------

def circle_test(n, subd=I_curve):
    '''Averaging over indep coords of f'''
    shape = Circle(radius=lambda x0: 0.1+0.0*x0[2]/2, degree=10)
    foo = df.Expression('x[2]*((x[0]-0.5)*(x[0]-0.5) + (x[1]-0.5)*(x[1]-0.5))', degree=3)
    
    mesh = df.UnitCubeMesh(n, n, n)
    V = df.FunctionSpace(mesh, 'CG', 3)
    v = df.interpolate(foo, V)

    f = df.MeshFunction('size_t', mesh, 1, 0)
    subd.mark(f, 1)

    true = df.Expression('x[2]*(0.1+0.0*x[2]/2)*(0.1+0.0*x[2]/2)', degree=4)
    
    line_mesh = EmbeddedMesh(f, 1)
    Q = average_space(V, line_mesh)
    q = df.Function(Q)
    
    Pi = avg_mat(V, Q, line_mesh, {'shape': shape})
    Pi.mult(v.vector(), q.vector())

    q0 = true
    # Error
    L = df.inner(q0 - q, q0 - q)*df.dx

    e = q.vector().copy()
    e.axpy(-1, df.interpolate(q0, Q).vector())
    
    return df.sqrt(abs(df.assemble(L)))

if True:
    for n in (4, 8, 16, 32): assert is_close(circle_test(n))
    
# --------------------------------------------------------------------
      
def disk_test(n, subd=I_curve):
    '''Averaging over indep coords of f'''
    shape = Disk(radius=lambda x0: 0.1+0.0*x0[2]/2, degree=10)
    foo = df.Expression('x[2]*((x[0]-0.5)*(x[0]-0.5) + (x[1]-0.5)*(x[1]-0.5))', degree=3)
    
    mesh = df.UnitCubeMesh(n, n, n)
    V = df.FunctionSpace(mesh, 'CG', 3)
    v = df.interpolate(foo, V)

    f = df.MeshFunction('size_t', mesh, 1, 0)
    subd.mark(f, 1)

    true = df.Expression('x[2]*(0.1+0.0*x[2]/2)*(0.1+0.0*x[2]/2)/2', degree=4)
    
    line_mesh = EmbeddedMesh(f, 1)
    Q = average_space(V, line_mesh)
    q = df.Function(Q)
    
    Pi = avg_mat(V, Q, line_mesh, {'shape': shape})
    Pi.mult(v.vector(), q.vector())

    q0 = true
    # Error
    L = df.inner(q0 - q, q0 - q)*df.dx

    e = q.vector().copy()
    e.axpy(-1, df.interpolate(q0, Q).vector())
    
    return df.sqrt(abs(df.assemble(L)))

if True:
    for n in (4, 8, 16, 32): assert is_close(disk_test(n))

# --------------------------------------------------------------------
    
def squarerim_test(n, subd=I_curve):
    '''Averaging over indep coords of f'''
    size = 0.1
    shape = SquareRim(P=lambda x0: x0 - np.array([size+size*x0[2], size+size*x0[2], 0]), degree=10)
    foo = df.Expression('x[2]*((x[0]-0.5)*(x[0]-0.5) + (x[1]-0.5)*(x[1]-0.5))', degree=3)
    
    mesh = df.UnitCubeMesh(n, n, n)
    V = df.FunctionSpace(mesh, 'CG', 3)
    v = df.interpolate(foo, V)

    f = df.MeshFunction('size_t', mesh, 1, 0)
    subd.mark(f, 1)

    true = df.Expression('x[2]*4./3*(size+size*x[2])*(size+size*x[2])', degree=4, size=size)
    
    line_mesh = EmbeddedMesh(f, 1)
    Q = average_space(V, line_mesh)
    q = df.Function(Q)
    
    Pi = avg_mat(V, Q, line_mesh, {'shape': shape})
    Pi.mult(v.vector(), q.vector())

    q0 = true
    # Error
    L = df.inner(q0 - q, q0 - q)*df.dx

    e = q.vector().copy()
    e.axpy(-1, df.interpolate(q0, Q).vector())
    
    return df.sqrt(abs(df.assemble(L)))

if True:
    for n in (4, 8, 16, 32): assert is_close(squarerim_test(n))

# --------------------------------------------------------------------
    
def square_test(n, subd=I_curve):
    '''Averaging over indep coords of f'''
    size = 0.1
    shape = Square(P=lambda x0: x0 - np.array([size+size*x0[2], size+size*x0[2], 0]), degree=10)
    foo = df.Expression('x[2]*((x[0]-0.5)*(x[0]-0.5) + (x[1]-0.5)*(x[1]-0.5))', degree=3)
    
    mesh = df.UnitCubeMesh(n, n, n)
    V = df.FunctionSpace(mesh, 'CG', 3)
    v = df.interpolate(foo, V)

    f = df.MeshFunction('size_t', mesh, 1, 0)
    subd.mark(f, 1)

    true = df.Expression('x[2]*2./3*(size+size*x[2])*(size+size*x[2])', degree=4, size=size)
    
    line_mesh = EmbeddedMesh(f, 1)
    Q = average_space(V, line_mesh)
    q = df.Function(Q)
    
    Pi = avg_mat(V, Q, line_mesh, {'shape': shape})
    Pi.mult(v.vector(), q.vector())

    q0 = true
    # Error
    L = df.inner(q0 - q, q0 - q)*df.dx

    e = q.vector().copy()
    e.axpy(-1, df.interpolate(q0, Q).vector())
    
    return df.sqrt(abs(df.assemble(L)))

for n in (4, 8, 16, 32): assert is_close(square_test(n))
