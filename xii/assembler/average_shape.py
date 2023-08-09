from abc import ABCMeta, abstractmethod
from collections import namedtuple
from itertools import product
from math import pi, sqrt
import numpy as np

# import quadpy

from numpy.polynomial.legendre import leggauss
import dolfin as df

from xii.linalg.matrix_utils import is_number
from xii.assembler.average_form import average_space
from xii.meshing.make_mesh_cpp import make_mesh

Quadrature = namedtuple('quadrature', ('points', 'weights'))


class BoundingSurface(metaclass=ABCMeta):
    '''Shape used for reducing a 3d function to 1d by carrying out integration'''
    
    @abstractmethod
    def quadrature(self, x0, n):
        '''Quadrature weights and points for reduction'''
        pass

    
class Square(BoundingSurface):
    r'''
    Square in plane(x0, n) with ll corner given by P(x\in R^3) -> R^3
    '''
    def __init__(self, P, degree):
        if isinstance(P, (tuple, list, np.ndarray)):
            assert all(is_number(Pi) for Pi in P)
            self.P = lambda x0, p=P: p
        else:
            self.P = P

        # Weights for [-1, 1] for 2d will do the tensor product
        self.xq, self.wq = leggauss(degree)

    @staticmethod
    def map_from_reference(x0, n, P):
        '''A map of [-1, 1]x[-1, 1] to square(x0, n, P).'''
        n = n / np.linalg.norm(n)
        # C   B
        #  x0
        # P   A
        vec = P - x0
        # We are in the place
        assert abs(np.dot(vec, n)) < 1E-13
    
        A = x0 + vec*np.cos(np.pi/2) + np.cross(n, vec)*np.sin(np.pi/2) + n*(n.dot(vec))*(1-np.cos(np.pi/2))
        C = x0 + vec*np.cos(3*np.pi/2) + np.cross(n, vec)*np.sin(3*np.pi/2) + n*(n.dot(vec))*(1-np.cos(3*np.pi/2))

        def mapping(x, P=P, u=A-P, v=C-P):
            x, y = x
            assert abs(x) < 1 + 1E-13 and abs(y) < 1 + 1E-13
            return P + 0.5*u*(1+x) + 0.5*v*(1+y) 
    
        return mapping

    def quadrature(self, x0, n):
        '''Gaussian qaudrature over the surface of the square'''
        xq, wq = self.xq, self.wq
        
        sq = Square.map_from_reference(x0, n, self.P(x0))
        # 1D
        A, B = sq(np.array([-1, -1])), sq(np.array([-1, 1]))
        size = 0.5*np.linalg.norm(B-A)
        # Scale the weights
        wq = wq*size

        # 2D
        wq = list(map(np.prod, product(wq, wq)))
        
        xq = list(map(np.array, product(xq, xq)))
        Txq = list(map(sq, xq))
        
        return Quadrature(Txq, wq)


class SquareRim(BoundingSurface):
    r'''
    Boundary of a square in plane(x0, n) with ll corner given by 
    P(x\in R^3) -> R^3
    '''
    def __init__(self, P, degree):
        if isinstance(P, (tuple, list, np.ndarray)):
            assert all(is_number(Pi) for Pi in P)
            self.P = lambda x0, p=P: p
        else:
            self.P = P

        # Weights for [-1, 1] for 2d will do the tensor product
        self.xq, self.wq = leggauss(degree)

    @staticmethod
    def map_from_reference(x0, n, P):
        '''
        A map of [-1, 1] to 4 points on the boundary of the square(x0, n, P)
        '''
        # Rectangle boundary mapping
        n = n / np.linalg.norm(n)
        # C   B
        #  x0
        # P   A
        vec = P - x0
        # We are in the place
        assert abs(np.dot(vec, n)) < 1E-13
        
        pts = [P,
               x0 + vec*np.cos(np.pi/2) + np.cross(n, vec)*np.sin(np.pi/2) + n*(n.dot(vec))*(1-np.cos(np.pi/2)),
               x0 - (P-x0),
               x0 + vec*np.cos(3*np.pi/2) + np.cross(n, vec)*np.sin(3*np.pi/2) + n*(n.dot(vec))*(1-np.cos(3*np.pi/2))]

        def mapping(x, pts=pts):
            assert abs(x) < 1 + 1E-13
            return [0.5*P*(1-x) + 0.5*Q*(1+x) for P, Q in zip(pts, pts[1:]+[pts[0]])]
    
        return mapping

    def quadrature(self, x0, n):
        '''Gaussian qaudrature over boundary of the square'''
        xq, wq = self.xq, self.wq

        sq_bdry = SquareRim.map_from_reference(x0, n, self.P(x0))
        corners = sq_bdry(-1)
        A, B = corners[:2]
        size = 0.5*np.linalg.norm(B-A)
        # Scale the weights
        wq = wq*size
        # One for each side
        wq = np.repeat(wq, 4)

        Txq = sum(list(map(sq_bdry, xq)), [])

        return Quadrature(Txq, wq)


class Circle(BoundingSurface):
    '''Circle in plane(x0, n) with radius given by radius(x0)'''
    def __init__(self, radius, degree):
        # Make constant function
        if is_number(radius):
            assert radius > 0
            self.radius = lambda x0, r=radius: r
        # Then this must map points on centerline to radius
        else:
            self.radius = radius

        # Will use Gauss quadrature on [-1, 1]
        self.xq, self.wq = leggauss(degree)

    @staticmethod
    def map_from_reference(x0, n, R):
        '''
        Map unit circle in z = 0 to plane to circle of radius R with center at x0.
        '''
        ez = np.array([0., 0., 1.])
        n = n / np.linalg.norm(n)
        # The idea here is to rotatate our z plane passing through origin
        # to the one with normal  n. First thing to do is to figure out
        # the axis along which we will rotate ...
        axis = np.cross(ez, n)
        # ... and by how much
        ctheta = np.dot(ez, n)
        stheta = np.sqrt(1 - ctheta**2)
        # Rotation matrix
        Rot = ctheta*np.eye(3) + stheta*np.array([[0, -axis[2], axis[1]],
                                                  [axis[2], 0, -axis[0]],
                                                  [-axis[1], axis[0], 0]]) + (1-ctheta)*np.outer(axis, axis)
        def transform(x, x0=x0, n=n, R=R, Rot=Rot):
            norm = np.dot(x, x)
            # Check assumptions
            assert abs(norm - 1) < 1E-13 and abs(x[2]) < 1E-13

            y = Rot@x
            # And then we just shift the origin
            return x0 + R*y

        return transform

    def quadrature(self, x0, n):
        '''Gauss quadratature over the boundary of the circle'''
        xq, wq = self.xq, self.wq
        xq = np.c_[np.cos(np.pi*xq), np.sin(np.pi*xq), np.zeros_like(xq)]

        R = self.radius(x0)
        # Circle viewed from reference
        Txq = list(map(Circle.map_from_reference(x0, n, R), xq))
        # Scaled weights (R is jac of T, pi is from theta=pi*(-1, 1)
        wq = wq*R*np.pi

        return Quadrature(Txq, wq)


class Disk(BoundingSurface):
    '''Disk in plane(x0, n) with radius given by radius(x0)'''    
    def __init__(self, radius, degree):
        # Make constant function
        if is_number(radius):
            assert radius > 0
            self.radius = lambda x0, r=radius: r
        # Then this must map points on centerline to radius
        else:
            self.radius = radius

        # Will use quadrature from quadpy over unit disk in z=0 plane
        # and center (0, 0, 0)
        quad = quadpy.disk.Lether(degree)
        self.xq, self.wq = quad.points, quad.weights

    @staticmethod
    def map_from_reference(x0, n, R):
        '''
        Map unit disk in z = 0 to plane to disk of radius R with center at x0.
        '''
        ez = np.array([0., 0., 1.])
        n = n / np.linalg.norm(n)
        # The idea here is to rotatate our z plane passing through origin
        # to the one with normal  n ...
        axis = np.cross(ez, n)
        ctheta = np.dot(ez, n)
        stheta = np.sqrt(1 - ctheta**2)
        # Rotation matrix
        Rot = ctheta*np.eye(3) + stheta*np.array([[0, -axis[2], axis[1]],
                                                  [axis[2], 0, -axis[0]],
                                                  [-axis[1], axis[0], 0]]) + (1-ctheta)*np.outer(axis, axis)
        
        def transform(x, x0=x0, n=n, R=R, Rot=Rot):
            norm = np.dot(x, x)
            # Check assumptions
            assert norm < 1 + 1E-13 and abs(x[2]) < 1E-13

            x_ = x/norm
            y = Rot@x
            return x0 + R*y
            
        return transform

    def quadrature(self, x0, n):
        '''Quadrature for disk(center x0, normal n, radius x0)'''
        xq, wq = self.xq, self.wq
        
        xq = np.c_[xq, np.zeros_like(wq)]

        R = self.radius(x0)
        # Circle viewed from reference
        Txq = list(map(Disk.map_from_reference(x0, n, R), xq))
        # Scaled weights (R is jac of T, pi is from theta=pi*(-1, 1)
        wq = wq*R**2

        return Quadrature(Txq, wq)

    
# Testing utils
def render_avg_surface(Pi):
    '''Plot the averaging surface via looking at the quadrature points used'''
    V = Pi.function_space()

    line_mesh = Pi.average_['mesh']
    shape = Pi.average_['shape']
        
    # Where the average will be represented
    Pi_V = average_space(V, line_mesh)

    # We produce a curve of quardrature points for each dof
    surface = []
    
    dm = Pi_V.dofmap()
    dofs_x = np.array(Pi_V.tabulate_dof_coordinates()).reshape((Pi_V.dim(), -1))
    for cell in df.cells(line_mesh):
        v0, v1 = np.array(cell.get_vertex_coordinates()).reshape((2, 3))
        n = v1 - v0

        for dof_x in dofs_x[dm.cell_dofs(cell.index())]:
            x = np.row_stack(shape.quadrature(dof_x, n).points)
            surface.append(x)

    return surface


def tube_render_avg_surface(Pi):
    '''Set things for tube filter.'''
    V = Pi.function_space()

    line_mesh = Pi.average_['mesh']
    shape = Pi.average_['shape']
    # Where the average will be represented
    Pi_V = average_space(V, line_mesh)

    # We want points cells and point values
    tubes = []
    for cell in df.cells(line_mesh):
        v0, v1 = np.array(cell.get_vertex_coordinates()).reshape((2, 3))
        n = v1 - v0

        tubes.append((np.row_stack(shape.quadrature(v0, n).points),
                      np.row_stack(shape.quadrature(v1, n).points)))

    return tubes

# --------------------------------------------------------------------

if __name__ == '__main__':
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt

    from xii.assembler.average_form import Average
    from xii.meshing.embedded_mesh import EmbeddedMesh
    import dolfin as df

    
    def is_close(a, b, tol=1E-8): return np.linalg.norm(a - b) < tol

    def shape_integrate(f, shape, x0, n):
        pts, weights = shape.quadrature(x0, n)
        l = sum(weights)
        return sum(wq*f(*xq) for xq, wq in zip(pts, weights))/l

    # Get the MEAN
    mesh = df.BoxMesh(df.Point(-1, -1, -1), df.Point(1, 1, 1), 16, 16, 16)
    # Make 1d
    f = df.MeshFunction('size_t', mesh, 1, 0)
    # df.CompiledSubDomain('near(x[0], x[1]) && near(x[1], x[2])').mark(f, 1)
    df.CompiledSubDomain('near(x[0], 0.) && near(x[1], 0.)').mark(f, 1)
    df.CompiledSubDomain('near(x[1], 0.) && near(x[2], 0.)').mark(f, 2)    
    
    # line_mesh = EmbeddedMesh(f, 1)
    line_mesh = EmbeddedMesh(f, 2)    

    # Circle ---------------------------------------------------------
    size = 0.125
    ci = Circle(radius=lambda x0: size, degree=12)

    u = df.Function(df.FunctionSpace(mesh, 'CG', 1))
    op = Average(u, line_mesh, ci)
        
    surface = render_avg_surface(op)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x0 = np.array([0, 0, 0.5])
    n = np.array([1, 0, 0])
    ci_integrate = lambda f, shape=ci, n=n, x0=x0: shape_integrate(f, shape, x0, n)


    
    # Sanity
    f = lambda x, y, z: 1
    value = ci_integrate(f)
    # assert is_close(value, 1)

    # Odd foo over sym interval
    f = lambda x, y, z: x - y - 0.5
    value = ci_integrate(f)
    # assert is_close(value, -0.5)

    # Odd foo over sym interval
    f = lambda x, y, z: x**3 - y - z
    value = ci_integrate(f)
    # assert is_close(value, -0.5)

    # Something that is constant on the dist
    dist = lambda x, y, z: np.dot(np.array([x, y, z])-x0, np.array([x, y, z])-x0)
    # assert is_close(ci_integrate(dist), 2*np.pi*size*size**2/(2*pi*size))

    # Zero by orthogonality
    null = lambda x, y, z: np.dot(np.array([x, y, z])-x0, n)
    # assert is_close(ci_integrate(null), 0.)

    f = lambda x, y, z: x**2 + y**2 - z**2
    value = ci_integrate(f)
    # assert is_close(value, (size**2 - 0.5**2)), (value, size**2 - 0.5**2)

    for plane in surface:
        ax.plot3D(plane[:, 0], plane[:, 1], plane[:, 2], marker='o', linestyle='none')
    plt.show()
    
    # Square ---------------------------------------------------------
    size = 0.125
    sq = SquareRim(P=lambda x0: x0 - np.array([size, size, 0]), degree=8)

    u = df.Function(df.FunctionSpace(mesh, 'CG', 1))
    op = Average(u, line_mesh, sq)
        
    surface = render_avg_surface(op)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
        
    for plane in surface:
        ax.plot3D(plane[:, 0], plane[:, 1], plane[:, 2], marker='o', linestyle='none')

    sq_integrate = lambda f, shape=sq, n=n, x0=x0: shape_integrate(f, shape, x0, n)
    # Sanity
    f = lambda x, y, z: 1
    value = sq_integrate(f)
    assert is_close(value, 1)

    # Odd foo over sym interval
    f = lambda x, y, z: x - y - 0.5
    value = sq_integrate(f)
    assert is_close(value, -0.5), (value, )

    # Odd foo over sym interval
    f = lambda x, y, z: x**3 - y - z
    value = sq_integrate(f)
    assert is_close(value, -0.5)

    # Zero by orthogonality
    null = lambda x, y, z: np.dot(np.array([x, y, z])-x0, n)
    assert is_close(sq_integrate(null), 0.)

    W = np.linalg.norm(np.array([size, size, 0]))
    # Something harder
    dist = lambda x, y, z: np.dot(np.array([x, y, z]) - x0, np.array([x, y, z])-x0)
    assert is_close(sq_integrate(dist), 4*8*(np.sqrt(2)*W/2)**3/3.)

    # Disk ---------------------------------------------------------
    R = 0.125
    di = Disk(radius=lambda x0: R, degree=12)

    u = df.Function(df.FunctionSpace(mesh, 'CG', 1))
    op = Average(u, line_mesh, di)
        
    surface = render_avg_surface(op)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
        
    for plane in surface:
        ax.plot3D(plane[:, 0], plane[:, 1], plane[:, 2], marker='o', linestyle='none')

    di_integrate = lambda f, shape=di, n=n, x0=x0: shape_integrate(f, shape, x0, n)
    # Sanity
    f = lambda x, y, z: 1
    value = sq_integrate(f)
    assert is_close(value, 1)

    # Zero by orthogonality
    null = lambda x, y, z: np.dot(np.array([x, y, z])-x0, n)
    assert is_close(di_integrate(null), 0)

    # Something harder
    dist = lambda x, y, z: np.dot(np.array([x, y, z])-x0, np.array([x, y, z])-x0)
    assert is_close(di_integrate(dist), np.pi/2*R**4/(np.pi*R**2))
    
    dist = lambda x, y, z: np.dot(np.array([x, y, z])-x0, np.array([x, y, z])-x0)**2
    assert is_close(di_integrate(dist), np.pi/3*R**6/(np.pi*R**2))

    # Square ---------------------------------------------------------
    size = 0.125
    sq = Square(P=lambda x0: x0 - np.array([size, size, 0]), degree=8)

    u = df.Function(df.FunctionSpace(mesh, 'CG', 1))
    op = Average(u, line_mesh, sq)


    from scipy.spatial import Delaunay
    from dolfin import File
        
    surface = render_avg_surface(op)

    nodes = np.row_stack(surface)
    tri = Delaunay(nodes)

    cells = np.fromiter(tri.simplices.flatten(), dtype='uintp').reshape(tri.simplices.shape)
    
    bounded_volume = make_mesh(nodes, cells, tdim=2, gdim=3)
    File('foo.pvd') << bounded_volume
    
    # for points in surface
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
        
    for plane in surface:
        ax.plot3D(plane[:, 0], plane[:, 1], plane[:, 2], marker='o', linestyle='none')
        
    sq_integrate = lambda f, shape=sq, n=n, x0=x0: shape_integrate(f, shape, x0, n)

    # Sanity
    one = lambda x, y, z: 1
    assert is_close(sq_integrate(one), 1)

    # Zero by orthogonality
    null = lambda x, y, z: np.dot(np.array([x, y, z])-x0, n)
    assert is_close(sq_integrate(null), 0)

    W = np.linalg.norm([size, size, 0])
    # Something harder
    area = 2*W**2
    dist = lambda x, y, z: np.dot(np.array([x, y, z])-x0, np.array([x, y, z])-x0)
    assert is_close(sq_integrate(dist), 8*(np.sqrt(2)*W/2)**4/3./area)

    plt.show()
