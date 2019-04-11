from abc import ABCMeta, abstractmethod
from xii.linalg.matrix_utils import is_number
from xii.assembler.average_form import average_space
from numpy.polynomial.legendre import leggauss
from itertools import product
from math import pi
import numpy as np
import quadpy


class BoundingCurve:
    '''API of a bounding curve in plane with normal n'''
    __metaclass__ = ABCMeta
    # NOTE: lenth and area a not the true measures of the curve. They
    # depend on how integration is done; in particular cancellation
    # due to mapping to reference domain
    @abstractmethod
    def weights(self):
        '''Quadrature weights for surface integral averaging'''
        pass

    @abstractmethod
    def length(self, n):
        '''\(x) -> length of averaging curve in plane with normal n'''
        pass
    
    @abstractmethod
    def points(self, n):
        '''
        \(x) -> quadrature points for surface averaging over curve in plane 
        with normal n'''
        pass
    
    @abstractmethod
    def cross_weights(self):
        '''Quadrature weights for cross section integral averaging'''
        pass

    @abstractmethod
    def area(self, n):
        '''\(x) -> area of averaging surface in plane with normal n'''
        pass

    @abstractmethod
    def cross_points(self, n):
        '''
        \(x) -> quadrature points for cross surface averaging over curve 
        in plane with normal n
        '''
        pass


class Square(BoundingCurve):
    '''
    Is specified by normal (will be given by 1d mesh) and functions:
    P(x\in R^3) -> R^3 the position of the ll corner
    '''
    def __init__(self, P, degree):
        if isinstance(P, (tuple, list, np.ndarray)):
            assert all(is_number(Pi) for Pi in P)
            self.P = lambda x0, p=P: p
        else:
            self.P = P

        xq, wq = leggauss(degree)
        # Point for edge -1 -- 1        
        self.__wq__ = wq  
        self.__xq__ = xq

    # Surface averaging
    #
    # NOTE: we want (*) 1/|sq|*\int_{sq} f*dl, len e be the square edge
    # then (*) = 1/4/|e|*\sum_e \int_{e} f*dl
    #          = 1/4/|e|*\sum_e \int_{A}^B f*dl
    #          = 1/4/|e|*\sum_e \int_{-1}^1 f(0.5*A*(1-s)+0.5*B*(1+s))*(0.5*|e|)dl
    #          = 1/4 * \sum_e \sum_q 0.5*wq f(0.5*A*(1-sq)+0.5*B*(1+sq))
    #
    # In the API of shape_surface_averege 0.5*wq are weights,
    #                                     4 has the role of length
    #                                     xq are clear
    def length(self, n):
        '''Length in plane with normal n'''
        # NOTE: Due to the cancellation above (wq is included in weight)
        # the length is 1 for the purpose of integration
        def circumnference(x0, n=n, me=self):
            return 4.
        return circumnference

    def weights(self):
        '''Quad weights'''
        # Repeat for 5 edges
        not hasattr(self, '__surf_weights__') and setattr(self, '__surf_weights__', 0.5*np.tile(self.__wq__, 4))
        return self.__surf_weights__

    def points(self, n):
        '''Quad points for square'''
        n = n/np.linalg.norm(n)

        def pts(x0, n=n, me=self):
            r = me.square(x0, n, me.P(x0))
            return me.square_xq(r)

        return pts

    def square_xq(self, (A, B, C, D)):
        '''Quad points for rectangle A--B--D--D--'''
        xq = []
        for X, Y in zip((A, B, C, D), (B, C, D, A)):
            xq.extend([0.5*X*(1 - s) + 0.5*Y*(1 + s) for s in self.__xq__])
        return xq

    def square(self, x0, n, P):
        '''
        Make square with center x0 in plane with normal n and other 
        vertices obtained by rotating (P-x0)
        '''
        # n = n/np.linalg.norm(n)
        #  P_       Pr
        # 
        #      x0
        # P         P+
        vec = P - x0
        Pm = x0 + vec*np.cos(pi/2) + np.cross(n, vec)*np.sin(pi/2) + n*(n.dot(vec))*(1-np.cos(pi/2))
        Pr = x0 - (P-x0)
        Pp = x0 + vec*np.cos(3*pi/2) + np.cross(n, vec)*np.sin(3*pi/2) + n*(n.dot(vec))*(1-np.cos(3*pi/2))

        return (P, Pm, Pr, Pp)

    # Cross section averaging
    # We want 1/|sq||*\int_{sq} f*dS
    def cross_weights(self):
        '''Quadrature weights for cross section integral averaging'''
        # Precompute
        if not hasattr(self, '__cross_weights__'):
            cross_weights = np.fromiter(map(np.prod, product(self.__wq__, self.__wq__)), 
                                        dtype=float)
            setattr(self, '__cross_weights__', cross_weights)
        return self.__cross_weights__
        
    def area(self, n):
        '''\(x) -> area of averaging surface in plane with normal n'''
        return (lambda x: 4)

    def cross_points(self, n):
        '''
        \(x) -> quadrature points for cross surface averaging over curve 
        in plane with normal n
        '''
        # Precompute
        if not hasattr(self, '__cross_points__'):
            # Want them as two column vectors for (s, t) reference coordinates
            s, t = np.fromiter(sum(product(self.__xq__, self.__xq__), ()),
                               dtype=float).reshape((-1, 2)).T
            s, t = np.array([s]).T, np.array([t]).T
            setattr(self, '__cross_points__', (s, t))

        s, t = self.__cross_points__
        n = n/np.linalg.norm(n)

        def pts(x0, n=n, me=self, s=s, t=t):
            A, B, C, D = me.square(x0, n, me.P(x0))
            return A + 0.5*(B-A)*(s+1) + 0.5*(D-A)*(t+1)

        return pts

    
class Cicle(BoundingCurve):
    '''Obtain the bounding surface by making a circle of radius r in the normal plane'''
    def __init__(self, radius, degree):
        # Make constant function
        if is_number(radius):
            assert radius > 0
            self.radius = lambda x0, r=radius: r
        # Then this must map points on centerline to radius
        else:
            self.radius = radius
            
        xq, wq = leggauss(degree)
        self.__weights__ = wq*0.5  # Scale down by 0.5

        # Precompute trigonometric part (only shift by tangents t1, t2)
        self.cos_xq = np.cos(np.pi*xq).reshape((-1, 1))
        self.sin_xq = np.sin(np.pi*xq).reshape((-1, 1))

        # Points for the unit disk
        quad = quadpy.disk.Lether(degree)
        self.__cross_weights__  = quad.weights
        # Unit circle in plane z = 0
        self.__cross_points__ = np.c_[quad.points, np.zeros_like(self.__cross_weights__)]
        
    # Surface averaging
    # NOTE: Let s by the arch length coordinate of the centerline of the 
    # cylinder. A tangent to 1d mesh at s is a normal to the plane in which
    # we draw a circle C(n, r) with radius r. For reduction of 3d we compute
    # 
    # |2*pi*R(s)|^-1\int_{C(n, r)} u dl =
    # 
    # |2*pi*R(s)|^-1\int_{-pi}^{pi} u(s + t1*R(s)*cos(theta) + t2*R(s)*sin(theta))R*d(theta) = 
    #
    # |2*pi*R(s)|^-1\int_{-1}^{1} u(s + t1*R(s)*cos(pi*t) + t2*R(s)*sin(pi*t))*pi*R*dt = 
    #
    # 1/2*sum_q u(s + t1*R(s)*cos(pi*x_q) + t2*R(s)*sin(pi*x_q))
    # 
    @property
    def weights(self):
        '''Quad weights'''
        return self.__weights__

    def points(self, n):
        '''Quad points for circle(x0, radius(x0)) in plane with normal n'''
        # Fix plane
        t1 = np.array([n[1]-n[2], n[2]-n[0], n[0]-n[1]])
    
        t2 = np.cross(n, t1)
        t1 = t1/np.linalg.norm(t1)
        t2 = t2/np.linalg.norm(t2)

        def pts(x0, me=self, t1=t1, t2=t2):
            rad = self.radius(x0)
            return x0 + rad*t1*me.sin_xq + rad*t2*me.cos_xq

        return pts

    def length(self, n):
        '''Length in plane with normal n'''
        # NOTE: Due to the cancellation above (wq is included in weight)
        # the length is 1 for the purpose of integration
        return lambda x: 1
    #
    # Cross section averaging
    # Do this by mapping to unit circle
    def cross_weights(self):
        '''Quadrature weights for cross section integral averaging'''
        return self.__cross_weights__
        
    def area(self, n):
        '''\(x) -> area of averaging surface in plane with normal n'''
        return (lambda x: pi)

    def cross_points(self, n):
        '''
        \(x) -> quadrature points for cross surface averaging over curve 
        in plane with normal n
        '''
        # One unit disk
        x = self.__cross_points__
        
        n = n/np.linalg.norm(n)
        # The tilter disk, to plane with normal n
        y = np.array([xi - n*np.dot(xi, n) for xi in x])

        # What remains here to scale according to radius and shift
        def pts(x0, me=self, y=y):
            rad = self.radius(x0)
            return x0 + rad*y

        return pts


# Testing utils
def render_avg_surface(Pi, which='surface'):
    '''Plot the averaging surface via looking at the quadrature points used'''
    V = Pi.function_space()
    line_mesh = Pi.average_['mesh']
    # Where the average will be represented
    Pi_V = average_space(V, line_mesh)

    curve = Pi.average_['bdry_curve']

    # We produce a curve of quardrature points for each dof
    surface = []
    
    dm = Pi_V.dofmap()
    dofs_x = Pi_V.tabulate_dof_coordinates().reshape((Pi_V.dim(), -1))
    for cell in df.cells(line_mesh):
        v0, v1 = cell.get_vertex_coordinates().reshape((2, 3))
        n = v1 - v0

        pts = (curve.points if which == 'surface' else curve.cross_points)(n=n)
        for dof_x in dofs_x[dm.cell_dofs(cell.index())]:
            x = np.row_stack(pts(dof_x))
            surface.append(x)

    return surface
        
# --------------------------------------------------------------------

if __name__ == '__main__':
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt

    from xii.assembler.average_form import Average
    from xii.meshing.embedded_mesh import EmbeddedMesh
    import dolfin as df

    # Get the MEAN
    mesh = df.BoxMesh(df.Point(-1, -1, -1), df.Point(1, 1, 1), 16, 16, 16)
    # Make 1d
    f = df.MeshFunction('size_t', mesh, 1, 0)
    # df.CompiledSubDomain('near(x[0], x[1]) && near(x[1], x[2])').mark(f, 1)
    df.CompiledSubDomain('near(x[0], 0.) && near(x[1], 0.)').mark(f, 1)
    
    line_mesh = EmbeddedMesh(f, 1)

    # Setup bounding curve
    size = 0.125
    ci = Cicle(radius=lambda x0: size, degree=8)

    u = df.Function(df.FunctionSpace(mesh, 'CG', 1))
    op = Average(u, line_mesh, ci)
        
    surface = render_avg_surface(op, which='cross')
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # NOTE: normal matches and point is on line mesh
    def ci_integrate(f, n=np.array([0, 0, 1]), x0=np.array([0, 0, 0.5])):
        return sum(wq*f(*xq)/(ci.area(None)(None))
                   for wq, xq in zip(ci.cross_weights(), ci.cross_points(n)(x0)))
    # ---
    f = lambda x, y, z: 1
    value = ci_integrate(f)
    assert abs(value - 1) < 1E-13

    # Odd foo over sym interval
    f = lambda x, y, z: x - y - 0.5
    value = ci_integrate(f)
    assert abs(value - -0.5) < 1E-13

    # Odd foo over sym interval
    f = lambda x, y, z: x**3 - y - z
    value = ci_integrate(f)
    assert abs(value - -0.5) < 1E-13

    #
    f = lambda x, y, z: x**2 + y**2 - z**2
    value = ci_integrate(f)
    assert abs(value - (size**2/2. - 0.5**2)) < 1E-13

    
    for plane in surface:
        ax.plot3D(plane[:, 0], plane[:, 1], plane[:, 2], marker='o', linestyle='none')

    # plt.show()
    # Square
    if False:
        mesh = df.BoxMesh(df.Point(-1, -1, -1), df.Point(1, 1, 1), 16, 16, 16)
        # Make 1d
        f = df.MeshFunction('size_t', mesh, 1, 0)
        df.CompiledSubDomain('near(x[0], 0) && near(x[1], 0)').mark(f, 1)
    
        line_mesh = EmbeddedMesh(f, 1)

        # Setup bounding curve
        size = 0.125
        sq = Square(P=lambda x0: np.array([size*np.cos(0.5*pi*x0[2]),
                                           size*np.sin(0.5*pi*x0[2]),
                                           x0[2]]),
                    degree=8)

        u = df.Function(df.FunctionSpace(mesh, 'CG', 1))
        op = Average(u, line_mesh, sq)
        
        surface = render_avg_surface(op)
    
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        for plane in surface:
            ax.plot3D(plane[:, 0], plane[:, 1], plane[:, 2], marker='o')

        # Setup bounding curve
        size = 0.15
        sq = Square(P=lambda x0: np.array([x0[0]-size,
                                           x0[1]-size,
                                           x0[2]]),
                    degree=8)

        u = df.Function(df.FunctionSpace(mesh, 'CG', 1))
        op = Average(u, line_mesh, sq)

        surface = render_avg_surface(op, which='cross')
    
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for plane in surface:
            ax.plot3D(plane[:, 0], plane[:, 1], plane[:, 2], marker='o')

        plt.show()

        f = lambda x, y, z: x + 2*y

        value = sum(wq*f(*xq)/(sq.area(None)(None))
                    for wq, xq in zip(sq.cross_weights(),
                                      sq.cross_points(np.array([0, 0, 1]))(np.array([0, 0, 0]))))
        assert abs(value - 0) < 1E-13


        f = lambda x, y, z: 1

        value = sum(wq*f(*xq)/(sq.area(None)(None))
                    for wq, xq in zip(sq.cross_weights(),
                                      sq.cross_points(np.array([0, 0, 1]))(np.array([0, 0, 0]))))
        assert abs(value - 1) < 1E-13

    
        f = lambda x, y, z: x**2

        value = sum(wq*f(*xq)/(sq.area(None)(None))
                    for wq, xq in zip(sq.cross_weights(),
                                      sq.cross_points(np.array([0, 0, 1]))(np.array([0, 0, 0]))))
        assert abs(value - size**2/3.) < 1E-13

        
        f = lambda x, y, z: x**2 + 3*y**2

        value = sum(wq*f(*xq)/(sq.area(None)(None))
                    for wq, xq in zip(sq.cross_weights(),
                                      sq.cross_points(np.array([0, 0, 1]))(np.array([0, 0, 0]))))
    
        assert abs(value - (size**2/3. + size**2)) < 1E-13


        f = lambda x, y, z: x**2 + 3*y**2 - x*y

        value = sum(wq*f(*xq)/(sq.area(None)(None))
                    for wq, xq in zip(sq.cross_weights(),
                                      sq.cross_points(np.array([0, 0, 1]))(np.array([0, 0, 0]))))
    
        assert abs(value - (size**2/3. + size**2)) < 1E-13
