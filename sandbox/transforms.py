from __future__ import absolute_import
import numpy as np
import itertools
from six.moves import map
from six.moves import zip


def C(x0, n, R):
    '''
    Map unit circle in z = 0 to plane to circle of radius R with center at x0.
    '''
    n = n / np.linalg.norm(n)
    def transform(x, x0=x0, n=n, R=R):
        norm = np.dot(x, x)
        # Check assumptions
        assert abs(norm - 1) < 1E-13 and abs(x[2]) < 1E-13
        
        y = x - n*np.dot(x, n)
        y = y / np.sqrt(norm - np.dot(x, n)**2)
        return x0 + R*y

    return transform


def D(x0, n, R):
    '''
    Map unit disk in z = 0 to plane to disk of radius R with center at x0.
    '''
    n = n / np.linalg.norm(n)
    def transform(x, x0=x0, n=n, R=R):
        norm = np.dot(x, x)
        # Check assumptions
        assert norm < 1 + 1E-13 and abs(x[2]) < 1E-13

        y = x - n*np.dot(x, n)
        y = y / np.sqrt(norm - np.dot(x, n)**2)
        return x0 + R*np.sqrt(norm)*y

    return transform


def RB(x0, n, P):
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


def S(x0, n, P):
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

# --------------------------------------------------------------------

if __name__ == '__main__':
    from numpy.polynomial.legendre import leggauss
    import quadpy

    is_close = lambda x, y: np.linalg.norm(x-y) < 1E-10
    # TODO:
    # -[OK] unit circle
    # -[OK] unit disk
    # -[OK] reference square (surface)
    # - reference square (interior)

    # -- Circle ----------
    # There are mean at pi*angles
    xq, wq = leggauss(10)
    # The real quadrature points for the circle are
    xq = np.c_[np.cos(np.pi*xq), np.sin(np.pi*xq), np.zeros_like(xq)]

    x0 = np.array([1, 2, 3])
    n = np.array([1, 0, 0])
    R = 0.123

    T = C(x0, n, R)

    Txq = list(map(T, xq))

    # Are there points in the plane
    assert all(abs(np.dot(y - x0, n)) < 1E-13 for y in Txq)
    # They are points on the circle
    assert all(abs(np.dot(y - x0, y - x0) - R**2) < 1E-13 for y in Txq)

    # But then I can integrate
    def circle_quad(x0, n, R, degree):
        '''TODO'''
        xq, wq = leggauss(degree)
        xq = np.c_[np.cos(np.pi*xq), np.sin(np.pi*xq), np.zeros_like(xq)]
        # Circle viewed from reference
        Txq = list(map(C(x0, n, R), xq))
        # Scaled weights (R is jac of T, pi is from theta=pi*(-1, 1)
        wq *= R*np.pi

        return lambda f: sum(wqi*f(yi) for yi, wqi in zip(Txq, wq))

    quad = circle_quad(x0=x0, n=n, R=R, degree=10)
    
    # 1 gives circumnference
    one = lambda x: 1
    assert is_close(quad(one), 2*np.pi*R)

    # Something that is constant on the dist
    dist = lambda x: np.dot(x-x0, x-x0)
    assert is_close(quad(dist), 2*np.pi*R*R**2)

    # Zero by orthogonality
    null = lambda x: np.dot(x-x0, n)
    assert is_close(quad(null), 0)

    # -- Disk ----------
    quad = quadpy.disk.Lether(10)
    xq, wq = quad.points, quad.weights
    xq = np.c_[xq, np.zeros_like(wq)]

    T = D(x0=x0, n=n, R=R)
    Txq = list(map(T, xq))
    
    # Are there points in the plane
    assert all(abs(np.dot(y - x0, n)) < 1E-13 for y in Txq)
    # They are points on the circle
    assert all(np.dot(y - x0, y - x0) < R**2 + 1E-13 for y in Txq)

    # But then I can integrate
    def disk_quad(x0, n, R, degree):
        '''TODO'''
        quad = quadpy.disk.Lether(degree)
        xq, wq = quad.points, quad.weights
        xq = np.c_[xq, np.zeros_like(wq)]

        # Circle viewed from reference
        Txq = list(map(D(x0, n, R), xq))
        # Scaled weights (R is jac of T, pi is from theta=pi*(-1, 1)
        wq *= R**2

        return lambda f: sum(wqi*f(yi) for yi, wqi in zip(Txq, wq))

    quad = disk_quad(x0=x0, n=n, R=R, degree=10)

    # 1 gives area
    one = lambda x: 1
    assert is_close(quad(one), np.pi*R**2)

    # Zero by orthogonality
    null = lambda x: np.dot(x-x0, n)
    assert is_close(quad(null), 0)

    # Something harder
    dist = lambda x: np.dot(x-x0, x-x0)
    assert is_close(quad(dist), np.pi/2*R**4)
    
    dist = lambda x: np.dot(x-x0, x-x0)**2
    assert is_close(quad(dist), np.pi/3*R**6)

    # -- Rectangle boundary ----------
    # NOTE: things below are wired for n[0] == n[1]
    w = 0.123
    shift = w*np.array([0.2, 0.3, 0.4])
    P = x0 + shift
    Q = x0 + np.array([0.1, 0.6, -0.4])
    n = np.cross(P-x0, Q-x0)
    n = n / np.linalg.norm(n)
    
    rect_bdry = RB(x0, n, P)
    # Corner points are in the plane
    corners = rect_bdry(-1)
    assert len(corners) == 4
    assert all(abs(np.dot(P - v, n)) < 1E-13 for v in corners)

    W = np.linalg.norm(shift)
    # Their distance to center
    d, = set(np.round(np.linalg.norm(x0 - v), 13) for v in corners)
    assert abs(d - W) < 1E-13, (d, W)
    
    # Their distance to each
    d, = set(np.round(np.linalg.norm(x-y), 13) for x, y in zip(corners, corners[1:] + [corners[0]]))
    assert abs(d - np.sqrt(2)*W) < 1E-13

    # Now I can integrate
    def square_bdry_quad(x0, n, P, degree):
        '''TODO'''
        xq, wq = leggauss(degree)

        sq_bdry = RB(x0, n, P)
        corners = sq_bdry(-1)
        A, B = corners[:2]
        size = 0.5*np.linalg.norm(B-A)
        # Scale the weights
        wq *= size

        Txq = sum(list(map(sq_bdry, xq)), [])
        
        return lambda f: sum(wqi*f(yi) for yi, wqi in zip(Txq, np.repeat(wq, 4)))

    quad = square_bdry_quad(x0, n, P, degree=8)
    
    # 1 gives circumnference
    one = lambda x: 1
    assert is_close(quad(one), 4*np.sqrt(2)*W)

    # Zero by orthogonality
    null = lambda x: np.dot(x-x0, n)
    assert is_close(quad(null), 0)

    # Something harder
    dist = lambda x: np.dot(x-x0, x-x0)
    assert is_close(quad(dist), 4*8*(np.sqrt(2)*W/2)**3/3.)
    
    # -- Square surface ----------
    rect = S(x0, n, P)
    # Corner points are in the plane
    corners = list(map(rect, np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])))

    assert all(abs(np.dot(P - v, n)) < 1E-13 for v in corners)

    # Their distance to center
    d, = set(np.round(np.linalg.norm(x0 - v), 13) for v in corners)
    assert abs(d - W) < 1E-13, (d, W)
    
    # Their distance to each
    d, = set(np.round(np.linalg.norm(x-y), 13) for x, y in zip(corners, corners[1:] + [corners[0]]))
    assert abs(d - np.sqrt(2)*W) < 1E-13

    # Now I can integrate
    def square_quad(x0, n, P, degree):
        '''TODO'''
        xq, wq = leggauss(degree)
        
        sq = S(x0, n, P)
        # 1D
        A, B = sq(np.array([-1, -1])), sq(np.array([-1, 1]))
        size = 0.5*np.linalg.norm(B-A)
        # Scale the weights
        wq *= size

        # 2D
        wq = list(map(np.prod, itertools.product(wq, wq)))
        
        xq = list(map(np.array, itertools.product(xq, xq)))
        Txq = list(map(sq, xq))
        
        return lambda f: sum(wqi*f(yi) for yi, wqi in zip(Txq, wq))

    quad = square_quad(x0, n, P, degree=10)
    # 1 gives area
    one = lambda x: 1
    assert is_close(quad(one), 2*W**2)

    # Zero by orthogonality
    null = lambda x: np.dot(x-x0, n)
    assert is_close(quad(null), 0)

    # Something harder
    dist = lambda x: np.dot(x-x0, x-x0)
    assert is_close(quad(dist), 8*(np.sqrt(2)*W/2)**4/3.)
