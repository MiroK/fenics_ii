from xii.linalg.matrix_utils import is_number
from numpy.polynomial.legendre import leggauss
import numpy as np


# Some predefined shapes for surfaces averaging
class Cylinder(object):
    '''Obtain the shape by making a circle of radius r in the normal plane'''
    def __init__(self, radius, degree):
        # Make constant function
        if is_number(radius):
            assert radius > 0
            self.radius = lambda x0, r=radius: r
        # Then this must map points on centerline to radius
        else:
            self.radius = radius
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
        
        xq, wq = leggauss(degree)
        self.__weights__ = wq*0.5  # Scale down by 0.5

        # Precompute trigonometric part (only shift by tangents t1, t2)
        self.cos_xq = np.cos(np.pi*xq).reshape((-1, 1))
        self.sin_xq = np.sin(np.pi*xq).reshape((-1, 1))

    @property
    def weights(self):
        '''Quad weights'''
        return self.__weights__

    def points(self, n):
        '''Quad points for circle(x0, radius(x0)) in plane with normal n'''
        # Fix plane
        t1 = np.array([n[1]-n[2], n[2]-n[0], n[0]-n[1]])
    
        t2 = np.cross(n, t1)
        t1 /= np.linalg.norm(t1)
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
