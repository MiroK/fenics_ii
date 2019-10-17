import numpy as np


def segment_distance(P, A, B):
    '''|AB|---P'''
    u, v = B - A, P - A
    l = np.linalg.norm(B-A)
    # See about intersect
    t = np.dot(u, v)/l**2

    # Is isect between ?
    # Nope
    if t < 0:
        return np.linalg.norm(A-P)
    # Nope
    elif t > 1:
        return np.linalg.norm(B-P)
    # Yep
    else:
        return np.linalg.norm(np.cross(u, v))/l
    

# -------------------------------------------------------------------

if __name__ == '__main__':
    from functools import partial
    from dolfin import *
    
    mesh = UnitCubeMesh(16, 16, 16)

    V = FunctionSpace(mesh, 'CG', 1)
    f = Function(V)

    ds = [partial(segment_distance, A=np.array([0.0, 0.5, 0.5]), B=np.array([1.0, 0.5, 0.5])),
          partial(segment_distance, A=np.array([0.5, 0.0, 0.5]), B=np.array([0.5, 1.0, 0.5])),
          partial(segment_distance, A=np.array([0.5, 0.5, 0.0]), B=np.array([0.5, 0.5, 1.0]))]
    
    dofs_x = V.tabulate_dof_coordinates().reshape((V.dim(), -1))
    
    distance = f.vector().get_local()
    distance += 3

    for d in ds:
        distance = np.minimum(distance, map(d, dofs_x))

    f.vector().set_local(distance)

    File('fpp.pvd') << f
