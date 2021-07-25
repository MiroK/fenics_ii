import numpy as np


def bbox(shape):
    '''Bounding box of [A, B, C, ...]'''
    return (np.min(shape, axis=0), np.max(shape, axis=0))


def box_collides(box, line, tol=1E-13):
    '''Yes/No on collision of line and a box'''

    if len(box) == 2:
        return box_collides_2d(line, box, tol)
    # FIXME: 3d?
    return box_collides_3d(line, box, tol)


def box_collides_2d(box, line, tol):
    ''' Liang-Barsky algorithm '''

    first
    
    return is_outside(line[0]) and is_outside(line[1])

    u1 = -np.inf
    u2 = np.inf
    for pi, qi in zip(p, q):
        if (abs(pi) < tol and qi < 0):
            return False

        t = qi/pi

        if (pi < 0 and u1 < t):
            u1 = max(0, t)
        elif (pi > 0 and u2 > t):
            u2 = min(1, t)

    if (u1 > u2): return False
    
    return True


def collides_cell(cell, seg, tol=1E-13):
    '''Yes/No on a collision of cell (segment, triangle) with segment.'''
    box = bbox(cell)

    # If there is no bbox collision then there is no collision
    #if not box_collides(box, seg, tol):
    #    print 'xxxx'
    #    return False
    # Otherwise we need to compute
    return intersects(cell, seg, tol)


def intersects(cell, seg, tol):
    '''Yes/No on the intersection of cell and segment'''
    X, Y = seg
    vec = X - cell[0]  # Rhs
    # Get the system
    if len(cell) == 2:
        A, B = cell
        mat = np.vstack([B-A, X-Y]).T
    else:
        A, B, C = cell
        mat = np.vstack([B-A, C-A, X-Y]).T
    
    try:
        x = np.linalg.solve(mat, vec)
    except np.linalg.LinAlgError:
        return False

    is_isect = all((0 - tol < xi < 1 + tol) for xi in x)
    if len(x) == 3:
        is_isect = is_isect and (0 - tol < x[0] + x[1] < 1 + tol)

    return is_isect

# --------------------------------------------------------------------

if __name__ == '__main__':

    seg = np.array([[0, 0], [1., 1]])
    box = bbox(seg)

    line = np.array([[3, 3], [4., 4]])
    print(box_collides(box, line))
    print(collides_cell(seg, line))
    
    line = np.array([[0, 1], [1., 0]])
    print(box_collides(box, line))
    print(collides_cell(seg, line))
