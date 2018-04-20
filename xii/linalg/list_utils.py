# Newly, b = np.array([Vector]) will for some reason call .array of vector
# b is not an array of vector object but of numbers! As a workaround
# we'll do things with lists

def shape_list(lst):
    '''Shape of a list (of lists of list)'''
    if not isinstance(lst, list) or not lst: return ()

    items_shape = set(map(shape_list, lst))
    shape = items_shape.pop()
    assert not items_shape

    return (len(lst), ) + shape


def flatten_list(lst):
    '''Collapse a nested list'''
    value_shape = shape_list(lst)

    fshape = value_shape
    while len(fshape) > 1:
        lst = sum(lst, [])
        fshape = (fshape[0]*fshape[1], ) + fshape[2:]
    return lst


def reshape_list(lst, shape):
    '''Same as array.reshape'''
    old_shape = shape_list(lst)
    assert reduce(lambda x, y: x*y, old_shape) == reduce(lambda x, y: x*y, shape)

    lst = flatten_list(lst)
    while len(shape) > 1:
        count = shape[-1]
        lst = package(lst, count)        
        shape = shape[:-1]
    return lst


def package(lst, count):
    '''Break a flat list into pieces with count items'''
    return [] if not lst else ([lst[:count]] + package(lst[count:], count))

# --------------------------------------------------------------------

if __name__ == '__main__':
    import numpy as np

    array = np.arange(8)
    lst = array.tolist()

    assert array.shape == shape_list(lst)

    array = array.reshape((4, 2))
    assert reshape_list(lst, (4, 2)) == array.tolist()

    array = array.reshape((2, 4))
    assert reshape_list(lst, (2, 4)) == array.tolist()

    array = array.reshape((2, 2, 2))
    assert reshape_list(lst, (2, 2, 2)) == array.tolist()

    array = array.reshape((8, 1))
    assert reshape_list(lst, (8, 1)) == array.tolist()

    array = array.reshape((1, 8))
    assert reshape_list(lst, (1, 8)) == array.tolist()




