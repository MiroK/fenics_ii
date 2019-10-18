from dolfin import FunctionSpace
import numpy as np


def first(iterable):
    return next(iter(iterable))


def elmtype(iterable):
    return type(first(iterable))

# Let there be a block_form of rank 1/2 which is either a linear/bilinear
# form or a list of block_form
def zeros(V, Q=None):
    if Q is None:
        if isinstance(V, FunctionSpace):
            return zeros([V])
    
        if elmtype(V) is FunctionSpace:
            return [0]*len(V)
    
        return map(zeros, V)
    else:
        return np.kron(zeros(V), zeros(Q))


class block_linear_form(object):
    def __init__(self, test_space):
        self.test_space = test_space
        self.sub_forms = zeros(test_space)

        
class block_form(list):
    def __init__(self, test_space, trial_space=None):

        self.test_space = test_space
        
        if trial_space is None:
            trial_space = test_space
        self.trial_space = trial_space

        # Alloc
        # self.sub_forms =

# def set_form(form, index, value)

# --------------------------------------------------------------------

if __name__ == '__main__':
    from dolfin import *

    mesh = UnitSquareMesh(32, 32)
    V = FunctionSpace(mesh, 'CG', 1)

    print zeros(V)
    print zeros([V, V])
    print zeros([[V, V, V], [V, V]])

    print zeros(V, V)
    print zeros([V, V], V)
    print zeros([[V, V, V], [V, V]], [V, V])
