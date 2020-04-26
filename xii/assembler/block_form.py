from __future__ import absolute_import
from __future__ import print_function
from xii.assembler.ufl_utils import traverse_terminals, replace as ii_replace
from itertools import dropwhile
from dolfin import *
import numpy as np
import ufl
from six.moves import filter
from six.moves import map
from six.moves import range
from six.moves import zip


def is_trial_function(arg):
    '''Are you?'''
    return arg.number() == 1


def is_test_function(arg):
    '''Are you?'''
    return arg.number() == 0


def is_number(form):
    '''Ask'''
    return isinstance(form, (int, float, Constant))


def trial_function(form):
    '''Extract trial function[s] of [block] form'''
    if isinstance(form, ufl.Form):
        return list(filter(is_trial_function, form.arguments()))
    return sum(list(map(trial_function, form)), ())


def test_function(form):
    '''Extract test function[s] of [block] form'''
    if isinstance(form, ufl.Form):
        return list(filter(is_test_function, form.arguments()))
    return sum(list(map(test_function, form)), ())


def is_bilinear_form(form):
    '''There is a trial function'''
    if isinstance(form, ufl.Form):
        return bool(trial_function(form))
    # Block
    return all(map(is_bilinear_form, form))


def is_linear_form(form):
    '''There is only a test function'''
    if isinstance(form, ufl.Form):
        return not is_bilinear_form(form) and len(test_function(form)) == 1 
    # Block
    return all(map(is_linear_form, form))


def first(iterable):
    '''#1'''
    return next(iter(iterable))


def elm_type(iterable):
    '''Iterable of'''
    return type(first(iterable))


class _block_form(list):
    '''A list with checking values (rectangular)'''
    def __init__(self, forms, V0, V1=None):
        # Check lin/bilin
        arity = 2 if elm_type(forms) in (list, tuple) else 1
        assert 0 < arity < 3
        self.arity = arity

        self.V0, self.V1 = V0, V1

        if arity == 1:
            list.__init__(self, forms)
            return None
        # We take it as fixed V1(trial space) and varying V
        list.__init__(self, [_block_form(row, V0, V1) for row, V1 in zip(forms, V0)])

    def __setitem__(self, index, value):
        '''self[i][j] 0 value'''
        if self.V1 is None:
            assert is_number(value) or ((Argument(self.V0[index], 0), ), None) == (test_function(value), None)
            return list.__setitem__(self, index, value)

        assert is_number(value) or (Argument(self.V1, 0), ) == test_function(value)
        assert is_number(value) or (Argument(self.V0[index], 1), ) == trial_function(value)
        
        return list.__setitem__(self, index, value)

    def add(self, form):
        '''self[i][j] += form with i, j extracted'''
        if len(form.integrals()) > 1:
            return list(map(self.add, [ufl.Form((i, )) for i in form.integrals()]))
            
        # Linear
        if self.arity == 1:
            assert trial_function(form) == ()
            
            test_f, = test_function(form)  # Only one
            # Look for the index
            V = test_f.function_space()
            i, _ = next(dropwhile(lambda i_Vi, V=V: i_Vi[1] != V, enumerate(self.V0)))
            
            self[i] += form
            return self

        # Bilinear
        test_f, = test_function(form)  # Only one
        trial_f, = trial_function(form)  # Only one

        V = trial_f.function_space()
        col, _ = next(dropwhile(lambda i_Vi, V=V: i_Vi[1] != V, enumerate(self.V0)))

        V = test_f.function_space()
        row, _ = next(dropwhile(lambda i_Vi, V=V: i_Vi[1] != V, enumerate(self.V0)))

        self[row][col] += form
        return self

    def __add__(self, other):
        '''Add block forms'''
        assert self.arity == other.arity
        
        assert self.V0 == other.V0
        W = self.V0
        
        if self.arity == 1:
            return _block_form([self[i] + other[i] for i in range(len(W))], W)

        return _block_form([[self[i][j] + other[i][j] for j in range(len(W))]
                            for i in range(len(W))], W)
        
                      
def null(V):
    '''0 on V'''
    return Constant(np.zeros(V.ufl_element().value_shape()))


def block_form(W, arity):
    '''Block linear or bilinear form'''
    assert 0 < arity < 3
    assert isinstance(W, (list, tuple))
    
    if arity == 1:
        return _block_form([inner(null(V), TestFunction(V))*dx for V in W], W)
    return _block_form([[0]*len(W) for _ in range(len(W))], W)


def is_null(thing):
    '''Zero block'''
    # Don't check zero-ity of form
    if isinstance(thing, (Form, ufl.Form)):
        return False

    # Init with Constant(0)    
    if isinstance(thing, Constant): return is_null(thing(0))
    
    return abs(thing) < DOLFIN_EPS


def is_upper_triangular(form):
    '''Empty below diagonal?'''
    return form.arity == 2 and all(is_null(form[i][j])
                                   for i in range(len(form.V0)) for j in range(i))


def is_lower_triangular(form):
    '''Empty above diagonal?'''
    return form.arity == 2 and all(is_null(form[j][i])
                                   for i in range(len(form.V0)) for j in range(i))


def form_adjoint(expr):
    '''Like UFL adjoint but keeping track of ii attributes'''
    if is_number(expr):
        return expr
    
    if isinstance(expr, ufl.Form):
        return ufl.Form(list(map(form_adjoint, expr.integrals())))

    if isinstance(expr, ufl.Integral):
        return expr.reconstruct(integrand=form_adjoint(expr.integrand()))

    # For adjoint we need one trial and one test function. The idea is
    # then exchange the two while keeping track of trace attributes
    arguments = set([f for f in traverse_terminals(expr) if isinstance(f, Argument)])

    ii_attrs = ('trace_', 'restriction_', 'average_')

    adj_type = {0: TrialFunction, 1: TestFunction}
    for arg in arguments:
        adj_arg = adj_type[arg.number()](arg.function_space())
        # Record trace attributes
        attrs = []
        for attr in ii_attrs:
            if hasattr(arg, attr):
                setattr(adj_arg, attr, getattr(arg, attr))
                attrs.append(attr)
        # Substitute
        expr = ii_replace(expr, arg, adj_arg, attrs)

    return expr


def make_selfadjoint(bform):
    '''Upper or lower triangular form to self-adjoint one'''
    n = len(bform.V0)
    if is_upper_triangular(bform):
        # Fill the empty ones
        for i in range(n):
            for j in range(i):
                bform[i][j] = form_adjoint(bform[j][i])
        return bform

    if is_lower_triangular(bform):
        # Fill the empty ones
        for i in range(n):
            for j in range(i):
                bform[j][i] = form_adjoint(bform[i][j])
        return bform

    # I could handle the rest by A + A.T but ...
    raise ValueError


def permute(bform, ordering):
    '''Construct symmetric permutation of bform'''
    # Compute it from the function space
    if elm_type(ordering) == FunctionSpace:
        ordering = [bform.V0.index(Vi) for Vi in ordering]
        
    # We can have [0, 2, 3, 1] meaning that old[0] is new[0]
    #                                       old[2] is new[1]
    #                                          [3] is new[2]    
    # Continuous range
    step,  = set(np.diff(np.sort(ordering)))
    assert abs(step) == 1

    # Sensible 
    assert len(ordering) == len(bform.V0)
    assert 0 == min(ordering) and len(bform.V0) -1 == max(ordering)

    # Permute space
    W = [bform.V0[i] for i in ordering]

    if bform.arity == 1:
        return _block_form([bform[i] for i in ordering], W)

    return _block_form([[bform[i][j] for j in ordering] for i in ordering], W)

# --------------------------------------------------------------------

if __name__ == '__main__':
    from dolfin import *

    mesh = UnitSquareMesh(32, 32)
    V = FunctionSpace(mesh, 'CG', 2)
    Q = FunctionSpace(mesh, 'CG', 1)
    W = [V, Q]


    u, p = list(map(TrialFunction, W))
    v, q = list(map(TestFunction, W))
        
    L = block_form(W, 1)

    a = block_form(W, 2)
    a.add(inner(u, q)*dx)
    
    # a[0][0] += inner(p ,q)*dx
    a00 = inner(u, v)*dx
    a01 = inner(p, v)*dx
    a10 = inner(q, u)*dx
    a11 = inner(p, q)*dx

    a.add(a00 + a01 + a10 + a11)

    this = block_form(W, 2)
    this.add(a00 + a11)

    print(is_upper_triangular(this))
    print(is_lower_triangular(this))
    
    that = block_form(W, 2)
    that.add(a10 + a01)
    print(is_upper_triangular(that))
    
    print(this + that)

    a = block_form(W, 2)
    a.add(a10)
    print(is_upper_triangular(a))
    print(is_lower_triangular(a))


    a = block_form(W, 2)
    a00 = inner(u, v)*dx
    a01 = inner(p, v)*dx
    a10 = inner(q, u)*dx
    a11 = inner(p, q)*dx
    a.add(a00 + a01 + a10 + a11)
    

    print() 
    b = permute(a, [1, 0])
    print(b[0][0].arguments() == a11.arguments())
    print(b[1][0].arguments() == a01.arguments())
    print(b[0][1].arguments() == a10.arguments())
    print(b[1][1].arguments() == a00.arguments())

    a = permute(b, [V, Q])

    print(a[0][0].arguments() == a00.arguments())
    print(a[1][0].arguments() == a10.arguments())
    print(a[0][1].arguments() == a01.arguments())
    print(a[1][1].arguments() == a11.arguments())
