from dolfin.function.argument import Argument
from ufl_legacy.core.terminal import Terminal

import dolfin as df


def topological_dim(thing):
    '''Extract topological dimension of thing's cell'''
    domain = thing.ufl_domain()  # None with e.g. Constants
    return -1 if domain is None else domain.ufl_cell().topological_dimension()


def geometric_dim(thing):
    '''Extract geoemtric dimension of thing's cell'''
    domain = thing.ufl_domain()
    return -1 if domain is None else domain.ufl_cell().geometric_dimension()


def is_terminal(o):
    '''Is o a terminal type'''
    return isinstance(o, Terminal)


def traverse(expr):
    '''Traverse the UFL expression tree'''
    if expr.ufl_operands:
        for op in expr.ufl_operands:
            for e in traverse(op):
                yield e
    yield expr

    
def traverse_terminals(expr):
    '''
    Yield all the termnals (can be duplicate) in the UFL expression tree
    '''
    return filter(is_terminal, traverse(expr))


def traverse_subexpr(expr):
    '''Yield nodes of the UFL expression tree that have arguments'''
    return filter(lambda e: not is_terminal(e), traverse(expr))


def is_equal_terminal(this, that, attributes=None):
    '''
    Trace introduces attributes to object. So comparison of terminals 
    must take this into account
    '''
    if is_terminal(this) and is_terminal(that):
        ufl_equal = this == that
        # For no attributes we default to standard ufl comparison
        if not ufl_equal or attributes is None:
            return ufl_equal
        # All atributes are there and objects agree on their value
        if all(hasattr(this, a) for a in attributes) and all(hasattr(that, a) for a in attributes):
            return all(getattr(this, a) == getattr(that, a) for a in attributes)

        return False

    return False


def matches(expr, target, attributes=None):
    '''Compare two UFL expression for equalty'''
    # NOTE: this is 99.9% duplecate for ufl functionalty
    # Terminal are the same if ==
    if is_terminal(expr) and is_terminal(target):
        return is_equal_terminal(expr, target, attributes)
    # Not terminal need to agree on type and have the same argument
    if not is_terminal(expr) and not is_terminal(target):
        return all((isinstance(expr, type(target)),
                    all(matches(*ops, attributes=None)
                        for ops in zip(expr.ufl_operands, target.ufl_operands))))
    return False

                                                              
def contains(expr, target, attributes=None):
    '''Is the target expression contained in the expression?'''
    # A tarminal target either agrees or is one of the expr terminals
    if is_terminal(target):
        if is_terminal(expr):
            return is_equal_terminal(expr, target, attributes)
        else:
            return any(matches(target, t, attributes) for t in traverse_terminals(expr))
        
    # Target is a expression
    if is_terminal(expr): return False

    # The nodes need to match
    ttarget = type(target)
    for sexpr in traverse_subexpr(expr):
        if matches(sexpr, target, attributes):
            return True
    return False


def replace(expr, arg, replacement, attributes=None):
    '''Replace and argument in the expression by the replacement'''
    # Do nothing if no way to substitute, i.e. return original
    if not contains(expr, arg, attributes):
        return expr
    # Identical 
    if matches(expr, arg, attributes):
        return replacement

    # Reconstruct the node with the substituted argument
    return type(expr)(*[replace(op, arg, replacement, attributes) for op in expr.ufl_operands])


def is_trial_function(v):
    '''Is v a trial function'''
    return isinstance(v, Argument) and v.number() == 1


def is_test_function(v):
    '''Is v a test function'''
    return isinstance(v, Argument) and v.number() == 0


def form_arity(form):
    '''How many args in the form'''
    return len(form.arguments())

# FIXME: fun problem is it linear?

def reconstruct(f):
    '''Make a copy of terminal object'''
    assert is_terminal(f)

    if is_trial_function(f):
        return df.TrialFunction(f.function_space())

    if is_test_function(f):
        return df.TestFunction(f.function_space())
    
    # Others?
    return f
