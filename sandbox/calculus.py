import sympy as sp
import dolfin as df
import numpy as np


def is_vector(f): return len(f.shape) == 2 and 1 in f.shape


def is_2tensor(f): return len(f.shape) == 2 and 1 not in f.shape


def Grad(f, X=sp.symbols('x[0], x[1]')):
    '''Of scalar or vector in 2d'''
    try:
        shape = f.shape
    except AttributeError:
        return sp.Matrix([f.diff(X[0], 1), f.diff(X[1], 1)])

    assert is_vector(f)

    return sp.Matrix([[fi.diff(xj) for xj in X] for fi in list(f.values())])


def Div(f, X=sp.symbols('x[0], x[1]')):
    '''Divergence of vector/tensor in 2d'''
    if is_vector(f):
        return sum(fi.diff(xi, 1) for fi, xi in zip(list(f.values()), X))

    assert is_2tensor(f)
    return sp.Matrix([Div(f.row(i)) for i in range(f.rows)])


def asExpr(f, degree=6):
    '''Convert sympy expression to Dolfin'''
    ccode = lambda s: sp.printing.ccode(s).replace('M_PI', 'pi')
    try:
        if is_vector(f) or is_2tensor(f):
            if is_vector(f):
                shape = (max(f.shape), )
            else:
                shape = f.shape
            code = np.array(list(map(ccode, list(f.values())))).reshape(shape)
            return df.Expression(code.tolist(), degree=degree)
    except AttributeError:
        return df.Expression(ccode(f), degree=degree)

# --------------------------------------------------------------------

if __name__ == '__main__':
    x, y = sp.symbols('x, y')

    assert Grad(x) == sp.Matrix([[1], [0]])
    assert Grad(sp.Matrix([x, y])) == sp.Matrix([[1, 0], [0, 1]]), Grad(sp.Matrix([x, y]))

    assert Grad(sp.Matrix([x**2, y])) == sp.Matrix([[2*x, 0], [0, 1]]), Grad(sp.Matrix([x, y]))

    assert Div(sp.Matrix([x, y])) == 2

    assert Div(Grad(x**2+y**2)) == 4
