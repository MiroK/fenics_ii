from __future__ import absolute_import
from __future__ import print_function
from dolfin import *
from block import block_assemble, block_bc, block_mat
from block.iterative import MinRes
from block.algebraic.petsc import AMG, LumpedInvDiag
from six.moves import map
from six.moves import range


u_exact = Expression(('cos(pi*x[1])', 'sin(pi*x[0])'), degree=4)
p_exact = Expression('sin(2*pi*(x[0]+x[1]))', degree=4)


def taylor_hood(n):
    '''Just check MMS'''
    mesh = UnitSquareMesh(n, n)
    # Just approx
    f_space = VectorFunctionSpace(mesh, 'DG', 1)
    h_space = TensorFunctionSpace(mesh, 'DG', 1)

    u_int = interpolate(u_exact, VectorFunctionSpace(mesh, 'CG', 2))
    p_int = interpolate(p_exact, FunctionSpace(mesh, 'CG', 2))
        
    f = project(-div(grad(u_int)) + grad(p_int), f_space)
    h = project(-p_int*Identity(2) + grad(u_int), h_space)

    # ----------------

    Vel = VectorElement('Lagrange', triangle, 2)
    Qel = FiniteElement('Lagrange', triangle, 1)
    Wel = MixedElement([Vel, Qel])
    W = FunctionSpace(mesh, Wel)

    u, p = TrialFunctions(W)
    v, q = TestFunctions(W)

    n = FacetNormal(mesh)
    
    a = inner(grad(u), grad(v))*dx - inner(div(v), p)*dx - inner(div(u), q)*dx
    L = inner(dot(h, n), v)*ds + inner(f, v)*dx

    bc = DirichletBC(W.sub(0), u_exact, 'near(x[0], 0)')

    wh = Function(W)
    solve(a == L, wh, bc)

    return wh.split(deepcopy=True), W.dim()


class Sum(Expression):
    def __init__(self, fs, **kwargs):
        value_shape = set(tuple(f.value_shape()) for f in fs)
        assert len(value_shape) == 1
        assert value_shape.pop() == (2, )
        self.fs = fs

    def value_shape(self):
        return (2, )  # To mathch the assertion

    def eval(self, values, x):
        fs = iter(self.fs)
        values[:] = next(fs)(x)
        for f in fs:
            values[:] += f(x)


def mini(n):
    '''Just check MMS'''
    mesh = UnitSquareMesh(n, n)
    # Just approx
    f_space = VectorFunctionSpace(mesh, 'DG', 1)
    h_space = TensorFunctionSpace(mesh, 'DG', 1)

    u_int = interpolate(u_exact, VectorFunctionSpace(mesh, 'CG', 2))
    p_int = interpolate(p_exact, FunctionSpace(mesh, 'CG', 2))
        
    f = project(-div(grad(u_int)) + grad(p_int), f_space)
    h = project(-p_int*Identity(2) + grad(u_int), h_space)

    # ----------------

    Vel = VectorElement('Lagrange', triangle, 1)
    Vel_b = VectorElement('Bubble', triangle, 3)
    Qel = FiniteElement('Lagrange', triangle, 1)
    Wel = MixedElement([Vel, Vel_b, Qel])
    W = FunctionSpace(mesh, Wel)

    u_, ub, p = TrialFunctions(W)
    v_, vb, q = TestFunctions(W)

    u = u_ + ub
    v = v_ + vb

    n = FacetNormal(mesh)
    
    a = inner(grad(u), grad(v))*dx - inner(div(v), p)*dx - inner(div(u), q)*dx
    L = inner(dot(h, n), v)*ds + inner(f, v)*dx

    bc = DirichletBC(W.sub(0), u_exact, 'near(x[0], 0)')

    wh = Function(W)
    solve(a == L, wh, bc)

    uh_, uhb, ph = wh.split(deepcopy=True)
                
    uh = Sum([uh_, uhb], degree=1)
                
    return uh, ph, W.dim()


def mini_block(n):
    '''Just check MMS'''
    mesh = UnitSquareMesh(n, n)
    # Just approx
    f_space = VectorFunctionSpace(mesh, 'DG', 1)
    h_space = TensorFunctionSpace(mesh, 'DG', 1)

    u_int = interpolate(u_exact, VectorFunctionSpace(mesh, 'CG', 2))
    p_int = interpolate(p_exact, FunctionSpace(mesh, 'CG', 2))
        
    f = project(-div(grad(u_int)) + grad(p_int), f_space)
    h = project(-p_int*Identity(2) + grad(u_int), h_space)

    # ----------------

    V = VectorFunctionSpace(mesh, 'Lagrange', 1)
    Vb = VectorFunctionSpace(mesh, 'Bubble', 3)
    Q = FunctionSpace(mesh, 'Lagrange', 1)
    W = [V, Vb, Q]

    u, ub, p = list(map(TrialFunction, W))
    v, vb, q = list(map(TestFunction, W))

    n = FacetNormal(mesh)

    a = [[0]*len(W) for _ in range(len(W))]
    a[0][0] = inner(grad(u), grad(v))*dx
    a[0][2] = - inner(div(v), p)*dx
    a[2][0] = - inner(div(u), q)*dx

    a[1][1] = inner(grad(ub), grad(vb))*dx
    a[1][2] = - inner(div(vb), p)*dx
    a[2][1] = - inner(div(ub), q)*dx

    a[0][1] = inner(grad(v), grad(ub))*dx
    a[1][0] = inner(grad(vb), grad(u))*dx

    # NOTE: bubbles don't contribute to surface
    L = [inner(dot(h, n), v)*ds + inner(f, v)*dx,
         inner(f, vb)*dx,
         inner(Constant(0), q)*dx]
    # Bubbles don't have bcs on the surface
    bcs = [[DirichletBC(W[0], u_exact, 'near(x[0], 0)')], [], []]

    AA = block_assemble(a)
    bb = block_assemble(L)

    block_bc(bcs, True).apply(AA).apply(bb)

    # Preconditioner
    B0 = AMG(AA[0][0])
    
    H1_Vb = inner(grad(ub), grad(vb))*dx + inner(ub, vb)*dx
    B1 = LumpedInvDiag(assemble(H1_Vb))
    
    L2_Q = assemble(inner(p, q)*dx)
    B2 = LumpedInvDiag(L2_Q)

    BB = block_mat([[B0, 0, 0],
                    [0, B1, 0],
                    [0, 0, B2]])
    
    x0 = AA.create_vec()
    x0.randomize()

    AAinv = MinRes(AA, precond=BB, tolerance=1e-10, maxiter=500, relativeconv=True, show=2,
                   initial_guess=x0)

    # Compute solution
    u, ub, p = AAinv * bb

    uh_ = Function(V, u)
    uhb = Function(Vb, ub)
    ph = Function(Q, p)
    
    uh = Sum([uh_, uhb], degree=1)
                
    return uh, ph, sum(Wi.dim() for Wi in W)

# -------------------------------------------------------------------

if __name__ == '__main__':
    solver = mini_block
    
    for n in (8, 16, 32, 64, 128):
        uh, ph, ndofs = solver(n)
        mesh = ph.function_space().mesh()
        
        e_uh = errornorm(u_exact, uh, 'H1', degree_rise=1, mesh=mesh)
        e_ph = errornorm(p_exact, ph, 'L2', degree_rise=1, mesh=mesh)

        print('\terrors', e_uh, e_ph, 'ndofs', ndofs)

