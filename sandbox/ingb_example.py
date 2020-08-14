from dolfin import *
from xii import *


# In 3d
if True:
    omega = UnitCubeMesh(8, 8, 8)
    gamma = MeshFunction('size_t', omega, 1, 0)
    CompiledSubDomain('near(x[0], 0.0) && near(x[1], 0.0)').mark(gamma, 1)

    gamma = EmbeddedMesh(gamma, 1)

    # Spaces
    V1 = FunctionSpace(omega, 'RT', 1)
    Q1 = FunctionSpace(omega, 'DG', 0)
    
    W = [V1, Q1]

    sigma1, u1 = map(TrialFunction, W)
    tau1, v1 = map(TestFunction, W)
    
    dx = Measure('dx', domain=omega)
    # The line integral
    dx_ = Measure('dx', domain=gamma)
    
    
    a = block_form(W, 2)
    a[0][0] = inner(sigma1, tau1)*dx
    a[0][1] = inner(div(tau1), u1)*dx
    a[1][0] = inner(div(sigma1), v1)*dx
    
    L = block_form(W, 1)

    # Almost the complete system
    A, b = map(ii_assemble, (a, L))
    
    
    # Now need to add (Pi(sigma), Pi(tau))*dGamma
    from xii.assembler.normal_average_matrix import normal_average_matrix

    radius = 0.01
    quad_degree = 10
    # We compute average over
    shape = Circle(radius=radius, degree=quad_degree)
    # The average lives in auxiliary space
    Pi_V1 = FunctionSpace(gamma, 'DG', 0)
    
    Pi = normal_average_matrix(V1, Pi_V1, shape)
    # To compute the inner product a mass matrix is needed
    M = assemble(inner(TrialFunction(Pi_V1), TestFunction(Pi_V1))*dx_)
    # The term is now
    foo = Pi.T*M*Pi
    
    # We add it to the matrix
    A[0][0] += foo
    
    # Some made up boundary conditions
    V1_bcs = [DirichletBC(V1, Constant((0, 0, 0)), 'near(x[2], 0)')]
    Q1_bcs = []
    bcs = [V1_bcs, Q1_bcs]
    
    A, b = apply_bc(A, b, bcs)
    
    wh = ii_Function(W)
    solve(ii_convert(A), wh.vector(), ii_convert(b))
    print(wh.vector().size())

# In 2d
if True:
    omega = UnitSquareMesh(32, 32)
    gamma = Point(0.5, 0.5)  # Center of the averaging shape
    
    # Spaces
    V1 = FunctionSpace(omega, 'RT', 1)
    Q1 = FunctionSpace(omega, 'DG', 0)
    
    W = [V1, Q1]
    
    sigma1, u1 = map(TrialFunction, W)
    tau1, v1 = map(TestFunction, W)
    
    dx = Measure('dx', domain=omega)

    a = block_form(W, 2)
    a[0][0] = inner(sigma1, tau1)*dx
    a[0][1] = inner(div(tau1), u1)*dx
    a[1][0] = inner(div(sigma1), v1)*dx
    
    L = block_form(W, 1)
    
    # Almost the complete system
    A, b = map(ii_assemble, (a, L))
    
    # Now need to add (Pi(sigma), Pi(tau))*dGamma
    from xii.assembler.normal_average_matrix import normal_average_matrix_2d
    
    radius = 0.01
    quad_degree = 10
    # We compute average over
    shape = Circle(radius=radius, degree=quad_degree)
    
    Pi = normal_average_matrix_2d(V1, gamma, shape)
    # The term is now
    foo = Pi.T*Pi
    # NOTE: it's safe to use ii_convert for this because Pi here is sparse
    # and the kronecker product that is foo will be also represented as a
    # sparse matrix
    
    # # We add it to the matrix
    A[0][0] += foo

    # Some made up boundary conditions
    V1_bcs = [DirichletBC(V1, Constant((0, 0)), 'near(x[2], 0)')]
    Q1_bcs = []
    bcs = [V1_bcs, Q1_bcs]
    
    A, b = apply_bc(A, b, bcs)
    
    wh = ii_Function(W)
    solve(ii_convert(A), wh.vector(), ii_convert(b))
    print(wh.vector().size())
