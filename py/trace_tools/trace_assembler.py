from dolfin import *
import trace_matrices
from petsc4py import PETSc
from ufl.formatting.graph import build_graph
from ufl.algorithms.traversal import iter_expressions
from ufl.restriction import NegativeRestricted, PositiveRestricted


def trace_assemble(form, embedded_mesh):
    '''
    Assemble bilinear forms that mix Trial/Test functions defined on mesh
    and embedded mesh.     '''
    # Let V be the full mesh space. The idea is that the form V x Q is assembled in
    # two steps: i) TV, a space of traces of V, is identified and a new form (defined
    # only in terms of arguments defined on the embedded mesh), i.e. Q and TV, is 
    # computed and assembled. ii) A matrix that maps V to TV is computed. 
    analysis = analyze_form(form, embedded_mesh)
    # Assemble i)
    M = as_backend_type(assemble(analysis['form'])).mat()
    # Assemble ii)
    T = trace_matrix(space=analysis['space'],
                     trace_space=analysis['trace_space'],
                     restriction=analysis['restriction'],
                     emesh=embedded_mesh)
    # Combine based on whether V -> Q or Q -> V (its transpose was requested)
    A = PETSc.Mat()
    M.matMult(T, A)

    if analysis['transpose']: 
        A = A.transpose(PETSc.Mat())
        # NOTE: without setting the dofmaps again we get PETSc error code 73.
        # rows are from space, cols aren't from trace space but from the space
        # due to the second argument
        from_space, to_space = analysis['space'], analysis['range']
        from_map, to_map = from_space.dofmap(), to_space.dofmap()
        comm = from_space.mesh().mpi_comm().tompi4py()
        # Local to global
        row_lgmap = PETSc.LGMap().create(map(int, from_map.tabulate_local_to_global_dofs()), comm=comm)
        col_lgmap = PETSc.LGMap().create(map(int, to_map.tabulate_local_to_global_dofs()), comm=comm)
        A.setLGMap(row_lgmap, col_lgmap)
    
    return PETScMatrix(A)


def trace_matrix(space, trace_space, restriction, emesh):
    '''
    Compute the matrix T such that if x is a vector of coefficients representing
    a function in space then Tx is the vector of a trace of the function in the
    trace_space.
    '''
    info('\tComputing trace matrix')
    timer = Timer('trace matrix')

    tdim = space.mesh().topology().dim()
    gdim = space.mesh().geometry().dim()
    trace_tdim = trace_space.mesh().topology().dim()
    trace_gdim = trace_space.mesh().geometry().dim()

    assert 0 < trace_tdim < tdim
    assert gdim == trace_gdim
    assert trace_tdim < tdim

    # Dispatch
    family = space.ufl_element().family()
    if family == 'Lagrange':
        assert trace_space.ufl_element().family() == 'Lagrange'
        # Here we compute by point eval (not using entity_map)
        Tmat = trace_matrices.Lagrange_trace_matrix(space, trace_space)
    else:
        assert family in ('Brezzi-Douglas-Marini', 'Raviart-Thomas')
        normal = emesh.normal
        assert normal is not None
        assert emesh.entity_map
        assert restriction in ('+', '-')
        # Pack the extra data for Hdiv trace to tuple
        Tmat = trace_matrices.BRM_RT_trace_matrix(space,
                                                  trace_space,
                                                  (normal, emesh.entity_map, restriction))

    info('\tDone in %g' % timer.stop())
    return Tmat


def analyze_form(form, emesh):
    '''See if the form'''
    mesh = emesh.mesh
    # We restrct this to single integrals forms
    integral = form.integrals()
    assert len(integral) == 1
    integral = integral[0]
    # defined in terms dx measure of the emebdded mesh
    assert integral.integral_type() == 'cell'
    domain = form.ufl_domain()
    tdim, gdim = domain.topological_dimension(), domain.geometric_dimension()
    assert 0 < tdim < gdim
    assert domain == mesh.ufl_domain()
    # The trial function and test function should have 'embedded cells'
    args = form.arguments()
    numbers = [arg.number() for arg in args]
    test_f, trial_f = args[numbers.index(0)], args[numbers.index(1)]

    test_d, trial_d = test_f.ufl_domain(), trial_f.ufl_domain()
    assert test_d != trial_d
    assert test_d.topological_dimension() < trial_d.topological_dimension() or \
           trial_d.topological_dimension() < test_d.topological_dimension()
    assert test_d.geometric_dimension() == trial_d.geometric_dimension()

    # The order of matrices that make the form matrix is determined by the test
    # function or trial function is over the domain
    if test_d == domain:
        do_transpose = False
        tform = form
    else:
        assert trial_d == domain
        do_transpose = True
        new_trial = TrialFunction(test_f.ufl_function_space()) 
        new_test = TestFunction(trial_f.ufl_function_space())

        tform = replace(form, {test_f: new_trial, trial_f: new_test})
        
        trial_f, test_f = new_trial, new_test
    space = trial_f.ufl_function_space()
    
    # Find the trace element so that the trace space of test can be defined
    element = trial_f.ufl_element()
    family, degree = element.family(), element.degree()
    cell = domain.ufl_cell()
    # It remains to perform the surgery on the form
    if family == 'Lagrange':
        FE = {0: FiniteElement, 1: VectorElement, 2: TensorElement}

        trace_element = FE[len(trial_f.ufl_shape)](family, cell, degree)
        trace_space = FunctionSpace(mesh, trace_element)
        trace_trial_f = TrialFunction(trace_space)
        # Here it is simply to replace the 'full mesh' function with trace one
        # where the latter will be computed by the trace operator
        trace_form = replace(tform, {trial_f: trace_trial_f})
        restriction = ''
    else:
        assert family in ('Brezzi-Douglas-Marini', 'Raviart-Thomas')
        # Make this work only for vector valued spaces
        assert len(element.value_shape()) == 1
        # See debug.py
        assert test_f.ufl_element().degree() == 0, 'See debug.py'

        # The form should have a normal which is the normal from the mesh
        form_normal = tform.coefficients()
        assert len(form_normal) == 1
        form_normal = form_normal[0]
        assert form_normal == emesh.normal

        normal_restriction = find_restriction_of(form_normal, tform)
        assert len(normal_restriction) == 1
        normal_restriction = normal_restriction.pop()
        ndegree = form_normal.ufl_element().degree()
        # Want 0 degreen so that normal at the midpoint where it makes sense
        # does not get poluted by sampling in the corners. Futher the space for
        # v.n is the same as for v component
        assert ndegree == 0
        # The assumption is that the form is v.n * q and by trace we plan to
        # replace v.n term
        trace_element = FiniteElement('Discontinuous Lagrange', cell, degree)
        trace_space = FunctionSpace(mesh, trace_element)
        trace_trial_f = TrialFunction(trace_space)
        # Max tdim measure
        measure = dx(domain=domain, subdomain_id=integral.subdomain_id(),
                     subdomain_data=integral.subdomain_data())
        trace_form = inner(trace_trial_f, test_f)*measure
        # from ufl.algorithms import estimate_total_polynomial_degree
        # print 'Degree tr', trace_form, estimate_total_polynomial_degree(trace_form)

        # To compute the trace the restriction is needed
        restriction = find_restriction_of(trial_f, tform)
        assert len(restriction) == 1
        restriction = restriction.pop()
        # To me it does not make sense to have the 2 restrictions differ
        assert restriction == normal_restriction, (restriction, normal_restriction)

    return {'form': trace_form,
            'space': space,                    # from
            'trace_space': trace_space,        # to#1
            'range': test_f.function_space(),  # to#2,
            'transpose': do_transpose,
            'restriction': restriction}


def find_restriction_of(f, form):
    '''See if f in the form is as f('+') or f('-').'''
    signs = []
    for expr in iter_expressions(form):
        V, E = build_graph(expr)
        for index, vertex in enumerate(V):
            # Once we found f as the vertex of the graph we check the edges
            # leading to it
            if vertex == f:
                connected = filter(lambda e: index in e, E)
                for edge in connected:
                    other = (set(edge) - set([index])).pop()
                    other = V[other]
                    if isinstance(other, PositiveRestricted):
                        signs.append('+')
                    elif isinstance(other, NegativeRestricted):
                        signs.append('-')
                    else:
                        pass
    return signs if signs else []
    
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    from embedded_mesh import EmbeddedMesh

    gamma = ['near((x[0]-0.25)*(x[0]-0.75), 0) && (0.25-DOLFIN_EPS < x[1]) && (x[1] < 0.75+DOLFIN_EPS)',
             'near((x[1]-0.25)*(x[1]-0.75), 0) && (0.25-DOLFIN_EPS < x[0]) && (x[0] < 0.75+DOLFIN_EPS)']
    gamma = map(lambda x: '('+x+')', gamma)
    gamma = ' || '.join(gamma)
    gamma = CompiledSubDomain(gamma)

    n = 2
    n *= 4
    omega_mesh = UnitSquareMesh(n, n)
    facet_f = FacetFunction('size_t', omega_mesh, 0)
    gamma.mark(facet_f, 1)

    gamma_mesh = EmbeddedMesh(omega_mesh, facet_f, 1)

    # Lagr
    V = FunctionSpace(omega_mesh, 'CG', 1)
    Q = FunctionSpace(gamma_mesh.mesh, 'DG', 1)

    u, p = TrialFunction(V), TrialFunction(Q)
    v, q = TestFunction(V), TestFunction(Q)

    dxGamma = Measure('dx', domain=gamma_mesh.mesh)

    a00 = inner(grad(u), grad(v))*dx
    a01 = inner(p, v)*dxGamma
    a10 = inner(u, q)*dxGamma

    A01 = trace_assemble(a01, gamma_mesh)
    assert (V.dim(), Q.dim()) == (A01.size(0), A01.size(1))
    A10 = trace_assemble(a10, gamma_mesh)
    assert (Q.dim(), V.dim()) == (A10.size(0), A10.size(1))

    # With normal...
    nx = 'near((x[0]-0.25)*(x[0]-0.75), 0, tol) ? ((x[0] > 0.5) ? 1 : -1) : 0'
    ny = 'near((x[1]-0.25)*(x[1]-0.75), 0, tol) ? ((x[1] > 0.5) ? 1 : -1) : 0'
    n = Expression((nx, ny), degree=0, tol=1E-8)
    gamma_mesh.normal = n

    V = FunctionSpace(omega_mesh, 'BDM', 1)
    Q = FunctionSpace(gamma_mesh.mesh, 'DG', 0)

    u = TrialFunction(V)
    q = TestFunction(Q)
    a10 = inner(dot(u('-'), n('-')), q)*dxGamma
    A10 = trace_assemble(a10, gamma_mesh)
    print Q.dim(), V.dim(), A10.size(0), A10.size(1)
