# Boundary conditions involving pressure for the Stokes problem and applications in computational hemodynamics 
#
# With sigma such that sigma(u, p) = 2*mu*sym*grad(u) - p*Id we look for u, p(, and lm)
# such that
#
#     -div(sigma(u, p)) = f in Omega \subset R^2
#     div(u) = 0
#
# And bcs
#
#     u.n = 0, P_n = 0 on Gamma,
#     sigma.n.n = g_p Gamma (on part of boundary)
#
# And standard bcs on the remaining parts. Note that if Gamma is flat
# then sigma.n.n = pressure. We will have Omega based on [0, 1]^2 where
# left edge (Gamma) will be optionally circle arc (0, 0.5) with radius 0.5
from dolfin import *
from xii import *
import ulfy 

from hsmg.hseig import HsEig
from block.algebraic.petsc import LU

tangent = lambda f, n: f - dot(f, n)*n


def setup_mms(flat_gamma):
    ''''''
    mesh = UnitSquareMesh(2, 2)
    # The following labeling of edges will be enforced
    #   4
    # 1   2
    #   3
    x, y = SpatialCoordinate(mesh)
    phi = sin(pi*(x-2*y))  # Auxiliary for div free velocity
    u = as_vector((phi.dx(1), -phi.dx(0)))
    p = cos(pi*(2*x - 3*y))

    sigma = 2*sym(grad(u)) - p*Identity(2)
    f = -div(sigma)

    # Normal of gamma and how to construct the mesh then
    if flat_gamma:
        normals = (Constant((-1, 0)), )

        def get_geometry(i):
            n = 2*2**i
            mesh = UnitSquareMesh(n, n)

            facet_f = MeshFunction('size_t', mesh, 1, 0)
            CompiledSubDomain('near(x[0], 0)').mark(facet_f, 1)
            CompiledSubDomain('near(x[0], 1)').mark(facet_f, 2)
            CompiledSubDomain('near(x[1], 0)').mark(facet_f, 3)
            CompiledSubDomain('near(x[1], 1)').mark(facet_f, 4)            
        
            return facet_f
    else:
        cx, cy, r = Constant(0), Constant(0.5), Constant(0.5)
        normals = (as_vector(((x-cx)/r, (y-cy)/r)), )

        try:
            import gmshnics, gmsh
            # We want to define
            #     /------|
            #    (       |
            #    (       |
            #     \------|

            def get_geometry(i):
                
                gmsh.initialize(['', '-clscale', str(1/2**i), '-v', '0'])
                model = gmsh.model
                factory = model.occ

                ul = factory.addPoint(0, 1, 0)
                c = factory.addPoint(0, 0.5, 0)
                front = factory.addPoint(-0.5, 0.5, 0)
                ll = factory.addPoint(0, 0, 0)
                lr = factory.addPoint(1, 0, 0)
                ur = factory.addPoint(1, 1, 0)

                arc_top = factory.addCircleArc(ul, c, front)
                arc_bottom = factory.addCircleArc(front, c, ll)
                bottom = factory.addLine(ll, lr)
                right = factory.addLine(lr, ur)
                top = factory.addLine(ur, ul)

                loop = factory.addCurveLoop([arc_top, arc_bottom, bottom, right, top])
                domain = factory.addPlaneSurface([loop])

                factory.synchronize()

                model.addPhysicalGroup(2, [domain], 1)
                model.addPhysicalGroup(1, [arc_top, arc_bottom], 1)
                model.addPhysicalGroup(1, [right], 2)
                model.addPhysicalGroup(1, [bottom], 3)
                model.addPhysicalGroup(1, [top], 4)

                factory.synchronize()

                nodes, topologies = gmshnics.msh_gmsh_model(model, 2)
                mesh, entity_functions = gmshnics.mesh_from_gmsh(nodes, topologies)

                gmsh.finalize()
                
                return entity_functions[1]
                            
        except ImportError:
            print('Missing gmshnics module')
            get_geometry = lambda i: None
        
    # The remaining normals
    normals = normals + (Constant((1, 0)), Constant((0, -1)), Constant((0, 1)))    

    R = Constant(((0, -1), (1, 0)))
    # In addition to u we will need data for Neumann boundaries and gamma
    g_dir = [u]*4
    g_neu = [dot(sigma, n) for n in normals]
    # On LM boundary we have tangential data for velocity ...
    g_lm_n = [dot(u, n) for n in normals]

    # ... and also traction data in normal direction
    g_lm_t = [tangent(dot(sigma, n), n) for n in normals]    

    # For multiplier we have a setup in mind where it is defind only on
    # the left boundary
    lm, = (-dot(n, dot(sigma, n)) for n in normals[:1])

    to_expr = lambda f: ulfy.Expression(f, degree=4)
    
    solution = tuple(map(to_expr, (u, p, lm)))
    
    return {'solution': solution,
            'f': to_expr(f),
            'dirichlet': dict(enumerate(map(to_expr, g_dir), 1)),
            'neumann': dict(enumerate(map(to_expr, g_neu), 1)),
            'lagrange_t': dict(enumerate(map(to_expr, g_lm_t), 1)),
            'lagrange_n': dict(enumerate(map(to_expr, g_lm_n), 1)),            
            'get_geometry': get_geometry}


def setup_problem(facet_f, mms, flat_gamma):
    '''Babuska on [0, 1]^2'''
    Rot = Constant(((0, -1), (1, 0)))
    
    mesh = facet_f.mesh()
    n = FacetNormal(mesh)
    t = dot(Rot, n)
    ds = Measure('ds', domain=mesh, subdomain_data=facet_f)
    
    lm_tags = (1, )      #    4
    dir_tags = (3, 4)     #  1   2
    neu_tags = (2, )    #    3
    
    bmesh = EmbeddedMesh(facet_f, lm_tags)
    n_ = OuterNormal(bmesh, [0.5, 0.5])
    t_ = dot(Rot, n_)
    dx_ = Measure('dx', domain=bmesh, subdomain_data=bmesh.marking_function)

    # Velocity, pressure and LM spaces
    V = VectorFunctionSpace(mesh, 'CG', 2)
    Q = FunctionSpace(mesh, 'CG', 1)
    M = FunctionSpace(bmesh, 'CG', 2)
    W = [V, Q, M]

    u, p, l = map(TrialFunction, W)
    v, q, m = map(TestFunction, W)
    Tu, Tv = Trace(u, bmesh), Trace(v, bmesh)

    cx, cy, r = 0, 0.5, 0.5
    if flat_gamma:
        nGamma = Expression(('(x[0]-cx)/r', '(x[1]-cy)/r'), cx=cx, cy=cy, r=r, degree=1)
    else:
        nGamma = Constant((-1, 0))
    normals = {1: nGamma,
               2: Constant((1, 0)),
               3: Constant((0, -1)),
               4: Constant((0, 1))}
    # nGamma = n_
        
    # We now define the system as
    a = block_form(W, 2)
    a[0][0] = Constant(2)*inner(sym(grad(u)), sym(grad(v)))*dx
    a[0][1] = -inner(div(v), p)*dx
    a[0][2] = inner(l, dot(Tv, n_))*dx_
    a[1][0] = -inner(div(u), q)*dx
    a[2][0] = inner(m, dot(Tu, n_))*dx_
    
    f = mms['f']
    L = block_form(W, 1)
    L[0] = inner(f, v)*dx
    # Neumman contribution
    L[0] += sum(inner(mms['neumann'][tag], v)*ds(tag) for tag in neu_tags)
    # contribution from the LM boundary
    L[0] += sum(inner(mms['lagrange_t'][tag], tangent(v, n))*ds(tag) for tag in lm_tags)    
    # LM bcs
    L[2] = sum(inner(mms['lagrange_n'][tag], m)*dx_(tag) for tag in lm_tags)

    # We have strong bcs on velocity as well
    V_bcs = [DirichletBC(V, mms['dirichlet'][tag], facet_f, tag) for tag in dir_tags]
    M_bcs = [DirichletBC(M, Constant(0), 'on_boundary')]
    W_bcs = [V_bcs, [], M_bcs]

    A, b = map(ii_assemble, (a, L))
    A, b = apply_bc(A, b, W_bcs)

    # Let's also build a preconditioner
    if True:
        B0 = LU(A[0][0])
        B1 = LU(assemble(inner(p, q)*dx))
        
        M_facet_f = MeshFunction('size_t', M.mesh(), M.mesh().topology().dim()-1, 0)
        DomainBoundary().mark(M_facet_f, 1)

        # Wishful thinking (which for now only points to importance of fractional laplacian
        # h = CellDiameter(M.mesh())
        # alpha = 0.5
        # b2_form = (h**alpha)*inner(l, m)*dx
        # B2, _ = assemble_system(b2_form,
        #                         inner(Constant(0), m)*dx,
        #                         None) #DirichletBC(M, Constant(0), M_facet_f, 1))
        # B2 = LU(B2)
        
        B2 = LU(HsEig(M, s=-0.5, bcs=[(M_facet_f, 1)]).collapse())
    else:
        h_ = CellDiameter(bmesh)
        y = Constant(0.0)
        
        a_prec = block_form(W, 2)
        a_prec[0][0] = Constant(2)*inner(sym(grad(u)), sym(grad(v)))*dx
        a_prec[1][1] = inner(p, q)*dx
        a_prec[2][2] = inner(m, l)*dx

        B = ii_assemble(a_prec)
        B, _ = apply_bc(B, b, W_bcs)
        B0, B1, B2 = (LU(B[i][i]) for i in range(len(W)))
        
    B = block_diag_mat([B0, B1, B2])
        
    return A, b, W, B

# ------------------------------------------------------------------------------

if __name__ == '__main__':
    from block.algebraic.petsc import KSP
    from common import ConvergenceLog, H1_norm, L2_norm
    import sys, argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Pick LM boundary shape
    parser.add_argument('--is_flat', type=int, default=1, choices=[0, 1])
    args, _ = parser.parse_known_args()
    
    # Reduce verbosity
    set_log_level(40)

    mms = setup_mms(args.is_flat)
    u_true, p_true, lm_true = mms['solution']

    clog = ConvergenceLog({
        'u': (u_true, H1_norm, '1'),
        'p': (p_true, L2_norm, '0'),
        'lm': (lm_true, L2_norm, '0'),
        #
        'niters': None
    })

    print(clog.header())
    for i in range(7):
        facet_f = mms['get_geometry'](i)
        # Capture not gmshnics
        if facet_f is None: continue
        
        A, b, W, B = setup_problem(facet_f, mms, flat_gamma=args.is_flat)

        wh = ii_Function(W)
        
        Ainv = KSP(A, precond=B, 
                   # PETScOptions
                   ksp_type='minres',
                   ksp_rtol=1E-12,
                   # ksp_view=None,
                   # ksp_monitor_true_residual=None,
                   # ksp_initial_guess_nonzero=1,
                   ksp_converged_reason=None)        
        # NOTE: assigning from block_vec
        wh.vector()[:] = ii_convert(Ainv*b)

        # solve(ii_convert(A), wh.vector(), ii_convert(b))
       
        uh, ph, lmh = wh
        niters = len(Ainv.residuals)
        
        clog.add((uh, ph, lmh, niters))
        print(clog.report_last(with_name=False))

    iru, _ = clog['u'].get_rate()
    irp, _ = clog['p'].get_rate()
    irl, _ = clog['lm'].get_rate()
    rates = (iru, irp, irl)
    
    if args.is_flat:
        passed = all(r > 1.95 for r in rates)
    else:
        # Bit more liberal here ...
        passed = facet_f is None or all(r > 1.8 for r in rates)

    sys.exit(int(passed))
