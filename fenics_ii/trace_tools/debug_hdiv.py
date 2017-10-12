from dolfin import *
from embedded_mesh import EmbeddedMesh
from trace_assembler import trace_assemble
import numpy as np

# Here our implementation of Hdiv normal trace is compared with Marie's
# As always we have [0, 1]^2 and [0.25, 0.75]^2
Gamma = ['near((x[0]-0.25)*(x[0]-0.75), 0) && (0.25-DOLFIN_EPS < x[1]) && (x[1] < 0.75+DOLFIN_EPS)',
         'near((x[1]-0.25)*(x[1]-0.75), 0) && (0.25-DOLFIN_EPS < x[0]) && (x[0] < 0.75+DOLFIN_EPS)']
Gamma = map(lambda x: '('+x+')', Gamma)
Gamma = ' || '.join(Gamma)
Gamma = CompiledSubDomain(Gamma)

intGamma = AutoSubDomain(lambda x, on_boundary:\
                         between(x[0], (0.25, 0.75)) and between(x[1], (0.25, 0.75)))

summary = {}
for elm in ('Brezzi-Douglas-Marini', 'Raviart-Thomas'):
    for bdm_deg in (1, 2):
        dgt_deg = 0#dgt_deg = bdm_deg - 1 
        for case in (0, 1, 2, 3):
            for ncells in [10, 20, 40, 80]:
                mesh = UnitSquareMesh(4*ncells, 4*ncells)
                facet_f = FacetFunction('size_t', mesh, 0)
                Gamma.mark(facet_f, 1)
                # NOTE: from dolfin bitbucket PR 199:
                # If cell_domains is provided, the '+' and '-' sides are chosen such that the
                # cell_domains value in the cell at the '+' side cell is larger than the
                # cell_domains value in the cell at the '-' side cell. If the values are equal or
                # the cell_domains are not provided, the sides are chosen arbitrarily.
                # ----
                # Mark X=[0.25, 0.75]^2 as 1 so that n('+') [outer normal of + cell points to
                # from X to rest
                cell_f = CellFunction('size_t', mesh, 0)
                intGamma.mark(cell_f, 1)

                gamma = EmbeddedMesh(mesh, facet_f, 1, Point(0.5, 0.5))
                normal = gamma.normal

                dS = Measure('dS', domain=mesh, subdomain_data=facet_f)
                dxG = Measure('dx', domain=gamma.mesh)
            # 
                bdm = FiniteElement(elm, mesh.ufl_cell(), bdm_deg)
                BDM = FunctionSpace(mesh, bdm)
                # HACK: In order to the comparison, we need to passl the normal orientation
                # to the form. This is done by using dx in a form. However, dx does not work
                # with DGT elements, so we create a mixed space and then extract the
                # relevant compoenent
                X = MixedElement(bdm, FiniteElement('DGT', mesh.ufl_cell(), dgt_deg))
                X = FunctionSpace(mesh, X)
                if case == 0:
                    vv = interpolate(Constant((1, 1)), BDM)
                    vv_X = interpolate(Constant((1, 1, 0)), X)
                elif case == 1:
                    vv = interpolate(Expression(('x[0]', 'x[0]+x[1]'), degree=1), BDM)
                    vv_X = interpolate(Expression(('x[0]', 'x[0]+x[1]', '1'), degree=1), X)
                elif case == 2:
                    vv = interpolate(Expression(('sin(x[0])', 'cos(x[0]+x[1])'), degree=1), BDM)
                    vv_X = interpolate(Expression(('sin(x[0])', 'cos(x[0]+x[1])', '1'), degree=1), X)
                else:
                    vv = Function(BDM)
                    vv.vector()[:] = np.random.rand(vv.vector().local_size())
                    vv_X = Function(X)
                    assign(vv_X.sub(0), vv)
                vv_0, _ = vv_X.split()
                q_X0, q_X1 = TestFunctions(X)
                # The hacked exact result
                dx = Measure('dx', domain=mesh, subdomain_data=cell_f)
                nal = FacetNormal(mesh)
                b = inner(dot(vv_0('+'), nal('+')), q_X1('+'))*dS(1) + inner(Constant((0, 0)), q_X0)*dx()
                b = assemble(b)
                b_values = b.array()

                # NOTE that normal restriction is not really taken into account
                S = FunctionSpace(gamma.mesh, 'DG', dgt_deg)
                u, q = TrialFunction(BDM), TestFunction(S)
                L = inner(dot(u('+'), normal('+')), q)*dxG
                # We get the vector by action of T
                T = trace_assemble(L, gamma)

                c = Function(S)
                T.mult(vv.vector(), c.vector())
                c_values = c.vector().array()

                # Check matches for individual dofs
                mesh.init(1, 2)
                mesh.init(2, 1)

                c2f = gamma.entity_map[1]
                Qdofm = X.sub(1).dofmap()   # DGT vs ours
                Sdofm = S.dofmap()

                errors = []
                dof_map = []
                for cell in cells(gamma.mesh):
                    index = cell.index()
                    Sdof = Sdofm.cell_dofs(index)[0]

                    facet_index = c2f[index]

                    mesh_cell = Cell(mesh, Facet(mesh, facet_index).entities(2)[0])
                    Qdofs = Qdofm.cell_dofs(mesh_cell.index())
                    for i, facet in enumerate(facets(mesh_cell)):
                        if facet.index() == facet_index:
                            assert cell.midpoint().distance(Facet(mesh, facet_index).midpoint()) < 1E-13
                            break
                    Qdof = Qdofs[i]
                    mp = (cell.midpoint()[0], cell.midpoint()[1])
                    e = abs(c_values[Sdof]-b_values[Qdof])
                    # print mp, e, (b_values[Qdof], c_values[Sdof]), '||', normal(*mp)
                    errors.append((e, mp))

                    dof_map.append((Sdof, Qdof))
                e = max(errors, key=lambda p: p[0])
                print e
                # Make sure that it is not all 0
                Sselect, Qselect = map(lambda p: p[0], dof_map), map(lambda p: p[1], dof_map)
                print np.sum(np.abs(b_values[Qselect])), np.sum(np.abs(c_values[Sselect]))

                summary[(elm, bdm_deg, case, ncells)] = e
    # The above shows that for elm=any/bdm_deg=any/dgt_deg=0 there is exact mathch.
    # With dgt_deg > 0 there is some error - for now Hdiv trace will work only for
    # the exact case
k = max(summary, key=lambda key: summary[key])
v = summary[k]
print k, v
assert v[0] < 1E-12
