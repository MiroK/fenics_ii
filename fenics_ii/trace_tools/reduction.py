from dolfin import Cell, FunctionSpace, Function, Point
import numpy as np


def DLT_DG_map(DLT, DG, gamma):
    '''Compute a mapping DG to DLT dofs.'''
    assert DLT.ufl_element().family() == DG.ufl_element().family() == 'Discontinuous Lagrange'
    assert DLT.ufl_element().degree() == DG.ufl_element().degree() == 0
    assert DLT.ufl_element().value_size() == DG.ufl_element().value_size() == 1
    
    # NOTE: this can be made more general but for now I only do facet - cell
    # stuff
    gamma_dim = gamma.mesh.topology().dim()
    assert gamma.base.topology().dim() == gamma_dim + 1
    gdim = gamma.base.geometry().dim()

    dofmapg = DG.dofmap()
    dofmapO = DLT.dofmap()
    cellg_facetO = gamma.entity_map[gamma_dim]
    gamma.base.init(gamma_dim, gamma_dim+1)
    facetOcellsO = gamma.base.topology()(gamma_dim, gamma_dim+1)
    normal = gamma.normal

    mapping = {}
    for cellg, facetO in enumerate(cellg_facetO):
        dofg = dofmapg.cell_dofs(cellg)[0]
        # A facet is connected to 2 cells, we orient the two dofs as +, -
        # meaning the orientation w.r.t. to normal
        cellsO = facetOcellsO(facetO)
        dofsO = [dofmapO.cell_dofs(cell)[0] for cell in cellsO]
        
        mp = Cell(gamma.mesh, cellg).midpoint()
        x = Cell(gamma.base, cellsO[0]).midpoint() - mp
        n = Point(normal(*[mp[i] for i in range(gdim)]))
        #    -, 0
        # ---------
        #    +, 1     | -->
        # ---------
        #
        # Positive dot means that the cell 0 is -, we have a conventional for
        # order +, - so revert then
        if n.dot(x) > 0: dofsO = dofsO[::-1]

        mapping[dofg] = dofsO
    return [mapping[key] for key in sorted(mapping.keys())]
    

def reduce_coef(f, op, gamma):
    '''Represent DLT_0 function f on DG0 space over Gamma as +, -, avg, jump.'''
    # cell of gamma & dof --> 2 cells of omega & dofs reductions
    DLT = f.function_space()
    DG = FunctionSpace(gamma.mesh, 'DG', 0)
    dlt_dofs = DLT_DG_map(DLT, DG, gamma)

    # Compute values of the reduced function from f
    values = f.vector().array()
    op = {'+': lambda p: values[p[0]],
          '-': lambda p: values[p[1]],
          'avg': lambda p: (values[p[0]]+values[p[1]])/2,
          'jump': lambda p: values[p[0]]-values[p[1]]}[op]
    values = np.array(map(op, dlt_dofs))

    # Set
    f = Function(DG)
    f.vector().set_local(values)
    f.vector().apply('insert')

    return f

# ----------------------------------------------------------------------------

if __name__ == '__main__':
    from embedded_mesh import EmbeddedMesh
    from dolfin import *

    gamma = ['near((x[0]-0.25)*(x[0]-0.75), 0) && (0.25-DOLFIN_EPS < x[1]) && (x[1] < 0.75+DOLFIN_EPS)',
             'near((x[1]-0.25)*(x[1]-0.75), 0) && (0.25-DOLFIN_EPS < x[0]) && (x[0] < 0.75+DOLFIN_EPS)']
    gamma = map(lambda x: '('+x+')', gamma)
    gamma = ' || '.join(gamma)
    gamma = CompiledSubDomain(gamma)

    n = 2
    n *= 4
    omega_mesh = UnitSquareMesh(n, n, 'crossed')
    facet_f = FacetFunction('size_t', omega_mesh, 0)
    gamma.mark(facet_f, 1)
    # plot(facet_f, interactive=True)
    gamma = EmbeddedMesh(omega_mesh, facet_f, 1, Point(0.5, 0.5))

    DLT = FunctionSpace(omega_mesh, 'DG', 0)

    intGamma = AutoSubDomain(lambda x, on_boundary:\
                             between(x[0], (0.25, 0.75)) and between(x[1], (0.25, 0.75)))
    cell_f = CellFunction('size_t', omega_mesh, 0)
    intGamma.mark(cell_f, 1)

    f0 = Expression('2', degree=0)
    f1 = Expression('3', degree=0)
    
    dofmap = DLT.dofmap()
    f = Function(DLT)
    f_values = f.vector().array()
    for cell in cells(omega_mesh):
        mp = cell.midpoint()
        mp = [mp[0], mp[1]]
        if cell_f[cell] == 0:
            f_values[dofmap.cell_dofs(cell.index())] = f0(*mp)
        else:
            f_values[dofmap.cell_dofs(cell.index())] = f1(*mp)
    f.vector()[:] = f_values


    foo = {'+': lambda x: f1(x),
           '-': lambda x: f0(x),
           'avg': lambda x: (f0(x) + f1(x))/2,
           'jump': lambda x: (f1(x) - f0(x))}
    
    DG = FunctionSpace(gamma.mesh, 'DG', 0)
    x = DG.tabulate_dof_coordinates().reshape((-1, 2))

    for op, exact in foo.items():
        rf = reduce_coef(f, op, gamma)
        assert max(abs(rf(xi)-exact(xi)) for xi in x) < 1E-13
