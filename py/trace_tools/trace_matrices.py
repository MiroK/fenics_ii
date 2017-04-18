from dolfin import *
from petsc4py import PETSc
import numpy as np

# FIXME: none of these really work in parallel
def Lagrange_trace_matrix(space, trace_space):
    '''Map from space to trace space of Lagrange elements.'''
    V, Q = space, trace_space
    assert V.ufl_element().family() == Q.ufl_element().family()
    assert V.ufl_element().family() == 'Lagrange'
    assert V.ufl_element().value_shape() == Q.ufl_element().value_shape()

    Vmesh, Vtdim = V.mesh(), V.mesh().topology().dim()
    Qmesh, Qgdim = Q.mesh(), Q.mesh().geometry().dim()

    # NOTE: for greater readability make rank > 0 implementation a separate thing
    # Yes, it could be implemented as part of this foo as well...
    if V.ufl_element().value_size() > 1:
        return vector_Lagrange_trace_matrix(space, trace_space)
    
    # The trace matrix is l_row(T(phi_col)) where l_row are degree of freedom of
    # Q and phi_col are basis functions of V
    tree = Vmesh.bounding_box_tree()
    limit = Vmesh.topology().size_global(Vtdim)

    # For Qcells we need to get dofs(coordinates) where we will evaluate.
    Qdofmap = Q.dofmap()
    Qdofsx = Q.tabulate_dof_coordinates().reshape((-1, Qgdim))
    Qfirst, Qlast = Qdofmap.ownership_range()

    Vdofmap = V.dofmap()
    Vfirst, Vlast = Vdofmap.ownership_range()
    Vel = V.element()

    comm = Vmesh.mpi_comm().tompi4py()
    mat = PETSc.Mat()
    mat.create(comm)
    mat.setSizes([[Qdofmap.index_map().size(IndexMap.MapSize_OWNED),
                   Qdofmap.index_map().size(IndexMap.MapSize_GLOBAL)],
                  [Vdofmap.index_map().size(IndexMap.MapSize_OWNED),
                   Vdofmap.index_map().size(IndexMap.MapSize_GLOBAL)]])
    mat.setType('aij')
    mat.setUp()
    # Local to global
    row_lgmap = PETSc.LGMap().create(map(int, Qdofmap.tabulate_local_to_global_dofs()), comm=comm)
    col_lgmap = PETSc.LGMap().create(map(int, Vdofmap.tabulate_local_to_global_dofs()), comm=comm)
    mat.setLGMap(row_lgmap, col_lgmap)
    # Qdof -> Qcell -> Vcell -> Vdofs: evaluate basis functions Vdofs at Qdof => a col

    space_dim = Vel.space_dimension()
    basis_values = np.zeros(space_dim)
    mat.assemblyBegin()
    for Qdof in xrange(Qlast - Qfirst):
        Qdof_global = Qdofmap.local_to_global_index(Qdof)
        if not Qfirst <= Qdof_global < Qlast: continue

        # Dofs are point evaluations
        x = Qdofsx[Qdof]
        c = tree.compute_first_entity_collision(Point(*x))
        if c >= limit: continue

        cell = Cell(Vmesh, c)
        # Now build row
        col_values = []
        col_indices = []
        # At each cell we want to evaluate basis functions at x
        vertex_coordinates = cell.get_vertex_coordinates()
        cell_orientation = cell.orientation()
        Vel.evaluate_basis_all(basis_values, x, vertex_coordinates, cell_orientation)

        Vdofs = Vdofmap.cell_dofs(c)

        for value, dof in zip(basis_values, Vdofs):
            dof_global = Vdofmap.local_to_global_index(dof)
            if Vfirst <= dof_global < Vlast:
                col_values.append(value)
                col_indices.append(dof_global)
        # Order. NOTE: I am not filtering out zeros
        col_values, col_indices = np.array(col_values, dtype='double'), np.array(col_indices, dtype='int32')
        ordered = np.argsort(col_indices)
        col_indices = col_indices[ordered]
        col_values = col_values[ordered]
        row_indices = [Qdof_global]
        mat.setValues(row_indices, col_indices, col_values, PETSc.InsertMode.INSERT_VALUES)
    mat.assemblyEnd()

    return mat


def vector_Lagrange_trace_matrix(space, trace_space):
    '''Map from space to trace space of Lagrange elements.'''
    V, Q = space, trace_space
    assert V.ufl_element().family() == Q.ufl_element().family()
    assert V.ufl_element().family() == 'Lagrange'
    assert V.ufl_element().value_shape() == Q.ufl_element().value_shape()

    Vmesh, Vtdim = V.mesh(), V.mesh().topology().dim()
    Qmesh, Qgdim = Q.mesh(), Q.mesh().geometry().dim()

    # NOTE: for greater readability make rank > 0 implementation a separate thing
    nsubs = V.ufl_element().value_size()
    assert  nsubs > 1 and nsubs == V.num_sub_spaces() == Q.num_sub_spaces()
    
    # The trace matrix is l_row(T(phi_col)) where l_row are degree of freedom of
    # Q and phi_col are basis functions of V. However one needs to distinguish
    # between components (of vector/tensor)
    tree = Vmesh.bounding_box_tree()
    limit = Vmesh.topology().size_global(Vtdim)

    # For Qcells we need to get dofs(coordinates) where we will evaluate.
    Qdofmap = Q.dofmap()
    Qdofsx = Q.tabulate_dof_coordinates().reshape((-1, Qgdim))
    Qfirst, Qlast = Qdofmap.ownership_range()

    Vdofmap = V.dofmap()
    Vfirst, Vlast = Vdofmap.ownership_range()
    Vel = V.element()

    comm = Vmesh.mpi_comm().tompi4py()
    mat = PETSc.Mat()
    mat.create(comm)
    mat.setSizes([[Qdofmap.index_map().size(IndexMap.MapSize_OWNED),
                   Qdofmap.index_map().size(IndexMap.MapSize_GLOBAL)],
                  [Vdofmap.index_map().size(IndexMap.MapSize_OWNED),
                   Vdofmap.index_map().size(IndexMap.MapSize_GLOBAL)]])
    mat.setType('aij')
    mat.setUp()
    # Local to global
    row_lgmap = PETSc.LGMap().create(map(int, Qdofmap.tabulate_local_to_global_dofs()), comm=comm)
    col_lgmap = PETSc.LGMap().create(map(int, Vdofmap.tabulate_local_to_global_dofs()), comm=comm)
    mat.setLGMap(row_lgmap, col_lgmap)
    # Qdof -> Qcell -> Vcell -> Vdofs: evaluate basis functions Vdofs at Qdof => a col

    mat.assemblyBegin()
    # FIXME: REALLY ONLY SERIAL
    for comp in range(nsubs):
        V_comp_map = V.sub(comp).dofmap()

        Vel = V.sub(comp).element()
        space_dim = Vel.space_dimension()
        basis_values = np.zeros(space_dim)

        for Qdof in Q.sub(comp).dofmap().dofs():

            # Dofs are point evaluations
            x = Qdofsx[Qdof]
            c = tree.compute_first_entity_collision(Point(*x))
            if c >= limit: continue

            cell = Cell(Vmesh, c)
            # Now build row
            col_values = []
            col_indices = []
            # At each cell we want to evaluate basis functions at x
            vertex_coordinates = cell.get_vertex_coordinates()
            cell_orientation = cell.orientation()
            Vel.evaluate_basis_all(basis_values, x, vertex_coordinates, cell_orientation)

            # Select value of the components
            values = np.array(basis_values, dtype='double')

            cols = np.array(V_comp_map.cell_dofs(c), dtype='int32')

            row_indices = [Qdof]
            mat.setValues(row_indices, cols, values, PETSc.InsertMode.INSERT_VALUES)
        mat.assemblyEnd()

    return mat


def BRM_RT_trace_matrix(space, trace_space, (normal, entity_map, restriction)):
    '''Map from BDM(vec).n -> DG(scalar)'''
    # NOTE: Let sigma in BDM and q in DG. We are after q such that q = sigma.n
    # If {phi_i}, {L_j} are respectively the basis functions and dofs of DG and
    # U the coefficient vector of q then U_k = L_k(sigma_n) = sigma.n(x_k) and
    # that's how we build the matrix

    # Spaces assumptions
    V, Q = space, trace_space
    assert V.ufl_element().family() in ('Brezzi-Douglas-Marini', 'Raviart-Thomas')
    assert V.element().value_rank() == 1
    assert Q.ufl_element().family() == 'Discontinuous Lagrange'
    assert Q.element().value_rank() == 0
    assert normal.value_rank() == 1

    # Meshes assumptions: the top. dimensionality gap is 1
    omega, gamma = V.mesh(), Q.mesh()
    tdimo, tdimg = omega.topology().dim(), gamma.topology().dim()
    assert tdimo == tdimg + 1
    gdim = gamma.geometry().dim()
    assert gdim == omega.geometry().dim()

    # For evaluating for each cell of gamma/facet of Omega look up cells of
    # Omega connected to it and evaluate according to restriction
    tdim = tdimg
    cellg_faceto = entity_map[tdim]  # cell of Gamma to facet of Omega

    omega.init(tdim, tdim+1)
    faceto_cello = omega.topology()(tdim, tdim+1)   # facet to cell of Omega

    Vdofm = V.dofmap()
    Qdofm = Q.dofmap()
    Qdofsx = Q.tabulate_dof_coordinates().reshape((-1, gdim))

    Vel = V.element()
    space_dim = Vel.space_dimension()
    value_size = V.ufl_element().value_size()
    basis_values = np.zeros(space_dim*value_size)

    # Now declare the matrix
    comm = omega.mpi_comm().tompi4py()
    assert comm.size == 1, 'Serial only'
    mat = PETSc.Mat()
    mat.create(comm)
    mat.setSizes([[Qdofm.index_map().size(IndexMap.MapSize_OWNED),
                   Qdofm.index_map().size(IndexMap.MapSize_GLOBAL)],
                  [Vdofm.index_map().size(IndexMap.MapSize_OWNED),
                   Vdofm.index_map().size(IndexMap.MapSize_GLOBAL)]])
    mat.setType('aij')
    mat.setUp()
    # Local to global
    row_lgmap = PETSc.LGMap().create(map(int, Qdofm.tabulate_local_to_global_dofs()), comm=comm)
    col_lgmap = PETSc.LGMap().create(map(int, Vdofm.tabulate_local_to_global_dofs()), comm=comm)
    mat.setLGMap(row_lgmap, col_lgmap)

    # Build
    for cellg in cells(gamma):
        indexg = cellg.index()
        # Map to facet of Omega
        indexo = cellg_faceto[indexg]
        # Get the two cells of Omega connected to it
        cellso = faceto_cello(indexo)
        assert len(cellso) == 2, cellso
        # Figure out orientation
        mpo, mpg = Cell(omega, cellso[0]).midpoint(), cellg.midpoint()
        r = mpo - mpg
        mpg = [mpg[i] for i in range(gdim)]
        n = Point(normal(mpg))  

        # We used first cell, + for being pointed to by the normal
        if r.dot(n) > 0:
            restrictions = ['+', '-']
        else:
            restrictions = ['-', '+']
        # Get cell to be used for restiction
        cello = cellso[restrictions.index(restriction)]
        n = np.array([n[i] for i in range(gdim)])  # Normal only here at midpoint. It's constant

        rows = Qdofm.cell_dofs(indexg)
        for row in rows:
            row_x = Qdofsx[row]

            cell = Cell(omega, cello)
            # Eval basis functions (so get sigma)
            vertex_coordinates = cell.get_vertex_coordinates()
            cell_orientation = cell.orientation()
            Vel.evaluate_basis_all(basis_values, row_x, vertex_coordinates, cell_orientation)
        
            # The columns candidates and values as sigma.n
            column_indices = Vdofm.cell_dofs(cello)
            column_values = [np.inner(sigma, n)
                             for sigma in basis_values.reshape((space_dim, value_size))]
            # Order. NOTE: I am not filtering out zeros
            col_values, col_indices = np.array(column_values, dtype='double'), np.array(column_indices, dtype='int32')
            row_indices = [row]
            mat.setValues(row_indices, col_indices, col_values, PETSc.InsertMode.INSERT_VALUES)
    mat.assemblyEnd()

    return mat

# ----------------------------------------------------------------------------

if __name__ == '__main__':
    import test_trace_matrices as test
    from test_trace_matrices import Hdiv_test

    # CG for Lagrange and scalar
    args = ([FunctionSpace, 'CG', 2], [FunctionSpace, 'CG', 1])
    tmat = Lagrange_trace_matrix
    test.test(tmat, args, 'scalar')

    # CG for Lagrange and vector
    args = ([VectorFunctionSpace, 'CG', 1], [VectorFunctionSpace, 'CG', 1])
    tmat = Lagrange_trace_matrix
    test.test(tmat, args, 'vector')

    # Hdiv
    import debug_hdiv
