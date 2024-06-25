from xii.linalg.matrix_utils import petsc_serial_matrix
from xii.assembler.trace_assembly import trace_cell
from xii.assembler.fem_eval import DegreeOfFreedom, FEBasisFunction
from xii.meshing.embedded_mesh import build_embedding_map
from xii.assembler.nonconforming_trace_matrix import nonconforming_trace_mat

from dolfin import Cell, PETScMatrix, warning, info, SubsetIterator, MeshFunction
from scipy.sparse import csr_matrix
import itertools, operator
from petsc4py import PETSc
import numpy as np

import dolfin as df
import tqdm

# Restriction operators are potentially costly so we memoize the results.
# Let every operator deal with cache keys as it sees fit
def memoize_trace(trace_mat):
    '''Cached trace'''
    cache = {}
    def cached_trace_mat(V, TV, trace_mesh, data):
        key = ((V.ufl_element(), V.mesh().id()),
               (TV.ufl_element(), TV.mesh().id()),
               data['restriction'], data['normal'], tuple(data['tag_data'][1]))
               
        if key not in cache:
            cache[key] = trace_mat(V, TV, trace_mesh, data)
        return cache[key]

    return cached_trace_mat


@memoize_trace
def trace_mat(V, TV, trace_mesh, data):
    '''
    A mapping for computing traces of function in V in TV. If f in V 
    then g in TV has coefficients equal to dofs_{TV}(trace V). Trace is 
    understood as D -> D-1.
    '''
    # Compatibility of spaces
    assert V.ufl_element().value_shape() == TV.ufl_element().value_shape()
    assert trace_cell(V) == TV.mesh().ufl_cell()
    assert V.mesh().geometry().dim() == TV.mesh().geometry().dim()

    # FIXME: trace element checking
    if trace_mesh is not None:
        assert trace_mesh.id() ==  TV.mesh().id()

    restriction = data['restriction']
    normal = data['normal']
    tag_data = data['tag_data']
    # Restriction is defined using the normal
    if restriction: assert normal is not None, 'R is %s' % restriction

    # Typically with CG spaces - any parent cell can set the valeus
    if not restriction:
        Tmat = trace_mat_no_restrict(V, TV, trace_mesh, tag_data=tag_data)
    else:
        if restriction in ('+', '-'):
            Tmat = trace_mat_one_restrict(V, TV, restriction, normal, trace_mesh, tag_data)
        else:
            assert restriction in ('avg', 'jump')
            Tmat = trace_mat_two_restrict(V, TV, restriction, normal, trace_mesh, tag_data)
    return PETScMatrix(Tmat)
                

def trace_mat_no_restrict(V, TV, trace_mesh=None, tag_data=None):
    '''The first cell connected to the facet gets to set the values of TV'''
    mesh = V.mesh()

    if trace_mesh is None: trace_mesh = TV.mesh()

    fdim = trace_mesh.topology().dim()

    # None means all
    if tag_data is None:
        try:
            marking_function = trace_mesh.marking_function
            tag_data = (marking_function, set(marking_function.array()))
        except AttributeError:
            tag_data = (MeshFunction('size_t', trace_mesh, trace_mesh.topology().dim(), 0),
                        set((0, )))

    trace_mesh_subdomains, tags = tag_data
    # Init/extract the mapping
    try:
        assert get_entity_map(mesh, trace_mesh, trace_mesh_subdomains, tags)
    except (AssertionError, IndexError):
        warning('Using non-conforming trace')
        # So non-conforming matrix returns PETSc.Mat
        return nonconforming_trace_mat(V, TV)

    if V.ufl_element().family() == 'HDiv Trace':
        assert V.ufl_element().degree() == 0
        # In this case
        return DLT_trace_mat(V, TV, trace_mesh=trace_mesh, tag_data=tag_data)

    # We can get it
    mapping = trace_mesh.parent_entity_map[mesh.id()][fdim]  # Map cell of TV to cells of V

    mesh.init(fdim, fdim+1)
    f2c = mesh.topology()(fdim, fdim+1)  # Facets of V to cell of V

    # The idea is to evaluate TV's degrees of freedom at basis functions of V
    Tdmap = TV.dofmap()
    TV_dof = DegreeOfFreedom(TV)

    dmap = V.dofmap()
    V_basis_f = FEBasisFunction(V)

    # Only look at tagged cells
    trace_cells = list(itertools.chain(*[map(operator.methodcaller('index'),
                                                   SubsetIterator(trace_mesh_subdomains, tag))
                                    for tag in tags]))


    ndofs_elm, nbasis_elm = TV_dof.elm.space_dimension(), V_basis_f.elm.space_dimension()
    local_values = np.zeros((nbasis_elm, ndofs_elm))
    
    if len(trace_cells) > 10_000:
        print(f'Trace mat {TV.ufl_element()} -> {V.ufl_element()}')
        trace_cells = tqdm.tqdm(trace_cells, total=len(trace_cells))

    rows, cols, values = [], [], []
    # DG spaces don't share rows between cells so we take advantage of
    # this in special branch
    if TV.ufl_element().family() == 'Discontinuous Lagrange':
        for trace_cell in trace_cells:
            TV_dof.cell = trace_cell
            # Many rows at once
            trace_dofs = Tdmap.cell_dofs(trace_cell)
            # Figure out the dofs of V to use here. Does not matter which
            # cell of the connected ones we pick
            cell = f2c(mapping[trace_cell])[0]
            V_basis_f.cell = cell

            # Columns for the rows
            dofs = dmap.cell_dofs(cell)
            for local, dof in enumerate(dofs):
                # Set which basis foo
                V_basis_f.dof = local
                # Get all rows at once
                # print('>>>>', TV_dof.eval_dofs(V_basis_f))
                
                local_values[local][:] = TV_dof.eval_dofs(V_basis_f)
            # Indices for the filled piece
            rows_ = np.tile(trace_dofs, nbasis_elm)
            cols_ = np.repeat(dofs, ndofs_elm)

            rows.extend(rows_)
            cols.extend(cols_)
            values.extend(local_values.flat)
    # FIXME: Othewise we need to take care of duplicate entrieselse:
    else:
        needs_fill = np.ones(TV.dim(), dtype=bool)

        for trace_cell in trace_cells:
            TV_dof.cell = trace_cell
            # Many rows at once
            trace_dofs = Tdmap.cell_dofs(trace_cell)

            # Don't add duplicates
            unseen = needs_fill[trace_dofs]   # Some will be true and 
            # For the future
            needs_fill[trace_dofs[unseen]] = False
            
            # Figure out the dofs of V to use here. Does not matter which
            # cell of the connected ones we pick
            cell = f2c(mapping[trace_cell])[0]
            V_basis_f.cell = cell

            # Columns for the rows
            dofs = dmap.cell_dofs(cell)
            for local, dof in enumerate(dofs):
                # Set which basis foo
                V_basis_f.dof = local
                # Get all rows at once
                local_values[local][:] = TV_dof.eval_dofs(V_basis_f)
            # Indices for the filled piece
            rows_ = np.tile(trace_dofs[unseen], nbasis_elm)
            cols_ = np.repeat(dofs, sum(unseen))

            rows.extend(rows_)
            cols.extend(cols_)
            values.extend(local_values[:, unseen].flat)
        
    mat = csr_matrix((values, (rows, cols)), shape=(TV.dim(), V.dim()))        

    return PETSc.Mat().createAIJ(comm=PETSc.COMM_WORLD,
                                 size=mat.shape,
                                 csr=(mat.indptr, mat.indices, mat.data))


def DLT_trace_mat(V, TV, trace_mesh=None, tag_data=None):
    '''Inject dofs from facets to DLT'''
    mesh = V.mesh()

    if trace_mesh is None: trace_mesh = TV.mesh()

    fdim = trace_mesh.topology().dim()

    # None means all
    if tag_data is None:
        try:
            marking_function = trace_mesh.marking_function
            tag_data = (marking_function, set(marking_function.array()))
        except AttributeError:
            tag_data = (MeshFunction('size_t', trace_mesh, trace_mesh.topology().dim(), 0),
                        set((0, )))

    trace_mesh_subdomains, tags = tag_data
    # Init/extract the mapping
    # We can get it
    mapping = trace_mesh.parent_entity_map[mesh.id()][fdim]  # Map cell of TV to cells of V

    if TV.num_sub_spaces() == 0:
        Tdmaps = [TV.dofmap()]
    else:
        Tdmaps = [TV.sub(i).dofmap() for i in range(TV.num_sub_spaces())]

    if V.num_sub_spaces() == 0:
        dmap = V.dofmap()
        facet2dofs = [dmap.entity_dofs(mesh, fdim)]
    else:
        facet2dofs = [V.sub(i).dofmap().entity_dofs(mesh, fdim) for i in range(V.num_sub_spaces())]

    assert len(Tdmaps) == len(facet2dofs)

    # Only look at tagged cells
    trace_cells = list(itertools.chain(*[map(operator.methodcaller('index'),
                                             SubsetIterator(trace_mesh_subdomains, tag))
                                         for tag in tags]))

    # Rows
    with petsc_serial_matrix(TV, V) as mat:

        for Tdmap, facet2dof in zip(Tdmaps, facet2dofs):
            for trace_cell in trace_cells:
                trace_dof, = Tdmap.cell_dofs(trace_cell)
                DLT_dof = facet2dof[mapping[trace_cell]]
                mat.setValues([trace_dof], [DLT_dof], [1.], PETSc.InsertMode.INSERT_VALUES)
    return mat


def trace_mat_one_restrict(V, TV, restriction, normal, trace_mesh=None, tag_data=None):
    '''
    Compute the trace values using +/- restriction. A + plus is the one 
    for which the vector cell.midpoint - facet.midpoint agrees in orientation
    with the normal on the facet.
    '''
    assert normal.function_space().ufl_element().family() == 'Discontinuous Lagrange'
    assert normal.function_space().ufl_element().degree() == 0
    
    mesh = V.mesh()
    fdim = mesh.topology().dim() - 1
    if trace_mesh is None: trace_mesh = TV.mesh()

    # None means all
    if tag_data is None:
        tag_data = (MeshFunction('size_t', trace_mesh, trace_mesh.topology().dim(), 0),
                    set((0, )))
    trace_mesh_subdomains, tags = tag_data

    # Only look at tagged cells
    trace_cells = itertools.chain(*[map(operator.methodcaller('index'),
                                                   SubsetIterator(trace_mesh_subdomains, tag))
                                    for tag in tags])
        
    # Init/extract entity map
    assert get_entity_map(mesh, trace_mesh, trace_mesh_subdomains, tags)
    # We can get it
    mapping = trace_mesh.parent_entity_map[mesh.id()][fdim]  # Map cell of TV to cells of V
        
    mesh.init(fdim, fdim+1)
    f2c = mesh.topology()(fdim, fdim+1)  # Facets of V to cell of V

    # The idea is to evaluate TV's degrees of freedom at basis functions
    # of V
    Tdmap = TV.dofmap()
    TV_dof = DegreeOfFreedom(TV)

    dmap = V.dofmap()
    V_basis_f = FEBasisFunction(V)

    gdim = mesh.geometry().dim()
    # Rows
    visited_dofs = [False]*TV.dim()
    # Column values
    dof_values = np.zeros(V_basis_f.elm.space_dimension(), dtype='double')

    normal_values = normal.vector().get_local().reshape((-1, gdim))
    with petsc_serial_matrix(TV, V) as mat:

        for trace_cell in trace_cells:
            TV_dof.cell = trace_cell
            trace_dofs = Tdmap.cell_dofs(trace_cell)

            # Figure out the dofs of V to use here
            facet_cells = f2c(mapping[trace_cell])

            assert 0 < len(facet_cells) < 3
            # Ignore boundary facets
            if len(facet_cells) == 1:
                cell = facet_cells[0]
            # Search which cell has the right sign
            else:
                signs = []
                t_mp = Cell(trace_mesh, trace_cell).midpoint().array()[:gdim]
                trace_cell_normal = normal_values[trace_cell]
                for fcell in facet_cells:
                    mp = Cell(mesh, fcell).midpoint().array()[:gdim]
                    
                    r = mp - t_mp
                    r = r / np.linalg.norm(r)

                    sign = '+' if np.inner(r, trace_cell_normal) > 0 else '-'
                    signs.append(sign)
                cell = facet_cells[signs.index(restriction)]
                    
            V_basis_f.cell = cell
            
            dofs = dmap.cell_dofs(cell)
            for local_T, dof_T in enumerate(trace_dofs):

                if visited_dofs[dof_T]:
                    continue
                else:
                    visited_dofs[dof_T] = True

                # Define trace dof
                TV_dof.dof = local_T
                
                # Eval at V basis functions
                for local, dof in enumerate(dofs):
                    # Set which basis foo
                    V_basis_f.dof = local
                    
                    dof_values[local] = TV_dof.eval(V_basis_f)

                # Can fill the matrix now
                col_indices = np.array(dofs, dtype='int32')
                # Insert
                mat.setValues([dof_T], col_indices, dof_values, PETSc.InsertMode.INSERT_VALUES)
    return mat


def trace_mat_two_restrict(V, TV, restriction, normal, trace_mesh=None):
    '''
    Compute the trace values using avg/jump restriction. A + plus is the one 
    for which the vector cell.midpoint - facet.midpoint agrees in orientation
    with the normal on the facet.
    '''
    mesh = V.mesh()
    fdim = mesh.topology().dim() - 1
    
    if trace_mesh is None: trace_mesh = TV.mesh()

    # Init/extract entity map
    assert get_entity_map(mesh, trace_mesh)
    # We can get it
    mapping = trace_mesh.parent_entity_map[mesh.id()][fdim]  # Map cell of TV to cells of V

    mesh.init(fdim, fdim+1)
    f2c = mesh.topology()(fdim, fdim+1)  # Facets of V to cell of V

    # The idea is to evaluate TV's degrees of freedom at basis functions
    # of V
    Tdmap = TV.dofmap()
    TV_dof = DegreeOfFreedom(TV)

    dmap = V.dofmap()
    V_basis_f = FEBasisFunction(V)

    # We define avg as sum(+, -)/2 and jump as sum(+, neg(-))
    operator_pieces = {'avg': (lambda x: x/2, lambda x: x/2),
                       'jump': (lambda x: x, lambda x: -x)}[restriction]

    gdim = mesh.geometry().dim()
    # Rows
    visited_dofs = [False]*TV.dim()
    # Column values
    dof_values = np.zeros(V_basis_f.elm.space_dimension(), dtype='double')
    with petsc_serial_matrix(TV, V) as mat:

        for trace_cell in range(trace_mesh.num_cells()):
            TV_dof.cell = trace_cell
            trace_dofs = Tdmap.cell_dofs(trace_cell)

            # Figure out the dofs of V to use here
            facet_cells = f2c(mapping[trace_cell])

            assert 0 < len(facet_cells) < 3
            # Ignore boundary facets
            if len(facet_cells) == 1:
                facet_cells = [facet_cells[0]]
                modifiers = (lambda x: x, )  # Do nothing
            # Search which cell has the right sign
            else:
                signs = [None, None]
                # Order such that '+' is first
                t_mp = Cell(trace_mesh, trace_cell).midpoint().array()[:gdim]
                mp = Cell(mesh, facet_cells[0]).midpoint().array()[:gdim]

                sign = '+' if np.inner(mp - t_mp, normal(t_mp)) > 0 else '-'
                # Need to flip
                if sign == '-':
                    facet_cells = facet_cells[::-1]
                # As requested
                modifiers = operator_pieces

            for local_T, dof_T in enumerate(trace_dofs):

                if visited_dofs[dof_T]:
                    continue
                else:
                    visited_dofs[dof_T] = True
                    
                # Define trace dof
                TV_dof.dof = local_T
                
                # Two sweeps to set the values in the row
                ADD_VALUES = False
                for modify, cell in zip(modifiers, facet_cells):
                    V_basis_f.cell = cell
                    dofs = dmap.cell_dofs(cell)

                    # Eval at V basis functions
                    for local, dof in enumerate(dofs):
                        # Set which basis foo
                        V_basis_f.dof = local
                        dof_values[local] = modify(TV_dof.eval(V_basis_f))
                    # Can fill the matrix now
                    col_indices = np.array(dofs, dtype='int32')

                    if not ADD_VALUES:
                        mat.setValues([dof_T], col_indices, dof_values, PETSc.InsertMode.INSERT_VALUES)
                        ADD_VALUES = True
                        #print 'setting', dof_T, col_indices, dof_values
                    else:
                        mat.setValues([dof_T], col_indices, dof_values, PETSc.InsertMode.ADD_VALUES)
                        #print 'adding', dof_T, col_indices, dof_values
    return mat


def get_entity_map(mesh, trace_mesh, subdomains=None, tags=None):
    '''
    Make sure that trace mesh has with it the data for mapping cells of
    TV to facets of V
    '''
    mesh_id = mesh.id()
    tags = set((0, )) if tags is None else set(tags)

    # There is map but we might be missing entry for the mesh
    if hasattr(trace_mesh, 'parent_entity_map'):
        assert hasattr(trace_mesh, 'tagged_cells')
        # Check if we have the map embedding into mesh
        if mesh_id not in trace_mesh.parent_entity_map:
            info('\tMissing map for mesh %d' % mesh_id)
            parent_entity_map = build_embedding_map(trace_mesh, mesh, subdomains, tags)
            trace_mesh.parent_entity_map[mesh_id] = parent_entity_map
        else:
            needed_tags = trace_mesh.tagged_cells - tags
            if needed_tags:
                info('\tMissing map for tags %r of mesh %d' % (needed_tags, mesh_id))
                parent_entity_map = build_embedding_map(trace_mesh, mesh, subdomains, tags)
                # Add new
                for edim in trace_mesh.parent_entity_map[mesh_id]:
                    trace_mesh.parent_entity_map[mesh_id][edim].update(parent_entity_map[edim])
    # Compute from scratch and rememeber for future
    else:
        info('\tComputing embedding map for mesh %d' % mesh_id)

        parent_entity_map = build_embedding_map(trace_mesh, mesh, subdomains, tags)
        # If success we attach it to the mesh (to prevent future recomputing)
        trace_mesh.parent_entity_map = {mesh_id: parent_entity_map}

        # Just made a map for those tags so attach
        if hasattr(trace_mesh, 'tagged_cells'):
            assert not trace_mesh.tagged_cells
            trace_mesh.tagged_cells.update(tags)
        else:
            trace_mesh.tagged_cells = tags
    return True
