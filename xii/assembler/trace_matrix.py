from xii.linalg.matrix_utils import petsc_serial_matrix
from xii.assembler.trace_assembly import trace_cell
from xii.assembler.fem_eval import DegreeOfFreedom, FEBasisFunction
from xii.meshing.mesh import build_embedding_map

from dolfin import Cell, info, PETScMatrix
from petsc4py import PETSc
import numpy as np


def trace_mat(V, TV, restriction='', normal=None, trace_mesh=None):
    '''
    A mapping for computing traces of function in V in TV. If f in V 
    then g in TV has coefficients equal to dofs_{TV}(trace V)
    '''
    # Compatibility of spaces
    assert V.dolfin_element().value_rank() == TV.dolfin_element().value_rank()
    assert V.ufl_element().value_shape() == TV.ufl_element().value_shape()
    assert trace_cell(V) == TV.mesh().ufl_cell()
    assert V.mesh().geometry().dim() == TV.mesh().geometry().dim()

    # FIXME: trace element checking
    if trace_mesh is not None:
        assert trace_mesh.id() ==  TV.mesh().id()
    
    # Restriction is defined using the normal
    if restriction: assert normal is not None, 'R is %s' % restriction

    # Typically with CG spaces - any parent cell can set the valeus
    if not restriction:
        Tmat = trace_mat_no_restrict(V, TV, trace_mesh=trace_mesh)
    else:
        if restriction in ('+', '-'):
            Tmat = trace_mat_one_restrict(V, TV, restriction, normal, trace_mesh)
        else:
            assert restriction in ('avg', 'jump')
            Tmat = trace_mat_two_restrict(V, TV, restriction, normal, trace_mesh)
    return PETScMatrix(Tmat)
                

def trace_mat_no_restrict(V, TV, trace_mesh=None):
    '''The first cell connected to the facet gets to set the values of TV'''
    mesh = V.mesh()
    fdim = mesh.topology().dim() - 1
    
    if trace_mesh is None: trace_mesh = TV.mesh()

    # Init/extract the mapping
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

    # Rows
    visited_dofs = [False]*TV.dim()
    # Column values
    dof_values = np.zeros(V_basis_f.elm.space_dimension(), dtype='double')
    with petsc_serial_matrix(TV, V) as mat:

        for trace_cell in range(TV.mesh().num_cells()):
            TV_dof.cell = trace_cell
            trace_dofs = Tdmap.cell_dofs(trace_cell)

            # Figure out the dofs of V to use here. Does not matter which
            # cell of the connected ones we pick
            cell = f2c(mapping[trace_cell])[0]
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


def trace_mat_one_restrict(V, TV, restriction, normal, trace_mesh=None):
    '''
    Compute the trace values using +/- restriction. A + plus is the one 
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
                cell = facet_cells[0]
            # Search which cell has the right sign
            else:
                signs = []
                for fcell in facet_cells:
                    t_mp = Cell(trace_mesh, trace_cell).midpoint().array()[:gdim]
                    mp = Cell(mesh, fcell).midpoint().array()[:gdim]

                    sign = '+' if np.inner(mp - t_mp, normal(t_mp)) > 0 else '-'
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


def get_entity_map(mesh, trace_mesh):
    '''
    Make sure that trace mesh has with it the data for mapping cells of
    TV to facets of V
    '''
    mesh_id = mesh.id()

    # There is map but we might be missing entry for the mesh
    if hasattr(trace_mesh, 'parent_entity_map'):
        # Check if we have the map embedding into mesh
        if mesh_id not in trace_mesh.parent_entity_map:
            info('Missing map for mesh %d' % mesh_id)
            parent_entity_map = build_embedding_map(trace_mesh, mesh)
            trace_mesh.parent_entity_map[mesh_id] = parent_entity_map
    # Compute from scratch and rememeber for future
    else:
        info('Computing embedding map for mesh %d' % mesh_id)

        parent_entity_map = build_embedding_map(trace_mesh, mesh)
        # If success we attach it to the mesh (to prevent future recomputing)
        trace_mesh.parent_entity_map = {mesh_id: parent_entity_map}
    return True
