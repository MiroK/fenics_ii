from ufl_legacy.corealg.traversal import traverse_unique_terminals
from xii.assembler.ufl_utils import *

from xii.linalg.matrix_utils import petsc_serial_matrix
from xii.assembler.trace_assembly import trace_cell
from xii.assembler.fem_eval import DegreeOfFreedom, FEBasisFunction
from xii.meshing.embedded_mesh import build_embedding_map
from xii.assembler.trace_matrix import get_entity_map

from xii.linalg.convert import numpy_to_petsc
from scipy.sparse import csr_matrix, lil_matrix

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
               data['restriction'], data['normal'], tuple(data['tag_data'][1]), data['expr'])
               
        if key not in cache:
            cache[key] = eval_dlt_trace_mat(V, TV, trace_mesh, data)
        return cache[key]

    return cached_trace_mat


#@memoize_trace
def eval_dlt_trace_mat(V, TV, trace_mesh, data):
    '''Eval in DLT and restrict'''
    
    # Compatibility of spaces
    assert V.ufl_element().value_shape() == TV.ufl_element().value_shape()
    assert trace_cell(V) == TV.mesh().ufl_cell()
    assert V.mesh().geometry().dim() == TV.mesh().geometry().dim()

    mesh = V.mesh()
    if trace_mesh is None:
        trace_mesh = TV.mesh()
    assert trace_mesh.id() ==  TV.mesh().id()

    fdim = trace_mesh.topology().dim()

    tag_data = data['tag_data']
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
    # get_entity_map(mesh, trace_mesh, trace_mesh_subdomains, tags)    
    assert get_entity_map(mesh, trace_mesh, trace_mesh_subdomains, tags)

    facet_f = MeshFunction('size_t', mesh, fdim, 0)

    trace_cells = np.fromiter(trace_mesh.parent_entity_map[mesh.id()][fdim].values(), dtype='uintp')
    facet_f.array()[trace_cells] = 1
    # Make sure that this is only surface measure ...
    dS = df.Measure('dS', domain=mesh, subdomain_data=facet_f)
    assert df.near(df.assemble(df.Constant(1)*dS(1)), 0, 1E-10*mesh.hmin())
    # Now the measure for projecting ...
    ds = df.Measure('ds', domain=mesh, subdomain_data=facet_f)

    expr = data['expr']
    # Now the idea is that we want to compute 1) matrix for assemblong the rhs
    # of projectioning from the space of the argument of the expr V to corresponding
    # DLT space Q
    Q = df.FunctionSpace(mesh, TV.ufl_element().reconstruct(cell=mesh.ufl_cell(), family='Discontinuous Lagrange Trace'))
    q = df.TestFunction(Q)
    
    v = df.TrialFunction(V)
    arg, = set(filter(is_argument, traverse_terminals(expr)))

    expr_v = replace(expr, arg, v)
    R = df.assemble(df.inner(q, expr_v)*ds(1))   # RHS form matrix

    # 2) Then to complete the projection we went to go apply the mass inverse of Q
    invM = inverse_mass_matrix(Q, trace_cells)

    print(V.ufl_element(),
          Q.ufl_element(),
          TV.ufl_element())
    # 3) And the finally we want to restrict 
    # E = DLT_trace_mat(V, TV, trace_mesh=trace_mesh, tag_data=tag_data)
    E = embedding_mat(Q, TV, trace_mesh, tag_data)
    
    return E*invM*R

# --------------------------------------------------------------------

def embedding_mat(V, TV, trace_mesh=None, tag_data=None):
    '''Inject dofs from facets to DLT'''
    mesh = V.mesh()
    assert V.ufl_element().family() == 'HDiv Trace'
    assert TV.ufl_element().family() in ('Discontinuous Lagrange', 'Quadrature')
    assert V.ufl_element().degree() == TV.ufl_element().degree()

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
    mapping = trace_mesh.parent_entity_map[mesh.id()][fdim]  # Map cell of TV to facets of V

    TV_dm = TV.dofmap()  # Per cell of trace mesh, i.e. facet

    nfacets = mesh.num_entities(fdim)
    V_dm = np.array(V.dofmap().entity_dofs(mesh, fdim)).reshape((nfacets, -1))

    # Only look at tagged cells
    trace_cells = list(itertools.chain(*[map(operator.methodcaller('index'),
                                             SubsetIterator(trace_mesh_subdomains, tag))
                                         for tag in tags]))
    # ---
    E = lil_matrix((TV.dim(), V.dim()))

    TV_elm = TV.element() 
    V_dofs_x = V.tabulate_dof_coordinates()

    orientations = np.zeros(trace_mesh.num_cells(), dtype=int)
    if len(trace_mesh.cell_orientations()) != 0:
        orientations[:] = np.asarray(trace_mesh.cell_orientations(), dtype=orientations.dtype)

        
    local_dim = len(TV_dm.cell_dofs(0))

    shape = TV.ufl_element().value_shape()
    if shape == ():
        value_size = 1
        columns = np.zeros(local_dim, dtype='uintp')
    else:
        value_size = np.prod(shape)
        columns = np.repeat(np.arange(value_size), local_dim/value_size)

    local_mat = np.zeros((local_dim, local_dim))
    # Rows
    for trace_cell in tqdm.tqdm(trace_cells, desc=f'Embedding of {V.ufl_element()} to DG {TV.ufl_element()}'):  # i.e. facet
        TV_dofs = TV_dm.cell_dofs(trace_cell)

        V_dofs = V_dm[mapping[trace_cell]]

        cell = Cell(trace_mesh, trace_cell)
        orientation = orientations[trace_cell]
        # We want to express DLT functions as linear combinations of DG
        # Transofrmation matrix is based on evaluating degrees of freedom of DLT
        # which given eye as the rhs. NOTE: DLT dofs are point evaluations
        for (row, xk) in enumerate(V_dofs_x[V_dofs]):
            basis_values = TV_elm.evaluate_basis_all(xk, cell.get_vertex_coordinates(), orientation).reshape((local_dim, value_size))
            local_mat[row][:] = basis_values[:, columns[row]]  # DOF for vector is point value of component
        E[np.ix_(TV_dofs, V_dofs)] = np.linalg.inv(local_mat)

    return numpy_to_petsc(E.tocsr())        


def inverse_mass_matrix(Q, trace_cells=None):
    '''Assemble'''
    elm = Q.ufl_element()
    assert elm.family() == 'HDiv Trace'

    mesh = Q.mesh()
    cell = elm.cell()

    dm = Q.dofmap()
    fdim = cell.topological_dimension()-1
    ndofs_per_facet = dm.num_entity_dofs(fdim)

    facet_dofs = np.array(dm.entity_dofs(Q.mesh(), fdim)).reshape((-1, ndofs_per_facet))
    
    p, q = df.TrialFunction(Q), df.TestFunction(Q)
    M = df.assemble(df.inner(p, q)*df.ds + df.inner(p('+'), q('+'))*df.dS)

    M_csr = csr_matrix(df.as_backend_type(M).mat().getValuesCSR()[::-1])

    if trace_cells is None:
        mesh.init(fdim)
        trace_cells = np.arange(mesh.num_entities(fdim))
    else:
        trace_cells = np.array(trace_cells)

    iM_lil = lil_matrix(M_csr.shape)
    for this_facet_dofs in tqdm.tqdm(facet_dofs[trace_cells], desc=f'{Q.ufl_element()} DLT inverse {M_csr.shape}'):
        idx = np.ix_(this_facet_dofs, this_facet_dofs)
        M_loc = M_csr[idx].todense()
        iM_lil[idx] = np.linalg.inv(M_loc)

    return numpy_to_petsc(iM_lil.tocsr())

# --------------------------------------------------------------------

if __name__ == '__main__':
    pass

    from dolfin import *

