from xii.linalg.convert import numpy_to_petsc
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
import numpy as np

# NOTE: for uniform extension the test can
# be that inner(Ev1d, p)*dxLM = inner(A*v1d, Pi(p))*dx1d
# where Pi is the surface average operator and A
# the crossection area.

# Restriction operators are potentially costly so we memoize the results.
# Let every operator deal with cache keys as it sees fit
def memoize_ext(ext_mat):
    '''Cached extension mapping'''
    cache = {}
    def cached_ext_mat(V, TV, extended_mesh, data):
        key = ((V.ufl_element(), V.mesh().id()),
               (TV.ufl_element(), TV.mesh().id()),
               data['type'])
               
        if key not in cache:
            cache[key] = extension_mat(V, TV, extended_mesh, data)
        return cache[key]

    return cached_ext_mat


#@memoize_ext
def extension_mat(V, EV, extended_mesh, data):
    '''
    # FIXME
    '''
    assert EV.mesh().id() == extended_mesh.id()
    
    assert V.mesh().topology().dim() == 1
    assert EV.mesh().topology().dim() == 2

    assert V.ufl_element().degree() == EV.ufl_element().degree()
    assert V.ufl_element().family() == EV.ufl_element().family()
    assert V.mesh().geometry().dim() == EV.mesh().geometry().dim() == 3

    if data['type'] == 'uniform':
        return uniform_extension_matrix(V, EV)
    else:
        assert False
        

def uniform_extension_matrix(V, EV):
    '''
    Map vector of coeficients of V(over 1d domain) to 
    vector of coefficients of EV(over 3d domain). The spaces
    need to used the same element type.
    '''
    # Assumption checks
    
    V_dofs_x = V.tabulate_dof_coordinates().reshape((V.dim(), -1))
    EV_dofs_x = EV.tabulate_dof_coordinates().reshape((EV.dim(), -1))
    # Compute distance from every EV dof(row) to every V dof(column)
    lookup = cdist(EV_dofs_x, V_dofs_x)
    # Make sure the two domains do not intersect
    assert np.linalg.norm(lookup, np.inf) > 0

    # Now get the closest dof to E
    columns = np.argmin(lookup, axis=1)
    # As csr (1 col per row)
    values = np.ones_like(columns)
    rows = np.arange(len(columns)+1)

    E = csr_matrix((values, columns, rows), shape=(EV.dim(), V.dim()))

    return numpy_to_petsc(E)
