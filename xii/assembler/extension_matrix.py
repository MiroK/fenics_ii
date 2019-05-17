from xii.linalg.convert import numpy_to_petsc
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
import dolfin as df
import numpy as np


# Extension operators are potentially costly so we memoize the results.
# Let every operator deal with cache keys as it sees fit
def memoize_ext(ext_mat):
    '''Cached extension mapping'''
    cache = {}
    def cached_ext_mat(V, TV, extended_mesh, data):
        key = ((V.ufl_element(), V.mesh().id()),
               (TV.ufl_element(), TV.mesh().id()),
               data['type'])
        
        if key not in cache:
            cache[key] = ext_mat(V, TV, extended_mesh, data)
        return cache[key]

    return cached_ext_mat


@memoize_ext
def extension_mat(V, EV, extended_mesh, data):
    '''
    Dispatch to individual methods for extending functions from V to 
    EV(on extended_mesh)
    '''
    assert EV.mesh().id() == extended_mesh.id()
    
    assert V.mesh().topology().dim() == 1
    assert EV.mesh().topology().dim() in (1, 2)  # Line to line or line to triangle

    assert V.ufl_element().degree() == EV.ufl_element().degree()
    assert V.ufl_element().family() == EV.ufl_element().family()
    assert V.mesh().geometry().dim() == EV.mesh().geometry().dim()
    assert V.mesh().geometry().dim() in (2, 3)

    # NOTE: add more here
    # - something based on radial basis function
    # - using the method of Green's functions
    return {'uniform': uniform_extension_matrix(V, EV)}[data['type']]


def uniform_extension_matrix(V, EV):
    '''
    Map vector of coeficients of V(over 1d domain) to vector of coefficients of 
    EV(over 2d domain). The spaces need to use the same element type.
    '''
    gdim = V.mesh().geometry().dim()
    assert gdim == EV.mesh().geometry().dim()
    
    # For Vector and Tensor elements more than 1 degree of freedom is 
    # associated with the same geometric point. It is therefore cheaper
    # to compute the mapping based only on the scalar/one subspace considerations.
    is_tensor_elm = isinstance(V.ufl_element(), (df.VectorElement, df.TensorElement))
    # Base on scalar
    if is_tensor_elm:
        V_dofs_x = V.sub(0).collapse().tabulate_dof_coordinates().reshape((-1, gdim))
        EV_dofs_x = EV.sub(0).collapse().tabulate_dof_coordinates().reshape((-1, gdim))
    # Otherwise 'scalar', (Hdiv element belong here as well)
    else:
        V_dofs_x = V.tabulate_dof_coordinates().reshape((V.dim(), gdim))
        EV_dofs_x = EV.tabulate_dof_coordinates().reshape((EV.dim(), gdim))
        
    # Compute distance from every EV dof(row) to every V dof(column)
    lookup = cdist(EV_dofs_x, V_dofs_x)
    # Make sure the two domains do not intersect
    assert np.linalg.norm(lookup, np.inf) > 0

    # Now get the closest dof to E
    columns = np.argmin(lookup, axis=1)
    # Every scalar can be used used to set all the components
    if is_tensor_elm:
        shift = V.dim()/len(V_dofs_x)
        component_idx = np.arange(shift)
        # shift*dof + components
        columns = (shift*np.array([columns]).T + component_idx).flatten()

    # As csr (1 col per row)
    values = np.ones_like(columns)
    rows = np.arange(EV.dim()+1)

    E = csr_matrix((values, columns, rows), shape=(EV.dim(), V.dim()))

    return numpy_to_petsc(E)
