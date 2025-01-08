from xii.assembler.trace_matrix import trace_mat_no_restrict
import dolfin as df

# Restriction operators are potentially costly so we memoize the results.
# Let every operator deal with cache keys as it sees fit
def memoize_trace(trace_mat):
    '''Cached trace'''
    cache = {}
    def cached_trace_mat(V, TV, reduced_mesh, tag_data):
        key = ((V.ufl_element(), V.mesh().id()),
               (TV.ufl_element(), TV.mesh().id()),
               reduced_mesh.id())
               
        if key not in cache:
            cache[key] = trace_mat(V, TV, reduced_mesh, tag_data)
        return cache[key]

    return cached_trace_mat


@memoize_trace
def transpose_trace_mat(V, TV, reduced_mesh, tag_data):
    '''TODO'''
    E = trace_mat_no_restrict(V, TV, trace_mesh=reduced_mesh)
    T = E.transpose()
    return df.PETScMatrix(T)
