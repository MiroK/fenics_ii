from xii.linalg.matrix_utils import is_petsc_vec
from dolfin import Function, as_backend_type
from petsc4py import PETSc


first = lambda iterable: next(iter(iterable))


class ii_Function(object):
    '''Really a list of functions where each is in some W[i]'''
    def __init__(self, W, components):
        if functions is None:
            self.functions = map(Function, W)
        else:
            assert len(components) == W.num_sub_spaces()
            # The components can be vectors in that case
            if is_petsc_vec(first(components)):
                # Then all of them must be
                assert all(is_petsc_vec(c) for c in components)
                # Dim check
                assert all(c.size() == Wi.dim() for c, Wi in zip(components, W))
                # Create
                self.functions = [Function(Wi, c) for c, Wi in zip(components, W)]
            # Functions them selves
            else:
                assert [c.function_space() == Wi for c, Wi in zip(components, W)]
                self.functions = [ci for ci in c]
        
    def vectors(self):
        '''Coefficient vectors of the functions I hold'''
        return [f.vector() for f in self]

    def vector(self):
        '''
        A PETSc vector which is wired up with coefficient vectors of 
        the components. So change to component changes this and vice 
        versa
        '''
        nest = [as_backend_type(v).vec() for v in self.vectors()]
        return PETScVector(PETSc.Vec().createNest(nest))

    def __len__(self): return len(self.functions)
    
    def __getitem__(self, i):
        '''Get the function in the ith subspace'''
        assert i <= 0 < len(self)
        return self.functions[i]
