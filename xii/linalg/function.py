from dolfin import Function, as_backend_type, PETScVector
from petsc4py import PETSc


first = lambda iterable: next(iter(iterable))


class ii_Function(object):
    '''Really a list of functions where each is in some W[i]'''
    def __init__(self, W, components=None):
        if components is None:
            self.functions = map(Function, W)
        else:
            assert len(components) == len(W)
            # Functions them selves
            if hasattr(first(components), 'function_space'):
                assert [c.function_space() == Wi for c, Wi in zip(components, W)]
                self.functions = [ci for ci in c]                
            # The components can be vectors in that case
            else:
                # Dim check
                assert all(c.size() == Wi.dim() for c, Wi in zip(components, W))
                # Create
                self.functions = [Function(Wi, c) for c, Wi in zip(components, W)]

    def vectors(self):
        '''Coefficient vectors of the functions I hold'''
        return [f.vector() for f in self.functions]

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
        assert 0 <= i < len(self), (i, len(self))
        return self.functions[i]

    def __iter__(self):
        for i in range(len(self)): yield self[i]
