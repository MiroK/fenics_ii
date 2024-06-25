import dolfin


class BlockNullspace(object):
    '''l^2 orthonormal basis'''
    def __init__(self, W, vectors=None):
        if vectors is None:
            self.W = W
            self.vectors = []
            return

        BlockNullspace.__init__(self, W)
        list(map(self.add_vector, vectors))

    def __len__(self):
        return len(self.vectors)

    def __iter__(self):
        for v in self.vectors:
            yield v

    def add_vector(self, vec):
        '''Try adding new linearly indep vector'''
        # Orthogonalize wrt previous
        vec = vec.copy()
        # Substite constancts
        for i, vi in enumerate(vec):
            if isinstance(vi, dolfin.Constant):
                vec[i] = interpolate(vi, self.W[i]).vector()
            elif isinstance(vi, (float, int)):
                vec[i] = interpolate(Constant(vi), self.W[i]).vector()
            elif isinstance(vi, (list, tuple)):
                vec[i] = interpolate(Constant(*vi), self.W[i]).vector()
                
        for b, alpha in zip(self, self.angles(vec)):
            assert abs(abs(alpha/vec.norm())-1) > 1E-10
            vec -= alpha*b
        # Normalize    
        vec *= 1./vec.norm()

        self.vectors.append(vec)

    def orthogonalize(self, vec):
        '''Remove from b the part in the nullspace'''
        for b, alpha in zip(self, self.angles(vec)):
            vec -= alpha*b
        return vec

    def angles(self, vec):
        return [vec.inner(v) for v in self]

# --------------------------------------------------------------------

if __name__ == '__main__':
    from dolfin import *
    from block import block_vec
    import numpy as np

    mesh = UnitSquareMesh(32, 32)
    V = FunctionSpace(mesh, 'CG', 1)
    W = [V, FunctionSpace(mesh, 'CG', 2)]
    
    x = block_vec([interpolate(Constant(1), V).vector(), interpolate(Constant(2), V).vector()])

    Z = BlockNullspace(W)
    Z.add_vector(x)
    assert len(Z) == 1
    
    x = block_vec([0, interpolate(Constant(3), V).vector()])
    Z.add_vector(x)
    assert len(Z) == 2

    y = block_vec([interpolate(Constant(1), V).vector(), interpolate(Constant(1), V).vector()])
    y = Z.orthogonalize(y)
    assert y.norm() < 1E-10

    Z = BlockNullspace(W, [x, y])
    assert len(Z) == 2
