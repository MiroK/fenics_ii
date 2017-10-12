from dolfin import Cell, Point, PETScMatrix, PETScVector
from petsc4py import PETSc
import numpy as np


def PointConstraintData(relations, spaces):
    '''
    We want to build a representation of constraints p0(x) - p1(x) = 0 where
    p0, p1 are trial functions of spaces and x is a constrained point. 

    The constraints are described in relations table made of tuples with :
      + space index, - space index, point
    The target is cbc.block so for each space a matrix is build with
    nrows=len(relations) and ncols=dim(space). We also build the transpose and a
    corresponding vector for the rhs. The multiplier space is thus R^nrows
    '''
    # The matrix is space0 space1 .... as a global system
    # First for each point search the dofs of the space that are effected by the
    # constraint
    entries = {}
    for row, cstr in enumerate(relations):
        space_indices, x = cstr[:2], cstr[2]
        for sign, index in zip(('+', '-'), space_indices):
            space = spaces[index]

            mesh = space.mesh()
            tree = mesh.bounding_box_tree()

            elm = space.element()

            space_dim = elm.space_dimension()
            values = np.zeros(space_dim, dtype=float)
        
            c = tree.compute_first_entity_collision(Point(*x))
            assert c > -1

            cell = Cell(mesh, c)
            # We want to evaluate basis functions at x
            vertex_coordinates = cell.get_vertex_coordinates()
            cell_orientation = cell.orientation()
            elm.evaluate_basis_all(values, x, vertex_coordinates, cell_orientation)
            # Local to space dofs
            dofs = space.dofmap().cell_dofs(c).tolist()
            if sign == '-': values *= -1

            if not index in entries:
                entries[index] = [(row, 
                                   np.array(dofs, dtype='int32'),
                                   np.array(values, dtype='double'))]
            else:
                entries[index].append((row, 
                                       np.array(dofs, dtype='int32'),
                                       np.array(values, dtype='double')))
    mats, matsT = [], []
    # We now build matrix for EACH space
    for index in entries:
        space = spaces[index]
        # Build the matrixG
        comm = space.mesh().mpi_comm().tompi4py()
        mat = PETSc.Mat()
        mat.create(comm)
        mat.setSizes([len(relations), space.dim()])
        mat.setType('aij')
        mat.setUp()
        # Local to global
        row_lgmap = PETSc.LGMap().create(map(int, np.arange(len(relations))), comm=comm)

        col_lgmap = space.dofmap().tabulate_local_to_global_dofs().tolist()
        col_lgmap = PETSc.LGMap().create(map(int, col_lgmap), comm=comm)
        mat.setLGMap(row_lgmap, col_lgmap)

        mat.assemblyBegin()
        for row, columns, values in entries[index]:
            mat.setValues([row], columns, values, PETSc.InsertMode.INSERT_VALUES)
        mat.assemblyEnd()

        matT = PETSc.Mat()
        mat.transpose(matT)
        matT.setLGMap(col_lgmap, row_lgmap)

        mats.append(mat)
        matsT.append(matT)

    rhs = PETSc.Vec().createWithArray(np.zeros(len(relations)))

    return map(lambda x: PETScMatrix(x), mats),\
           map(lambda x: PETScMatrix(x), matsT),\
           PETScVector(rhs)

# ----------------------------------------------------------------------------

if __name__ == '__main__':
    from embedded_mesh import EmbeddedMesh
    from dolfin import *

    mesh = UnitSquareMesh(20, 20)

    left = CompiledSubDomain('near(x[0], 0) && on_boundary')
    top = CompiledSubDomain('near(x[1], 1) && on_boundary')

    boundaries = FacetFunction('size_t', mesh, 0)
    left.mark(boundaries, 1)
    top.mark(boundaries, 2)

    mesh1 = EmbeddedMesh(mesh, boundaries, 1).mesh
    mesh2 = EmbeddedMesh(mesh, boundaries, 2).mesh

    V1 = FunctionSpace(mesh1, 'CG', 1)
    V2 = FunctionSpace(mesh2, 'CG', 2)

    Ts, TTs, b = PointConstraintData(relations=[(0, 1, np.array([0., 1.]))], 
                                     spaces=[V1, V2])

    print [T.array() for T in Ts],
    print [TT.array() for TT in TTs]
    print b
