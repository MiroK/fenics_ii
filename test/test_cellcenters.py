from xii.meshing.cell_centers import (CellCentroid, CellCircumcenter, CircumVector,
                                      CircumDistance, CircumDistance2,
                                      CentroidDistance, CentroidDistance2)
import matplotlib.pyplot as plt
import dolfin as df
import numpy as np
import gmshnics
import pytest


def test_centroid_2d(_tol=1E-12):
    mesh = df.UnitSquareMesh(32, 32)
    
    c = CellCentroid(mesh)
    V = c.function_space()
    dm = V.dofmap()
    coefs = c.vector().get_local()
    
    for cell in df.cells(mesh):
        x = np.array(cell.get_vertex_coordinates()).reshape((-1, 2))
        their = x.mean(axis=0)

        dof = dm.cell_dofs(cell.index())
        mine = coefs[dof]
        assert np.linalg.norm(their - mine) < _tol


def test_centroid_3d(_tol=1E-12):
    mesh = df.UnitCubeMesh(8, 8, 8)
    
    c = CellCentroid(mesh)
    V = c.function_space()
    dm = V.dofmap()
    coefs = c.vector().get_local()
    
    for cell in df.cells(mesh):
        x = np.array(cell.get_vertex_coordinates()).reshape((-1, 3))
        their = x.mean(axis=0)

        dof = dm.cell_dofs(cell.index())
        mine = coefs[dof]
        assert np.linalg.norm(their - mine) < _tol

# --------------------------------------------------------------------

@pytest.mark.parametrize('mesh, ', (lambda: df.UnitSquareMesh(2, 2), lambda: gmshnics.gUnitSquare(0.4)[0]))
def test_circumcenter_2d(mesh, _tol=1E-10, plot=False):
    mesh = mesh()
    mesh.init(2, 1)
    mesh.init(1, 0)

    c = CellCircumcenter(mesh)

    centers, mps = [], []
    for cell in df.cells(mesh):
        center = c(cell.midpoint())
        centers.append(center)
        
        for facet in df.facets(cell):
            mp = facet.midpoint().array()[:2]
            mps.append(mp)
            
            nu = mp - center
            size = np.sqrt(np.dot(nu, nu))
            if size > _tol:
                nu = nu/size
                    
            n = facet.normal().array()[:2]

            assert size < _tol or abs(abs(np.dot(n, nu))-1) < 100*_tol
    centers = np.array(centers)
    mps = np.array(mps)

    if plot:
        df.plot(mesh)
        plt.scatter(mps[:, 0], mps[:, 1], marker='x', color='red')                
        plt.scatter(centers[:, 0], centers[:, 1], marker='x', color='red')            

        plt.show()


@pytest.mark.parametrize('mesh, ', (lambda: df.UnitCubeMesh(2, 2, 2), lambda: gmshnics.gUnitCube(0.6)[0]))        
def test_circumcenter_3d(mesh, _tol=1E-10):
    '''Should be equally far for the vertices'''
    mesh = mesh()
    c = CellCircumcenter(mesh)

    for cell in df.cells(mesh):
        center = c(cell.midpoint())

        x = np.array(cell.get_vertex_coordinates()).reshape((-1, 3))
        d = np.linalg.norm(x - center, 2, axis=1)
        assert np.linalg.norm(d - d[0], np.inf) < _tol

# ---

@pytest.mark.parametrize('mesh, ', (lambda: df.UnitSquareMesh(2, 2),
                                    lambda: gmshnics.gUnitSquare(0.4)[0],
                                    lambda: df.UnitCubeMesh(2, 2, 3),
                                    lambda: gmshnics.gUnitCube(0.8)[0]))
def test_centroid_distances(mesh, _tol=1E-10, plot=False):
    mesh = mesh()
    a = CentroidDistance(mesh)
    b = CentroidDistance2(mesh)

    print(np.c_[a.vector().get_local(), b.vector().get_local()])
    L = df.inner(df.avg(a-b), df.avg(a-b))*df.dS + df.inner(a-b, a-b)*df.ds
    e = df.sqrt(abs(df.assemble(L)))

    assert e < _tol

# --------------------------------------------------------------------


# FIXME: 
#        weighted distance as in ...

#        check 3d
#        Is there better accuracy with Neumann where there are no bdry facets?
#        Use in DG

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    # test_centroid_2d()
    # test_centroid_3d()

    # test_circumcenter_2d(mesh=df.UnitSquareMesh(4, 4), plot=True)
    # test_circumcenter_3d()
    mesh, _ = gmshnics.gUnitSquare(0.4)
    # print(is_delaunay(mesh))
