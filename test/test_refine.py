from xii.meshing.refinement import (centroid_refine, circumcenter_refine,
                                    point_is_inside)
from xii.meshing.dual_mesh import DualMesh

try:
    from gmshnics import gUnitSquare
    has_gmshnics = True
except:
    has_gmshnics = False

from dolfin import *
import numpy as np
import pytest

strategies = (centroid_refine,
              lambda mesh: centroid_refine(mesh, 2))

@pytest.mark.parametrize('refine_method', strategies)
def test_vertices(refine_method):
    # Every vertex in coarse is in the refine one'''
    mesh = UnitSquareMesh(5, 3)

    _do_test_vertices(mesh, refine_method)

    
def _do_test_vertices(mesh, refine_method):
    rmesh = refine_method(mesh)
    mapping = rmesh.parent_entity_map
    
    assert mesh.id() in mapping

    mapping, = mapping.values()

    vertex_map = mapping[0]

    new_idx, old_idx = map(list, zip(*vertex_map.items()))

    assert np.linalg.norm(rmesh.coordinates()[new_idx] - mesh.coordinates()[old_idx]) < 1E-13    

    
@pytest.mark.parametrize('refine_method', strategies)
def test_cells(refine_method):
    # Centroid of each child is inside the right parent
    mesh = UnitSquareMesh(5, 8)

    _do_test_cells(mesh, refine_method)

    
def _do_test_cells(mesh, refine_method):
    x = mesh.coordinates()
    cells = x[mesh.cells()]

    rmesh = refine_method(mesh)
    rx = rmesh.coordinates()
    rcells = rx[rmesh.cells()]
    
    mapping = rmesh.parent_entity_map
    
    assert mesh.id() in mapping

    mapping, = mapping.values()

    cell_map = mapping[mesh.topology().dim()]

    for cid, pid in cell_map.items():
        child_mid = np.mean(rcells[cid], axis=0)

        pcell = cells[pid]
        assert point_is_inside(child_mid, pcell, 1E-8)


@pytest.mark.skipif(not has_gmshnics, reason='gmshnics not found')
@pytest.mark.parametrize('refine_method', (circumcenter_refine, lambda mesh: circumcenter_refine(mesh, 2)))
def test_cells_centroid(refine_method):
    # Centroid of each child is inside the right parent
    mesh = gUnitSquare(0.5)[0]

    _do_test_cells(mesh, refine_method)


@pytest.mark.skipif(not has_gmshnics, reason='gmshnics not found')
@pytest.mark.parametrize('refine_method', (circumcenter_refine, lambda mesh: circumcenter_refine(mesh, 2)))
def test_vertices_centroid(refine_method):
    # Centroid of each child is inside the right parent
    mesh = gUnitSquare(0.5)[0]

    _do_test_vertices(mesh, refine_method)
