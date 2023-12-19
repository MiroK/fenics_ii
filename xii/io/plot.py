# VTK for visualizing `pi3 install --user vtk`
from xii.assembler.average_shape import render_avg_surface, tube_render_avg_surface
    
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkQuad, vtkPolyData, vtkTriangle, vtkLine
from vtk.util.numpy_support import numpy_to_vtk
from vtk import vtkPolyDataWriter
import itertools, os
import numpy as np

import vtk


def vtk_tube(avg_op, path):
    '''Render averaging surface as tube network'''
    assert path
    _, ext = os.path.splitext(path)
    assert ext == '.vtk'
    
    tubes = tube_render_avg_surface(avg_op)
    data = vtk_tube_render(tubes)

    writer = vtkPolyDataWriter()
    writer.SetInputData(data)
    writer.SetFileName(path)
    writer.Write()

    return path


def vtk_tube_render(tubes):
    '''Join the integration circles for segments'''
    first = 0
    points, quads = [], []
    for ring0, ring1 in tubes:
        points.extend(ring0)
        idx0 = np.r_[np.arange(first, first+len(ring0)), first]
        first += len(ring0)

        points.extend(ring1)        
        idx1 = np.r_[np.arange(first, first+len(ring1)), first]
        first += len(ring1)

        indices = np.row_stack([idx1, idx0])
        nrows, ncols = indices.shape
        for j in range(ncols-1):
            quads.append((indices[0, j], indices[0, j+1], indices[1, j+1], indices[1, j]))

    vtk_points = vtkPoints()
    [vtk_points.InsertNextPoint(*x) for x in points]    

    facetsPolyData = vtkPolyData()    
    # Add the points to the polydata container
    facetsPolyData.SetPoints(vtk_points)
    
    facets = vtkCellArray()
    # Lines
    for vs in quads:
        facet = vtkQuad()        
        [facet.GetPointIds().SetId(loc, vi) for (loc, vi) in enumerate(vs)]
        facets.InsertNextCell(facet)
    facetsPolyData.SetPolys(facets)

    VTK_data = numpy_to_vtk(num_array=np.ones(facets.GetNumberOfCells()), deep=True, array_type=vtk.VTK_FLOAT)
    facetsPolyData.GetCellData().SetScalars(VTK_data)
        
    return facetsPolyData
    

def vtk_plot_data(surfaces):
    '''Simple plot of f on marked subdomains. One value per facet!'''
    points = []
    for surface in surfaces:
        circle_points = radial_order_points(surface)
        points.append(circle_points)
    offsets = list(map(len, points))
    offsets = np.r_[0, np.cumsum(offsets)]

    vtk_points = vtkPoints()
    [vtk_points.InsertNextPoint(*x) for x in itertools.chain(*points)]    

    facetsPolyData = vtkPolyData()    
    # Add the points to the polydata container
    facetsPolyData.SetPoints(vtk_points)


    facets = vtkCellArray()
    vtkFacet, nvertices_per_facet = vtkLine, 2
    # Lines
    for first, last in zip(offsets[:-1], offsets[1:]):
        for v0 in range(first, last-1):
            facet = vtkFacet()        
            facet.GetPointIds().SetId(0, v0)
            facet.GetPointIds().SetId(1, v0+1)
            facets.InsertNextCell(facet)
        facet = vtkFacet()                    
        facet.GetPointIds().SetId(0, v0+1)
        facet.GetPointIds().SetId(1, first)
        facets.InsertNextCell(facet)            
    facetsPolyData.SetPolys(facets)

    VTK_data = numpy_to_vtk(num_array=np.ones(facets.GetNumberOfCells()), deep=True,
                            array_type=vtk.VTK_FLOAT)
    facetsPolyData.GetCellData().SetScalars(VTK_data)
        
    return facetsPolyData


def radial_order_points(points):
    '''By angle'''
    npts, gdim = points.shape
    assert gdim == 3
    assert npts > 3
    
    com = np.mean(points, axis=0)

    v0, v1, v2 = points[:3]
    t0, t1 = v1 - v0, v2 - v0

    normal = np.cross(t0, t1)
    normal = normal/np.linalg.norm(normal)
    assert all(abs(np.dot(p - com, normal)) < 1E-10 for p in points), [abs(np.dot(p - com, normal)) for p in points]
    
    P = np.eye(3) - np.outer(normal, normal)
    eigvals, eigvecs = np.linalg.eigh(P)
    assert abs(eigvals[0]) < 1E-13

    t0, t1 = eigvecs[:, 1], eigvecs[:, 2]

    # 2d coordinates of the points
    u, v = np.array([[np.dot(p, t0), np.dot(p, t1)] for p in points]).T

    angles = np.angle(u + v*1j, deg=True)
    idx = np.argsort(angles)
    return points[idx]
