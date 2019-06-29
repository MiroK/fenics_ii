from xii import EmbeddedMesh
import dolfin as df
import numpy as np

function collides{D, T}(line::Segment{D, T}, box::BoundingBox{D, T})
    x, y = line.A
    dx = line.B[1] - x
    dy = line.B[2] - y

    p = [-dx, dx, -dy, dy]
    q = [x-box.low[1], box.high[1]-x, y-box.low[2], box.high[2]-y]

    u1 = -Inf
    u2 = Inf
    for i in 1:4
        (p[i] == 0 && q[i] < 0) && return false

        t = q[i]/p[i]

        if (p[i] < 0 && u1 < t)
            u1 = max(0, t)
        elseif (p[i] > 0 && u2 > t)
            u2 = min(1, t)
        end
    end

   (u1 > u2) && return false

    true
end

def collides_seg(line, seg):
    pass

def collides_seg(tri, seg):
    pass



def CutSurfaceMesh(bckg, cut, scale=0.5, tol=1E-13):
    '''
    Mesh of facets of bckg which can be viewed as an approximation 
    of a virtual (cylinderical) surface around the 1d cut mesh.
    '''
    assert cut.topology().dim() == 1
    assert bckg.topology().dim() > 1

    tdim = bckg.topology().dim()
    fdim = tdim - 1

    cell_f = df.MeshFunction('size_t', bckg, tdim, 0)
    cell_f_values = cell_f.array()
    # The idea here is to first get bckg cells that are intersected (in
    # a sense that they contain cut vertices).
    tree = bckg.bounding_box_tree()
    limit = bckg.num_cells()
    # Color the intersected ones
    for x in cut.coordinates():
        cs = tree.compute_entity_collisions(df.Point(x))
        cs = filter(lambda c: c < limit, cs)
        cell_f_values[cs] = 1
    File('bar.pvd') << cell_f
    cut_bckg = SubMesh(bckg, cell_f, 1)
        
    # Next we make mesh out of the facets of the intersected cells
    cut_bckg.init(tdim, fdim)
    c2f = cut_bckg.topology()(tdim, fdim)

    # Keep track of boundary facets and facets to be used for the skeleton
    bdry = df.MeshFunction('size_t', cut_bckg, fdim, 0)
    bdry_values = bdry.array()
    bdry_values[np.hstack(map(c2f, range(cut_bckg.num_cells())))] = 2

    # The full one containing interesected cells
    skeleton = EmbeddedMesh(bdry, 2)

    # Next based on actually computing the collisions

    # cell_f = MeshFunction('size_t', skeleton, fdim, 1)
    # # Remove those
    # for x in cut.coordinates():
    #     c = tree.compute_first_entity_collision(df.Point(x))
    #     if c >= limit: continue
        
    #     cell_f[c] = 0

    # skeleton = SubMesh(skeleton, cell_f, 1)
    # File('foo.pvd') << skeleton
    # # As an 
    # cell_f = df.MeshFunction('size_t', bckg, tdim, 0)
    # cell_values = cell_f.array()
    # bdry_map = defaultdict(set)

    # df.File('foo.pvd') << cell_f
    # # Get the intersected volume mesh
    # volume = df.SubMesh(bckg, cell_f, 1)
    # # The surface is then
    # surface = BoundaryMesh(volume, 'exterior')
    
    # # Now we would like to remove the facets that were on the boundary
    # # So we need to establish corrospondence surface cells -> cells of
    # # volume -> cells of bckg -> facet of bckg
    # volume.init(fdim, tdim)
    # f2c = volume.topology()(fdim, tdim)
    # # Get first from suface cells to cells of volume
    # sc2vv = np.fromiter([f2c(f)[0] for f in surface.entity_map(fdim).array()],
    #                     dtype='uintp')
    # # Now we are get to cells of bckg
    # sc2vb = volume.data().array('parent_cell_indices', tdim)[sc2vv]
    # # Use geometry predicate to establish the correspondence between facets
    # sc2bf = np.zeros(surface.num_cells(), dtype='uintp')
    # # At the same time we can figure out which cut cells to remove
    # cell_f = MeshFunction('size_t', surface, surface.topology().dim(), 1)
    
    # for cut_cell, bckg_vol in enumerate(sc2vb):
    #     x = df.Cell(surface, cut_cell).midpoint()

    #     found = False
    #     facets = iter(c2f(bckg_vol))
    #     while not found:
    #         bckg_facet = next(facets)
    #         found = df.Facet(bckg, bckg_facet).midpoint().distance(x) < tol
    #     assert found

    #     sc2bf[cut_cell] = bckg_facet

    #     if bckg_facet in bdry_map: cell_f[cut_cell] = 0

    # cut_mesh = SubMesh(surface, cell_f, 1)
    # # Attache the mapiing of cut_mesh cells to facets ...
    # cut_mesh.facet_map = sc2bf[cut_mesh.data().array('parent_cell_indices', fdim)]

    # return cut_mesh
    
# --------------------------------------------------------------------

if __name__ == '__main__':
    from dolfin import *
    from xii import EmbeddedMesh
    
    bckg = UnitSquareMesh(32, 20)
    # Make the cut as x = 0.5
    mesh = RectangleMesh(Point(0.5, 0), Point(1, 1), 5, 100)
    facet_f = MeshFunction('size_t', mesh, 1, 0)
    CompiledSubDomain('near(x[0], 0.5)').mark(facet_f, 1)

    cut = EmbeddedMesh(facet_f, 1)

    #print point_cloud(cut, bckg)
    f = CutSurfaceMesh(bckg, cut)
    #print f.num_cells()
    
    # File('foo.pvd') << f

    # mapping = f.facet_map
    # # Sanity of the map
    # #assert all(cell.midpoint().distance(Facet(bckg, mapping[i]).midpoint()) < 1E-13
    # #           for i, cell in enumerate(cells(f)))
