from dolfin import *
import networkx as nx
from random import choice


def random_path(mesh):
    '''Random curve through mesh.'''
    graph_nodes = range(mesh.num_vertices())

    mesh.init(1, 0)
    graph_edges = (tuple(map(int, edge.entities(0))) for edge in edges(mesh))

    G = nx.Graph()
    G.add_nodes_from(graph_nodes)
    G.add_edges_from(graph_edges)

    def get_node(visited):
        n = choice(graph_nodes)
        if n in visited:
            return get_node(visited)
        else:
            return n

    visited = set([])
    paths = []
    start = None
    for i in range(10):
        if start is None:
            start = get_node(visited)
            visited.add(start)

        stop = get_node(visited)
        visited.add(stop)

        path = nx.shortest_path(G, start, stop)

        paths.extend(path)
        start = stop

    mesh.init(0, 1)
    v2e = mesh.topology()(0, 1)
    e2v = mesh.topology()(1, 0)
    edge_path = []
    for v0, v1 in zip(paths[:-1], paths[1:]):
        target = set([v0, v1])
        for e in v2e(v0):
            if set(map(int, e2v(e))) == target:
                edge_path.append(e)
                break

    edge_f = EdgeFunction('size_t', mesh, 0)
    for edge in edge_path: edge_f[int(edge)] = 1

    return edge_f

# ----------------------------------------------------------------------------

if __name__ == '__main__':
    mesh = UnitCubeMesh(10, 10, 20)

    edge_f = lightning_path(mesh)
    plot(edge_f)
    interactive()
